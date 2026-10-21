"""Bounded, fresh retrieval of public HTML, text and PDF evidence."""

from __future__ import annotations

import asyncio
import hashlib
import http.client
import ipaddress
import re
import socket
import ssl
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from io import BytesIO
from urllib.parse import quote, urljoin, urlsplit

from pypdf import PdfReader
from pypdf.errors import PyPdfError


@dataclass
class RetrievedSource:
    url: str
    retrieved_at: str
    status: str
    final_url: str = ""
    text: str = ""
    content_type: str = ""
    content_sha256: str = ""
    http_status: int | None = None
    truncated: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript", "svg"):
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript", "svg") and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def public_addresses(url: str) -> tuple[str, int, list[str]]:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        msg = "Evidence URLs must use public HTTP(S)"
        raise ValueError(msg)
    if parsed.username or parsed.password or parsed.port not in (None, 80, 443):
        msg = "Credentials and nonstandard ports are not allowed in evidence URLs"
        raise ValueError(msg)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    addresses = list(
        dict.fromkeys(
            str(row[4][0])
            for row in socket.getaddrinfo(
                parsed.hostname, port, type=socket.SOCK_STREAM
            )
        )
    )
    resolved = [ipaddress.ip_address(address) for address in addresses]
    if not resolved or any(
        not address.is_global or address.is_multicast for address in resolved
    ):
        msg = "Evidence URLs must resolve exclusively to public addresses"
        raise ValueError(msg)
    return parsed.hostname, port, addresses


def extract_text(
    raw: bytes, content_type: str, charset: str = "utf-8"
) -> tuple[str, bool]:
    truncated = False
    if content_type == "application/pdf":
        reader = PdfReader(BytesIO(raw))
        truncated = len(reader.pages) > 200
        text = " ".join(page.extract_text() or "" for page in reader.pages[:200])
    elif content_type in ("text/html", "application/xhtml+xml", "text/plain"):
        text = raw.decode(charset, errors="replace")
        if content_type != "text/plain":
            parser = _TextParser()
            parser.feed(text)
            text = " ".join(parser.parts)
    else:
        msg = f"Unsupported evidence content type: {content_type}"
        raise ValueError(msg)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 100:
        msg = "Source has too little readable text; it may require a browser"
        raise ValueError(msg)
    return text[:250_000], truncated or len(text) > 250_000


class PublicSourceRetriever:
    """Connect only to validated public IPs, preserving the TLS hostname."""

    def __init__(self, timeout: float = 25.0) -> None:
        self.timeout = timeout

    async def retrieve(self, url: str) -> RetrievedSource:
        return await asyncio.to_thread(self.retrieve_sync, url)

    def retrieve_sync(self, url: str) -> RetrievedSource:
        retrieved_at = datetime.now(timezone.utc).isoformat()
        current = url
        http_status = None
        try:
            for _ in range(6):
                hostname, port, addresses = public_addresses(current)
                parsed = urlsplit(current)
                context = (
                    ssl.create_default_context() if parsed.scheme == "https" else None
                )
                connection = (
                    http.client.HTTPSConnection(
                        hostname, port, timeout=self.timeout, context=context
                    )
                    if context is not None
                    else http.client.HTTPConnection(
                        hostname, port, timeout=self.timeout
                    )
                )
                raw_socket = None
                last_error = None
                for address in sorted(addresses, key=lambda a: ":" in a):
                    try:
                        # Connect to the validated address, not a second DNS lookup.
                        raw_socket = socket.create_connection(
                            (address, port), timeout=self.timeout
                        )
                        break
                    except OSError as exc:
                        last_error = exc
                if raw_socket is None:
                    msg = f"Could not connect to a public address: {last_error}"
                    raise OSError(msg)  # noqa: TRY301
                try:
                    connection.sock = (
                        context.wrap_socket(raw_socket, server_hostname=hostname)
                        if context is not None
                        else raw_socket
                    )
                    target = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
                    if parsed.query:
                        target += "?" + quote(
                            parsed.query, safe="=&%/:?@!$'()*+,;+-._~"
                        )
                    connection.request(
                        "GET",
                        target,
                        headers={
                            "User-Agent": "ITK-Evidence-Review/1.0",
                            "Accept-Encoding": "identity",
                            "Cache-Control": "no-cache",
                        },
                    )
                    response = connection.getresponse()
                    http_status = response.status
                    if response.status in (301, 302, 303, 307, 308):
                        location = response.headers.get("Location")
                        if not location:
                            msg = "Redirect has no Location header"
                            raise ValueError(msg)
                        current = urljoin(current, location)
                        continue
                    if response.status != 200:
                        msg = f"HTTP {response.status}: {response.reason}"
                        raise ValueError(msg)
                    content_type = response.headers.get_content_type()
                    limit = (
                        8_000_000 if content_type == "application/pdf" else 2_000_000
                    )
                    raw = response.read(limit + 1)
                    if len(raw) > limit:
                        msg = f"Source exceeds the {limit} byte retrieval limit"
                        raise ValueError(msg)
                    text, truncated = extract_text(
                        raw,
                        content_type,
                        response.headers.get_content_charset() or "utf-8",
                    )
                    return RetrievedSource(
                        url=url,
                        final_url=current,
                        retrieved_at=retrieved_at,
                        status="retrieved",
                        text=text,
                        content_type=content_type,
                        content_sha256=hashlib.sha256(raw).hexdigest(),
                        http_status=response.status,
                        truncated=truncated,
                    )
                finally:
                    connection.close()
                    raw_socket.close()
            msg = "Source exceeded the redirect limit"
            raise ValueError(msg)  # noqa: TRY301
        except (
            OSError,
            ValueError,
            http.client.HTTPException,
            PyPdfError,
            LookupError,
        ) as exc:
            return RetrievedSource(
                url=url,
                final_url=current,
                retrieved_at=retrieved_at,
                status="failed",
                http_status=http_status,
                error=str(exc),
            )
