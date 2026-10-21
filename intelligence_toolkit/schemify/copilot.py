"""Tool-free structured completions through an authenticated local Copilot CLI."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from typing import Any

from jsonschema import validate


def completion_from_events(output: str) -> tuple[str, dict]:
    messages = []
    metadata: dict = {}
    finished = False
    for line in output.splitlines():
        event = json.loads(line)
        if event.get("type") == "session.error":
            msg = f"Copilot session error: {event.get('data')}"
            raise RuntimeError(msg)
        if event.get("type") == "assistant.message":
            data = event["data"]
            if data.get("toolRequests"):
                msg = "Tool-free verification unexpectedly requested tools"
                raise RuntimeError(msg)
            if data.get("phase") == "final_answer":
                messages.append(data.get("content", ""))
                metadata["model"] = data.get("model")
        if event.get("type") == "result":
            finished = True
            if event.get("exitCode") != 0:
                msg = "Copilot reported an unsuccessful completion"
                raise RuntimeError(msg)
            metadata["usage"] = event.get("usage", {})
    if not finished or len(messages) != 1:
        msg = "Expected one completed final Copilot message"
        raise ValueError(msg)
    return messages[0], metadata


class CopilotCompletionClient:
    def __init__(self, timeout: float = 600.0) -> None:
        executable = shutil.which("copilot")
        if not executable:
            msg = "Copilot CLI is not installed on this host"
            raise ValueError(msg)
        self.executable = executable
        self.timeout = timeout
        self.calls: list[dict] = []

    async def structured_completion(
        self,
        prompt: str,
        response_format: dict[str, Any],
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        formatted = prompt.format(**variables) if variables is not None else prompt
        schema = response_format["json_schema"]["schema"]
        formatted += "\nReturn only JSON conforming to this schema:\n" + json.dumps(
            schema
        )
        return await asyncio.to_thread(self._complete, formatted, schema)

    def _complete(self, prompt: str, schema: dict) -> dict:
        result = subprocess.run(
            [
                self.executable,
                "--silent",
                "--stream",
                "off",
                "--output-format",
                "json",
                "--available-tools=",
                "--disable-builtin-mcps",
                "--log-level",
                "error",
            ],
            input=prompt,
            capture_output=True,
            encoding="utf-8",
            timeout=self.timeout,
            check=False,
        )
        if result.returncode:
            msg = f"Copilot exited {result.returncode}: {result.stderr[-2000:]}"
            raise RuntimeError(msg)
        text, metadata = completion_from_events(result.stdout)
        self.calls.append(metadata)
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        value = json.loads(text)
        validate(value, schema)
        return value
