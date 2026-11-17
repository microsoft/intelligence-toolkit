"""Schemify CLI: research, audit, verify-review, dedup, schemas and dashboards."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"
DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"


def _cmd_run(args: argparse.Namespace) -> int:
    """Run a discovery + extraction job from a JSON config."""
    import asyncio

    from . import Schemify, SchemifyConfig

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    category = cfg["category"]
    guidance = cfg.get("guidance", "")
    max_queries = int(cfg.get("max_queries", 100))
    concurrency = int(cfg.get("concurrency", 5))
    phase_split = tuple(cfg.get("phase_split", [0.6, 0.2, 0.2]))
    output_dir = Path(cfg.get("output_dir") or args.out or f"output/{cfg_path.stem}")
    prior_dataset = cfg.get("prior_dataset")
    verify = cfg.get("verify", True)
    complete = cfg.get("complete", False)
    schema_attributes = cfg.get("schema_attributes")  # optional override
    # Names-only seeding: re-research known entities from scratch (no old values carried over).
    seed_entities = None
    if cfg.get("seed_entities"):
        seed_data = json.loads(Path(cfg["seed_entities"]).read_text(encoding="utf-8"))
        seed_rows = seed_data.get("records", []) if isinstance(seed_data, dict) else seed_data
        seed_entities = [
            {"label": r.get("label") or r.get("name"), "aliases": list(r.get("aliases") or [])}
            for r in seed_rows if (r.get("label") or r.get("name"))
        ]

    # SchemifyConfig: api_key from env unless config overrides it, plus any other knobs
    sc_kwargs = dict(cfg.get("config", {}))
    sc_kwargs.setdefault("api_key", os.environ.get("OPENAI_API_KEY", ""))
    if not sc_kwargs["api_key"]:
        print("error: OPENAI_API_KEY not set (and no api_key in config.config)", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] category={category!r}")
    print(f"[run] output_dir={output_dir}")
    if prior_dataset:
        print(f"[run] prior_dataset={prior_dataset}")
    if seed_entities:
        print(f"[run] seed_entities={len(seed_entities)} names (values re-researched)")
    print(f"[run] max_queries={max_queries} concurrency={concurrency} verify={verify} complete={complete}")

    async def _go():
        schemify = Schemify(SchemifyConfig(**sc_kwargs))
        # If schema_attributes provided, build SchemaAttribute instances
        sa = None
        if schema_attributes:
            from .models import SchemaAttribute
            sa = [SchemaAttribute(**a) for a in schema_attributes]
        await schemify.initialize(category=category, guidance=guidance, schema_attributes=sa)
        await schemify.run_agentic(
            max_queries=max_queries,
            concurrency=concurrency,
            output_dir=str(output_dir),
            seed_state=prior_dataset,
            seed_records=seed_entities,
            phase_split=phase_split,
        )
        if complete:
            await schemify.complete_all(concurrency=concurrency, verbose=True)
            schemify.save(str(output_dir / "after_completion.json"))
        if verify:
            await schemify.verify_unverified(concurrency=concurrency, output_dir=str(output_dir))
        schemify.finalize(output_dir=str(output_dir))

    asyncio.run(_go())

    # Normalize to repo's dataset convention: copy final.json → data.json
    final = output_dir / "final.json"
    target = output_dir / "data.json"
    if final.exists():
        shutil.copy(final, target)
        print(f"[run] wrote {target}")
    else:
        print(f"warning: {final} missing; finalize may have failed", file=sys.stderr)
        return 2
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from .audit import run_audit

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    policy = Path(args.policy).read_text(encoding="utf-8") if args.policy else None
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else PROMPTS_DIR
    result = run_audit(
        data,
        prompts_dir=prompts_dir,
        policy=policy,
        backend=args.backend,
        model=args.model,
        subagent_command=args.subagent_command,
    )

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    json_path.write_text(result.to_json(), encoding="utf-8")
    md_path.write_text(result.to_markdown(), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


def _cmd_verify_review(args: argparse.Namespace) -> int:
    import asyncio
    import hashlib
    import logging

    from .review_verification import load_proposals, make_client, verify_proposals

    logging.basicConfig(level=logging.INFO)
    data_path = Path(args.data)
    raw_data = data_path.read_bytes()
    data = json.loads(raw_data)
    proposal_path = Path(args.proposals)
    content = proposal_path.read_text(encoding="utf-8-sig")
    if proposal_path.suffix.lower() == ".json":
        envelope = json.loads(content)
        expected = envelope.get("source_hashes", {}).get(data_path.name) if isinstance(envelope, dict) else None
        if expected and hashlib.sha256(raw_data).hexdigest() != expected:
            msg = "Proposal source hash does not match the supplied dataset file"
            raise ValueError(msg)
    proposals = load_proposals(content, proposal_path.suffix.lstrip("."))
    if args.record_ids:
        selected_ids = {int(s.strip()) for s in args.record_ids.split(",")}
        if selected_ids - {p["record_id"] for p in proposals}:
            msg = "Selected record IDs are not all present in the proposal file"
            raise ValueError(msg)
        proposals = [p for p in proposals if p["record_id"] in selected_ids]
    client = make_client(args.backend, model=args.model)

    def progress(report: dict) -> None:
        summary = report["summary"]
        print(
            f"[verify-review] {summary['completed_records']}/{summary['total_records']} "
            f"records; checks={summary['check_statuses']}; errors={summary['processing_errors']}",
            flush=True,
        )

    report = asyncio.run(verify_proposals(
        data, proposals, Path(args.out), client,
        concurrency=args.concurrency, max_sources=args.max_sources,
        resume=args.resume, retry_failed_sources=args.retry_failed_sources,
        backend=args.backend, on_progress=progress,
    ))
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != hashlib.sha256(raw_data).hexdigest():
        msg = "Dataset file changed during verification; do not use these results as current"
        raise ValueError(msg)
    print(f"[verify-review] report: {Path(args.out) / 'report.json'}; dataset not modified")
    return 2 if report["summary"]["processing_errors"] else 0


def _cmd_propose_schema(args: argparse.Namespace) -> int:
    from .propose import run_schema_proposal

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    constraints = Path(args.constraints).read_text(encoding="utf-8")
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else PROMPTS_DIR
    target_attrs = args.attrs.split(",") if args.attrs else None

    proposal = run_schema_proposal(
        data,
        constraints=constraints,
        target_attrs=target_attrs,
        prompts_dir=prompts_dir,
        backend=args.backend,
        model=args.model,
        subagent_command=args.subagent_command,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8")
    issues = proposal.get("validation_issues") or []
    print(f"wrote {out}")
    print(f"  schema: {len(proposal.get('schema', []))} attrs")
    print(f"  record_remappings: {len(proposal.get('record_remappings', []))}")
    print(f"  out_of_scope_records: {len(proposal.get('out_of_scope_records', []))}")
    print(f"  unresolved: {len(proposal.get('unresolved', []))}")
    if issues:
        print(f"  validation_issues: {issues}", file=sys.stderr)
        return 2
    return 0


def _cmd_recategorize(args: argparse.Namespace) -> int:
    from .audit import apply_recategorization

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    proposal = json.loads(Path(args.proposal).read_text(encoding="utf-8"))
    out = apply_recategorization(data, proposal)
    Path(args.out).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {args.out} ({len(out.get('records', []))} records, "
          f"{len(out.get('schema_attributes', []))} attributes)")
    return 0


def _cmd_build_dashboard(args: argparse.Namespace) -> int:
    """Bundle dashboard.html + theme + data into a standalone dist folder."""
    data_path = Path(args.data)
    theme_dir = Path(args.theme)
    out_dir = Path(args.out)
    dashboard_dir = Path(args.dashboard_dir) if args.dashboard_dir else DASHBOARD_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(dashboard_dir / "dashboard.html", out_dir / "dashboard.html")

    # Theme files: theme.js (required) plus any assets in the theme dir
    if not (theme_dir / "theme.js").exists():
        print(f"error: {theme_dir}/theme.js not found", file=sys.stderr)
        return 1
    for item in theme_dir.iterdir():
        if item.is_file():
            shutil.copy(item, out_dir / item.name)

    # Dataset → dashboard_data.js
    data = json.loads(data_path.read_text(encoding="utf-8"))
    js = "const DASHBOARD_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n"
    (out_dir / "dashboard_data.js").write_text(js, encoding="utf-8")

    print(f"built {out_dir}/")
    return 0


def _dedup_group_key(cluster) -> str:
    import hashlib

    payload = json.dumps([cluster.reason, sorted(r.label for r in cluster.records)], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cmd_dedup_audit(args: argparse.Namespace) -> int:
    """Find duplicate records and misattached aliases; never modifies the dataset."""
    import asyncio
    import csv
    import hashlib
    from datetime import datetime, timezone

    from . import dedup_audit as da
    from .models import RecordSet
    from .review_verification import make_client

    data_path = Path(args.data)
    raw = data_path.read_bytes()
    rs = RecordSet.from_dict(json.loads(raw))
    out = Path(args.out)
    groups_dir = out / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)

    clusters = da.find_candidate_clusters(
        rs, max_cluster=args.max_cluster, min_token_overlap=args.min_token_overlap,
    )
    if args.limit:
        clusters = clusters[: args.limit]
    done: dict[str, da.DedupAuditResult] = {}
    pending = []
    for cluster in clusters:
        path = groups_dir / f"{_dedup_group_key(cluster)}.json"
        if path.exists():
            result = da.DedupAuditResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if result.error is None:
                done[_dedup_group_key(cluster)] = result
                continue
        pending.append(cluster)
    print(f"[dedup-audit] {len(clusters)} candidate groups; {len(done)} reused; {len(pending)} to review", flush=True)

    if pending:
        client = make_client(args.backend, model=args.model)

        def on_result(cluster, result) -> None:
            key = _dedup_group_key(cluster)
            (groups_dir / f"{key}.json").write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            done[key] = result

        def progress(i: int, n: int) -> None:
            if i == n or i % 10 == 0:
                print(f"[dedup-audit] reviewed {i}/{n}", flush=True)

        asyncio.run(da.audit_clusters(
            rs, client, clusters=pending, concurrency=args.concurrency,
            on_result=on_result, progress_cb=progress,
        ))

    results = [done[_dedup_group_key(c)] for c in clusters if _dedup_group_key(c) in done]
    errors = [r for r in results if r.error]
    rows = da.proposal_rows(results, confidence_threshold=args.confidence)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": data_path.name,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "backend": args.backend,
        "confidence_threshold": args.confidence,
        "candidate_groups": len(clusters),
        "errors": len(errors),
        "proposals": {
            "merge": sum(1 for r in rows if r["kind"] == "merge"),
            "remove_alias": sum(1 for r in rows if r["kind"] == "remove_alias"),
        },
        "results": [r.to_dict() for r in results],
    }
    (out / "dedup_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "proposals.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=da.PROPOSAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"[dedup-audit] {report['proposals']['merge']} merges and "
        f"{report['proposals']['remove_alias']} alias removals proposed; "
        f"{len(errors)} groups failed. Review {out / 'proposals.csv'} (edit 'approve'), "
        f"then run dedup-apply. Dataset not modified."
    )
    return 2 if errors else 0


def _cmd_dedup_apply(args: argparse.Namespace) -> int:
    """Apply approved dedup-audit proposals to a copy of the dataset."""
    import csv
    from datetime import datetime

    from . import dedup_audit as da
    from .models import RecordSet

    data_path, out_path = Path(args.data), Path(args.out)
    if data_path.resolve() == out_path.resolve():
        print("error: --out must differ from the input dataset", file=sys.stderr)
        return 1
    data = json.loads(data_path.read_text(encoding="utf-8"))
    rs = RecordSet.from_dict(data)
    originals = {id(rec): raw for rec, raw in zip(rs.records, data.get("records", []))}
    names_before = {id(rec): (rec.label, list(rec.aliases), dict(rec.alias_counts)) for rec in rs.records}
    applied_at = datetime.now().isoformat(timespec="seconds")

    cleaned: list[dict] = []
    if not args.no_clean_aliases:
        for rec in rs.records:
            for old, new in rec.clean_aliases():
                cleaned.append({"label": rec.label, "old": old, "new": new})
        if cleaned:
            rs.history.append({
                "op": "clean_aliases", "applied_at": applied_at,
                "changes": cleaned, "source": "duplicate audit",
            })

    proposals_path = Path(args.proposals)
    if proposals_path.suffix.lower() == ".json":
        rows = json.loads(proposals_path.read_text(encoding="utf-8"))
    else:
        with proposals_path.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
    primaries: list = []
    statuses = da.apply_proposal_rows(rs, rows, applied_at=applied_at, primaries=primaries)
    merged = {id(rec) for rec in primaries}

    # Untouched records are written back verbatim; name-only edits patch
    # the original record so unrelated fields are not re-serialised.
    out_records = []
    for rec in rs.records:
        raw = originals.get(id(rec))
        if raw is None or id(rec) in merged:
            out_records.append(rec.to_dict())
            continue
        if (rec.label, list(rec.aliases), dict(rec.alias_counts)) != names_before[id(rec)]:
            raw = {**raw, "label": rec.label, "aliases": list(rec.aliases), "alias_counts": dict(rec.alias_counts)}
        out_records.append(raw)
    out = {**data, "records": out_records, "history": list(rs.history)}
    original_dnm = {frozenset(p) for p in data.get("do_not_merge") or [] if len(p) == 2}
    if set(rs.do_not_merge or set()) != original_dnm:
        out["do_not_merge"] = [sorted(p) for p in rs.do_not_merge]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = Path(args.report) if args.report else out_path.with_name(out_path.stem + ".applied.csv")
    with report_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["proposal_id", "kind", "status", "label", "detail"])
        writer.writeheader()
        writer.writerows(statuses)
        for c in cleaned:
            writer.writerow({"proposal_id": "", "kind": "clean_alias", "status": "applied",
                             "label": c["label"], "detail": f"{c['old']} -> {c['new'] or '(removed)'}"})
    applied = [s for s in statuses if s["status"] == "applied"]
    print(
        f"[dedup-apply] {len(data.get('records', []))} -> {len(out_records)} records; "
        f"{sum(1 for s in applied if s['kind'] == 'merge')} merges, "
        f"{sum(1 for s in applied if s['kind'] == 'remove_alias')} alias removals, "
        f"{len(cleaned)} alias clean-ups; {len(statuses) - len(applied)} skipped."
    )
    print(f"[dedup-apply] wrote {out_path} and {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="schemify", description="Schemify CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    prun = sub.add_parser("run", help="Run a discovery + extraction job from a JSON config.")
    prun.add_argument("config", help="Path to run config JSON")
    prun.add_argument("--out", help="Override output_dir from the config")
    prun.set_defaults(func=_cmd_run)

    pa = sub.add_parser("audit", help="Audit a dataset for quality issues.")
    pa.add_argument("data", help="Path to data.json")
    pa.add_argument("--out", required=True, help="Output path prefix (.json/.md added)")
    pa.add_argument("--policy", help="Optional policy text file")
    pa.add_argument("--backend", default="openai", choices=["openai", "subagent"],
                    help="openai = direct API; subagent = pipe to a CLI on stdin")
    pa.add_argument("--model", default="gpt-5.2", help="Model name (openai backend only)")
    pa.add_argument("--subagent-command",
                    help="CLI invocation for the subagent backend "
                         "(default: $SCHEMIFY_SUBAGENT_CMD or 'claude -p')")
    pa.add_argument("--prompts-dir", help="Override prompts directory")
    pa.set_defaults(func=_cmd_audit)

    pr = sub.add_parser("recategorize", help="Apply a schema proposal to a dataset.")
    pr.add_argument("data", help="Path to data.json")
    pr.add_argument("--proposal", required=True, help="Path to schema_proposal.json")
    pr.add_argument("--out", required=True, help="Output dataset path")
    pr.set_defaults(func=_cmd_recategorize)

    ps = sub.add_parser("propose-schema",
                        help="Propose tightened taxonomies + per-record remappings for a dataset.")
    ps.add_argument("data", help="Path to data.json")
    ps.add_argument("--constraints", required=True,
                    help="Path to a text file describing desired schema changes")
    ps.add_argument("--out", required=True, help="Output schema_proposal.json path")
    ps.add_argument("--attrs",
                    help="Comma-separated attribute names to tighten (default: all closed-set attrs)")
    ps.add_argument("--backend", default="openai", choices=["openai", "subagent"],
                    help="openai = direct API; subagent = pipe to a CLI on stdin")
    ps.add_argument("--model", default="gpt-5.2", help="Model name (openai backend only)")
    ps.add_argument("--subagent-command",
                    help="CLI invocation for the subagent backend "
                         "(default: $SCHEMIFY_SUBAGENT_CMD or 'claude -p')")
    ps.add_argument("--prompts-dir", help="Override prompts directory")
    ps.set_defaults(func=_cmd_propose_schema)

    pb = sub.add_parser("build-dashboard", help="Bundle a shippable dashboard folder.")
    pb.add_argument("--data", required=True, help="Path to data.json")
    pb.add_argument("--theme", required=True, help="Theme directory (contains theme.js)")
    pb.add_argument("--out", required=True, help="Output dist directory")
    pb.add_argument("--dashboard-dir", help="Override dashboard source directory")
    pb.set_defaults(func=_cmd_build_dashboard)

    pv = sub.add_parser("verify-review", help="Verify proposed corrections against fresh public sources; never apply them.")
    pv.add_argument("data", help="Exact dataset JSON to review")
    pv.add_argument("--proposals", required=True, help="Review CSV or JSON file")
    pv.add_argument("--out", required=True, help="New evidence/report directory (or existing directory with --resume)")
    pv.add_argument("--backend", choices=["openai", "copilot"], default="openai")
    pv.add_argument("--model", help="OpenAI model only; Copilot uses the authenticated account's defaults")
    pv.add_argument("--concurrency", type=int, default=4)
    pv.add_argument("--max-sources", type=int, default=12)
    pv.add_argument("--record-ids", help="Optional comma-separated zero-based record IDs")
    pv.add_argument("--resume", action="store_true", help="Explicitly reuse this run's original timestamped capture/checkpoints")
    pv.add_argument("--retry-failed-sources", action="store_true",
                    help="With --resume, retry failed URLs, preserve attempt history and reassess affected records")
    pv.set_defaults(func=_cmd_verify_review)

    pd = sub.add_parser("dedup-audit",
                        help="Find duplicate records and misattached aliases; never modifies the dataset.")
    pd.add_argument("data", help="Dataset JSON to audit")
    pd.add_argument("--out", required=True, help="Output directory (rerun to resume from per-group checkpoints)")
    pd.add_argument("--backend", choices=["openai", "copilot"], default="openai")
    pd.add_argument("--model", help="OpenAI model only; Copilot uses the authenticated account's defaults")
    pd.add_argument("--concurrency", type=int, default=4)
    pd.add_argument("--confidence", type=float, default=0.7,
                    help="Pre-approve proposals at or above this confidence in proposals.csv")
    pd.add_argument("--max-cluster", type=int, default=12, help="Maximum records per reviewed group")
    pd.add_argument("--min-token-overlap", type=int, default=2,
                    help="Shared label/alias/description words needed to group records of the same "
                         "organisation; 0 reviews all of an organisation's records together")
    pd.add_argument("--limit", type=int, help="Review only the first N candidate groups (smoke test)")
    pd.set_defaults(func=_cmd_dedup_audit)

    pda = sub.add_parser("dedup-apply", help="Apply approved dedup-audit proposals to a copy of the dataset.")
    pda.add_argument("data", help="Dataset JSON the proposals were generated from")
    pda.add_argument("--proposals", required=True, help="proposals.csv (or JSON list of rows) from dedup-audit")
    pda.add_argument("--out", required=True, help="Output dataset path (must differ from the input)")
    pda.add_argument("--report", help="Applied-changes CSV (default: <out>.applied.csv)")
    pda.add_argument("--no-clean-aliases", action="store_true",
                     help="Skip removing placeholder aliases and repairing truncated brackets")
    pda.set_defaults(func=_cmd_dedup_apply)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
