"""Schemify CLI: run, audit, propose-schema, recategorize, build-dashboard."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"
DASHBOARD_DIR = REPO_ROOT / "dashboard"


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
    schema_attributes = cfg.get("schema_attributes")  # optional override

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
    print(f"[run] max_queries={max_queries} concurrency={concurrency} verify={verify}")

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
            phase_split=phase_split,
        )
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

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
