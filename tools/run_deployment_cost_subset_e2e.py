#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    pretty = " ".join(shlex.quote(x) for x in cmd)
    print(f"[RUN] {pretty}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def cleanup_render_artifacts(render_out_dir: Path, case_ids: list[str], template: str) -> int:
    removed = 0
    methods = ("direct", "cap", "cap_event")
    targets = (
        render_out_dir / "summaries",
        render_out_dir / "case_results",
        render_out_dir / "summary_caps",
    )
    for cid in case_ids:
        for method in methods:
            stem = f"{cid}_{method}_{template}"
            for base in targets:
                for suffix in (".txt", ".json"):
                    p = base / f"{stem}{suffix}"
                    if p.exists():
                        p.unlink()
                        removed += 1
    return removed


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "End-to-end deployment-cost run on subset: "
            "Direct (X->Y) vs CAP (X->C->Y) vs CAP+Event (X->C->E->Y). "
            "This excludes reference-note CAP extraction by design."
        )
    )
    p.add_argument("--subset-size", type=int, default=30)
    p.add_argument("--seed", type=int, default=57)
    p.add_argument("--cases-path", default="outputs/shared/run_aci_all_v1_realistic/cases.jsonl")
    p.add_argument("--model", default="ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g")
    p.add_argument("--api-key", default=None, help="If omitted, use RUNPOD_API_KEY or OPENAI_API_KEY or 'dummy'.")
    p.add_argument("--request-timeout", type=int, default=240)
    p.add_argument("--template", default="sectioned")
    p.add_argument("--subset-dir", default="results/subset_backbone_sensitivity")
    p.add_argument("--render-out-dir", default=None)
    p.add_argument("--problem-state-out-dir", default=None)
    p.add_argument("--skip-subset-build", action="store_true")
    p.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Delete existing subset artifacts for direct/cap/cap_event before running, to force fresh Y-generation timing/tokens.",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]

    subset_dir = Path(args.subset_dir)
    if not subset_dir.is_absolute():
        subset_dir = root / subset_dir
    subset_dir.mkdir(parents=True, exist_ok=True)

    case_ids_file = subset_dir / f"case_ids_n{args.subset_size}_seed{args.seed}.txt"

    if not args.skip_subset_build:
        run(
            [
                "python3",
                "tools/build_backbone_subset.py",
                "--subset-size",
                str(args.subset_size),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(subset_dir),
            ],
            cwd=root,
        )

    if not case_ids_file.exists():
        raise SystemExit(f"Missing subset file: {case_ids_file}")

    case_ids = [x.strip() for x in case_ids_file.read_text().splitlines() if x.strip()]
    if not case_ids:
        raise SystemExit("No case ids found in subset file.")

    api_key = args.api_key or os.getenv("RUNPOD_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy"

    render_out_dir = args.render_out_dir or f"results/backbone_gemma3_subset{args.subset_size}_{args.template}_direct_cap_capevent"
    ps_out_dir = args.problem_state_out_dir or f"outputs/problem_state_tracking_subset{args.subset_size}_seed{args.seed}_gemma3"
    render_out_path = Path(render_out_dir) if Path(render_out_dir).is_absolute() else root / render_out_dir

    if args.force_regenerate:
        removed = cleanup_render_artifacts(render_out_path, case_ids, args.template)
        print(f"[INFO] force-regenerate: removed {removed} existing subset artifacts under {render_out_path}", flush=True)

    run(
        [
            "python3",
            "code/run_problem_state_tracking_experiments.py",
            "--cases-path",
            args.cases_path,
            "--output-dir",
            ps_out_dir,
            "--case-ids",
            *case_ids,
            "--model",
            args.model,
            "--extractor-model",
            args.model,
            "--api-key",
            api_key,
            "--temperature",
            "0.0",
            "--request-timeout",
            str(args.request_timeout),
            "--deployment-cost-only",
        ],
        cwd=root,
    )

    run(
        [
            "python3",
            "code/run_template_rendering_experiments.py",
            "--cases-path",
            args.cases_path,
            "--problem-state-dir",
            ps_out_dir,
            "--output-dir",
            render_out_dir,
            "--methods",
            "direct",
            "cap",
            "cap_event",
            "--templates",
            args.template,
            "--case-ids",
            *case_ids,
            "--model",
            args.model,
            "--api-key",
            api_key,
            "--temperature",
            "0.0",
            "--request-timeout",
            str(args.request_timeout),
            "--task-workers",
            "1",
            "--case-workers",
            "1",
            "--generation-only",
            "--write-case-files",
        ],
        cwd=root,
    )

    csv_path = f"{render_out_dir}/pipeline_latency_tokens_n{args.subset_size}_seed{args.seed}.csv"
    tex_path = f"{render_out_dir}/pipeline_latency_tokens_n{args.subset_size}_seed{args.seed}.tex"

    run(
        [
            "python3",
            "tools/summarize_pipeline_latency_tokens.py",
            "--render-case-metrics",
            f"{render_out_dir}/case_metrics.csv",
            "--problem-state-case-metrics",
            f"{ps_out_dir}/case_metrics.csv",
            "--case-ids-file",
            str(case_ids_file),
            "--template",
            args.template,
            "--output-csv",
            csv_path,
        ],
        cwd=root,
    )

    run(
        [
            "python3",
            "tools/export_pipeline_table_latex.py",
            "--input-csv",
            csv_path,
            "--output-tex",
            tex_path,
            "--caption",
            f"Offline latency and token-cost proxy on a {args.subset_size}-case stratified subset (paper model, {args.template} template).",
            "--label",
            "tab:latency_token_subset_gemma3",
        ],
        cwd=root,
    )

    print(f"[DONE] render output: {render_out_dir}")
    print(f"[DONE] problem-state output: {ps_out_dir}")
    print(f"[DONE] CSV: {csv_path}")
    print(f"[DONE] LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
