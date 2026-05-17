#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, Optional, Set


def to_float(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def load_case_ids(path: Optional[Path]) -> Optional[Set[str]]:
    if path is None:
        return None
    return {x.strip() for x in path.read_text().splitlines() if x.strip()}


def load_c_runtime_map(path: Path, allowed: Optional[Set[str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["case_id"]
            if allowed is not None and cid not in allowed:
                continue
            out[cid] = to_float(row.get("transcript_cap_runtime_seconds", "0"))
    return out


def load_generation_rows(path: Path, allowed: Optional[Set[str]], template: str):
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("template") != template:
                continue
            cid = row.get("case_id", "")
            if allowed is not None and cid not in allowed:
                continue
            rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize end-to-end pipeline latency for Direct / CAP / CAP+Event.")
    p.add_argument("--render-case-metrics", type=Path, required=True, help="case_metrics.csv from run_template_rendering_experiments.py (generation-only recommended)")
    p.add_argument("--cap-case-metrics", type=Path, required=True, help="case_metrics.csv from run_problem_state_tracking_experiments.py")
    p.add_argument("--case-ids-file", type=Path, default=None)
    p.add_argument("--template", default="sectioned")
    p.add_argument("--output-csv", type=Path, required=True)
    args = p.parse_args()

    allowed = load_case_ids(args.case_ids_file)
    c_runtime = load_c_runtime_map(args.cap_case_metrics, allowed)
    render_rows = load_generation_rows(args.render_case_metrics, allowed, args.template)

    by_method: Dict[str, list[float]] = {"direct": [], "cap": [], "cap_event": []}

    for row in render_rows:
        method = row.get("method", "")
        if method not in by_method:
            continue
        cid = row.get("case_id", "")
        y_runtime = to_float(row.get("summary_runtime_seconds", "0"))
        e_runtime = to_float(row.get("event_plan_runtime_seconds", "0"))
        c = c_runtime.get(cid, 0.0)

        if method == "direct":
            total = y_runtime  # X -> Y
        elif method == "cap":
            total = c + y_runtime  # X -> C -> Y
        else:  # cap_event
            total = c + e_runtime + y_runtime  # X -> C -> E -> Y
        by_method[method].append(total)

    direct_mean = mean(by_method["direct"]) if by_method["direct"] else 0.0
    rows_out = []
    labels = {
        "direct": "Direct (X->Y)",
        "cap": "CAP (X->C->Y)",
        "cap_event": "CAP+Event (X->C->E->Y)",
    }
    for k in ("direct", "cap", "cap_event"):
        vals = by_method[k]
        if not vals:
            continue
        avg = mean(vals)
        rel = (avg / direct_mean) if direct_mean > 0 else 0.0
        rows_out.append(
            {
                "pipeline": labels[k],
                "method_key": k,
                "template": args.template,
                "n_cases": len(vals),
                "avg_end_to_end_latency_s": f"{avg:.4f}",
                "relative_vs_direct": f"{rel:.4f}",
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        fieldnames = [
            "pipeline",
            "method_key",
            "template",
            "n_cases",
            "avg_end_to_end_latency_s",
            "relative_vs_direct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[OK] wrote {args.output_csv}")


if __name__ == "__main__":
    main()

