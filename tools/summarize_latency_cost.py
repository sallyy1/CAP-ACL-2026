#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


def to_float(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def load_case_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    return ids or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize latency/cost proxy stats from case_metrics.csv")
    parser.add_argument("--case-metrics", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--template", default="sectioned")
    parser.add_argument("--methods", nargs="+", default=["direct", "cap"])
    parser.add_argument("--case-ids-file", type=Path, default=None)
    args = parser.parse_args()

    allowed_ids = load_case_ids(args.case_ids_file)
    by_method: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)

    with args.case_metrics.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("method", "")
            template = row.get("template", "")
            case_id = row.get("case_id", "")
            if method not in args.methods or template != args.template:
                continue
            if allowed_ids is not None and case_id not in allowed_ids:
                continue
            s = to_float(row.get("summary_runtime_seconds", "0"))
            c = to_float(row.get("summary_cap_runtime_seconds", "0"))
            e = to_float(row.get("event_plan_runtime_seconds", "0"))
            by_method[method].append((s, c, e))

    rows_out = []
    base_total = None
    if by_method.get("direct"):
        base_total = mean([s + c + e for s, c, e in by_method["direct"]])

    for method in args.methods:
        vals = by_method.get(method, [])
        if not vals:
            continue
        s_avg = mean([v[0] for v in vals])
        c_avg = mean([v[1] for v in vals])
        e_avg = mean([v[2] for v in vals])
        total = s_avg + c_avg + e_avg
        rel = (total / base_total) if base_total and base_total > 0 else 0.0
        rows_out.append(
            {
                "method": method,
                "template": args.template,
                "n_cases": len(vals),
                "avg_summary_runtime_s": f"{s_avg:.4f}",
                "avg_summary_cap_runtime_s": f"{c_avg:.4f}",
                "avg_event_plan_runtime_s": f"{e_avg:.4f}",
                "avg_total_runtime_s": f"{total:.4f}",
                "relative_vs_direct": f"{rel:.4f}",
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        fieldnames = [
            "method",
            "template",
            "n_cases",
            "avg_summary_runtime_s",
            "avg_summary_cap_runtime_s",
            "avg_event_plan_runtime_s",
            "avg_total_runtime_s",
            "relative_vs_direct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[OK] wrote {args.output_csv}")


if __name__ == "__main__":
    main()

