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


def to_int(v: str) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def load_case_ids(path: Optional[Path]) -> Optional[Set[str]]:
    if path is None:
        return None
    return {x.strip() for x in path.read_text().splitlines() if x.strip()}


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize pipeline latency + token usage for Direct/CAP/CAP+Event.")
    p.add_argument("--render-case-metrics", type=Path, required=True)
    p.add_argument("--problem-state-case-metrics", type=Path, required=True)
    p.add_argument("--case-ids-file", type=Path, default=None)
    p.add_argument("--template", default="sectioned")
    p.add_argument("--output-csv", type=Path, required=True)
    args = p.parse_args()

    allowed = load_case_ids(args.case_ids_file)
    c_stage: Dict[str, Dict[str, float]] = {}
    with args.problem_state_case_metrics.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("case_id", "")
            if not cid:
                continue
            if allowed is not None and cid not in allowed:
                continue
            c_stage[cid] = {
                "c_runtime": to_float(row.get("transcript_cap_runtime_seconds", "0")),
                "c_prompt": to_int(row.get("transcript_cap_prompt_tokens", "0")),
                "c_completion": to_int(row.get("transcript_cap_completion_tokens", "0")),
            }

    rows = []
    with args.render_case_metrics.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("template") != args.template:
                continue
            if row.get("method") not in {"direct", "cap", "cap_event"}:
                continue
            cid = row.get("case_id", "")
            if allowed is not None and cid not in allowed:
                continue
            rows.append(row)

    by_method: Dict[str, Dict[str, list]] = {
        "direct": {"lat": [], "in_tok": [], "out_tok": []},
        "cap": {"lat": [], "in_tok": [], "out_tok": []},
        "cap_event": {"lat": [], "in_tok": [], "out_tok": []},
    }
    for r in rows:
        m = r["method"]
        cid = r.get("case_id", "")
        y_runtime = to_float(r.get("summary_runtime_seconds", "0"))
        y_prompt = to_int(r.get("llm_prompt_tokens", "0"))
        y_completion = to_int(r.get("llm_completion_tokens", "0"))
        e_runtime = to_float(r.get("event_plan_runtime_seconds", "0"))
        e_prompt = to_int(r.get("event_plan_prompt_tokens", "0"))
        e_completion = to_int(r.get("event_plan_completion_tokens", "0"))
        c = c_stage.get(cid, {"c_runtime": 0.0, "c_prompt": 0, "c_completion": 0})

        if m == "direct":
            lat = y_runtime
            in_tok = y_prompt
            out_tok = y_completion
        elif m == "cap":
            lat = c["c_runtime"] + y_runtime
            in_tok = c["c_prompt"] + y_prompt
            out_tok = c["c_completion"] + y_completion
        else:  # cap_event
            lat = c["c_runtime"] + e_runtime + y_runtime
            in_tok = c["c_prompt"] + e_prompt + y_prompt
            out_tok = c["c_completion"] + e_completion + y_completion

        by_method[m]["lat"].append(lat)
        by_method[m]["in_tok"].append(in_tok)
        by_method[m]["out_tok"].append(out_tok)

    direct_lat = mean(by_method["direct"]["lat"]) if by_method["direct"]["lat"] else 0.0
    direct_in = mean(by_method["direct"]["in_tok"]) if by_method["direct"]["in_tok"] else 0.0
    direct_out = mean(by_method["direct"]["out_tok"]) if by_method["direct"]["out_tok"] else 0.0
    direct_total = direct_in + direct_out

    labels = {
        "direct": "Direct (X->Y)",
        "cap": "CAP (X->C->Y)",
        "cap_event": "CAP+Event (X->C->E->Y)",
    }

    out = []
    for key in ("direct", "cap", "cap_event"):
        lats = by_method[key]["lat"]
        in_toks = by_method[key]["in_tok"]
        out_toks = by_method[key]["out_tok"]
        if not lats:
            continue
        lat_avg = mean(lats)
        in_avg = mean(in_toks) if in_toks else 0.0
        out_avg = mean(out_toks) if out_toks else 0.0
        total_avg = in_avg + out_avg
        out.append(
            {
                "pipeline": labels[key],
                "method_key": key,
                "template": args.template,
                "n_cases": len(lats),
                "avg_latency_s": f"{lat_avg:.4f}",
                "avg_total_tokens_input": f"{in_avg:.2f}",
                "avg_total_tokens_output": f"{out_avg:.2f}",
                "avg_total_tokens_all": f"{total_avg:.2f}",
                "latency_relative_vs_direct": f"{(lat_avg / direct_lat) if direct_lat > 0 else 0.0:.4f}",
                "input_tokens_relative_vs_direct": f"{(in_avg / direct_in) if direct_in > 0 else 0.0:.4f}",
                "output_tokens_relative_vs_direct": f"{(out_avg / direct_out) if direct_out > 0 else 0.0:.4f}",
                "total_tokens_relative_vs_direct": f"{(total_avg / direct_total) if direct_total > 0 else 0.0:.4f}",
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        fields = [
            "pipeline",
            "method_key",
            "template",
            "n_cases",
            "avg_latency_s",
            "avg_total_tokens_input",
            "avg_total_tokens_output",
            "avg_total_tokens_all",
            "latency_relative_vs_direct",
            "input_tokens_relative_vs_direct",
            "output_tokens_relative_vs_direct",
            "total_tokens_relative_vs_direct",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"[OK] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
