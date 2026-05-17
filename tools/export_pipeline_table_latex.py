#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Export pipeline latency/token CSV to LaTeX table.")
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-tex", type=Path, required=True)
    p.add_argument("--caption", default="Latency and token-cost proxy on a 30-case stratified subset.")
    p.add_argument("--label", default="tab:latency_token_subset")
    args = p.parse_args()

    rows = []
    with args.input_csv.open() as f:
        r = csv.DictReader(f)
        rows = list(r)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Pipeline & Avg latency (s) & Rel. latency & Avg input toks & Avg output toks & Avg total toks & Rel. total toks \\")
    lines.append(r"\midrule")
    for row in rows:
        pipeline = row["pipeline"]
        lat = row["avg_latency_s"]
        rlat = row["latency_relative_vs_direct"]
        in_tok = row["avg_total_tokens_input"]
        out_tok = row["avg_total_tokens_output"]
        total_tok = row["avg_total_tokens_all"]
        rtotal_tok = row["total_tokens_relative_vs_direct"]
        lines.append(f"{pipeline} & {lat} & {rlat} & {in_tok} & {out_tok} & {total_tok} & {rtotal_tok} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{args.caption}}}")
    lines.append(rf"\label{{{args.label}}}")
    lines.append(r"\end{table}")

    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text("\n".join(lines) + "\n")
    print(f"[OK] wrote {args.output_tex}")


if __name__ == "__main__":
    main()
