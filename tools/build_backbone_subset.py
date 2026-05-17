#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def load_cap_sectioned_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("method") == "cap" and row.get("template") == "sectioned":
                rows.append(row)
    if not rows:
        raise RuntimeError("No cap/sectioned rows found in case_metrics.csv")
    return rows


def to_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def stratify(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    # Use semantic_cap_f1 as the main stratification axis.
    sorted_rows = sorted(rows, key=lambda r: to_float(r.get("semantic_cap_f1", "0")))
    n = len(sorted_rows)
    i1 = n // 3
    i2 = (2 * n) // 3
    low = sorted_rows[:i1]
    mid = sorted_rows[i1:i2]
    high = sorted_rows[i2:]
    return low, mid, high


def sample_bucket(rows: List[Dict[str, str]], k: int, rng: random.Random) -> List[Dict[str, str]]:
    if k <= 0:
        return []
    if len(rows) <= k:
        return list(rows)
    return rng.sample(rows, k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stratified subset of case IDs for backbone sensitivity runs.")
    parser.add_argument(
        "--case-metrics",
        type=Path,
        default=Path("results/template_rendering_full207_v11_main_ablation_eval/case_metrics.csv"),
    )
    parser.add_argument("--subset-size", type=int, default=30, help="Total target size (recommended: 24 or 30).")
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--output-dir", type=Path, default=Path("results/subset_backbone_sensitivity"))
    args = parser.parse_args()

    if args.subset_size < 6:
        raise ValueError("subset-size must be >= 6 for 3-way stratification.")

    rows = load_cap_sectioned_rows(args.case_metrics)
    low, mid, high = stratify(rows)
    rng = random.Random(args.seed)

    per_bucket = args.subset_size // 3
    remainder = args.subset_size % 3
    bucket_sizes = [per_bucket, per_bucket, per_bucket]
    for i in range(remainder):
        bucket_sizes[i] += 1

    chosen = []
    chosen.extend(sample_bucket(low, bucket_sizes[0], rng))
    chosen.extend(sample_bucket(mid, bucket_sizes[1], rng))
    chosen.extend(sample_bucket(high, bucket_sizes[2], rng))

    # Keep stable ordering by case id for reproducibility in downstream commands.
    chosen = sorted(chosen, key=lambda r: r["case_id"])
    case_ids = [r["case_id"] for r in chosen]

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"case_ids_n{len(case_ids)}_seed{args.seed}.txt"
    meta_path = out_dir / f"subset_meta_n{len(case_ids)}_seed{args.seed}.json"

    txt_path.write_text("\n".join(case_ids) + "\n")
    meta = {
        "subset_size": len(case_ids),
        "seed": args.seed,
        "source_case_metrics": str(args.case_metrics),
        "stratification_metric": "semantic_cap_f1 (cap, sectioned)",
        "bucket_sizes": {
            "low": bucket_sizes[0],
            "mid": bucket_sizes[1],
            "high": bucket_sizes[2],
        },
        "case_ids": case_ids,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[OK] wrote {txt_path}")
    print(f"[OK] wrote {meta_path}")


if __name__ == "__main__":
    main()
