# CAP-ACL-2026

Code and artifacts for the BioNLP 2026 paper:

**CAP: A Source-Grounded Proposition Scaffold for Faithful Clinical Dialogue-to-Note Generation**

This work was conducted at AITRICS.

This repository is organized so that readers can:
- Inspect the end-to-end pipeline implementation (`X -> C -> E -> Y`).
- Inspect prompt-based reimplementations of baselines (Direct, Cluster2Sent-inspired, MEDSUM-ENT-inspired).
- Reproduce paper tables/figures from the included precomputed outputs without rerunning LLM inference.

## Repository Layout

- `code/`: Main runner scripts and shared utilities.
- `prompts/`: Selected prompt text files; most prompts are defined as Python string constants in `code/`.
- `results/`: Precomputed outputs (CAP extraction + rendering/evaluation) used to build the quantitative tables and figures.
- `examples/`: Small qualitative bundles used in the paper (generated notes for selected cases).
- `data/`: No raw benchmark data is redistributed here; see `data/README.md`.

## Quickstart (Reproducing Tables/Figures From Included Results)

The precomputed results are stored under:
- `results/problem_state_tracking_full207_v11_shared_gemma3/`
- `results/template_rendering_full207_v11_main_ablation_eval/`

Key files:
- `aggregate_metrics.csv`
- `case_metrics.csv`

## Running The Pipeline (Optional)

Running the full pipeline requires access to LLM endpoints (local or hosted) and API keys.
We recommend using explicit `--api-base-url` / `--judge-api-base-url` arguments rather than relying on any defaults.

Main entrypoints:
- `code/run_problem_state_tracking_experiments.py` (CAP extraction)
- `code/run_template_rendering_experiments.py` (note rendering + evaluation)
- `tools/run_deployment_cost_subset_e2e.py` (deployment-path latency/token analysis on subset: `X->Y`, `X->C->Y`, `X->C->E->Y`)

Example (deployment-path cost analysis on a 30-case subset):
```bash
python3 tools/run_deployment_cost_subset_e2e.py --subset-size 30 --seed 57 --force-regenerate
```

## Prompts

Prompts are implemented as Python constants inside the runner scripts (search for `PROMPT` in `code/`).
For reviewer convenience, we export prompt snapshots to `prompts/` with an index:
- `prompts/INDEX.md`

To regenerate these snapshots from `code/`:
```bash
python3 tools/export_prompts.py
```

## Notes

This repository intentionally excludes:
- `.env` files / API keys
- personal identifiers
- legacy backups and experimental scratch files
