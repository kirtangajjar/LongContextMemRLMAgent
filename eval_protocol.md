# Baseline replication protocol (current)

This protocol is for the **baseline-only** replication pass (no GEPA/Pro/tool-ablation variants yet).

## Dataset
- Hugging Face dataset: `xiaowu0162/longmemeval-cleaned`
- File: `longmemeval_oracle.json`
- Questions: 500

## Solver configuration
- Runner: `scripts/run_baseline.py`
- Task formatter: `--task longmemeval`
- Solver mode: `--solver rlm` (iterative RLM-style loop)
- Model (pinned default): `gemini-3-flash-preview`
- Temperature: `0.0`
- Max iterative steps: `3`
- Per-question timeout: `120s`

## Prompting/scoring
- Context includes question + haystack sessions + session IDs + dates + question date.
- Scoring reported as:
  - `strict_accuracy` (basic normalized exact match)
  - `relaxed_accuracy` (punctuation-insensitive with acceptable-variant parsing)

## Reproducibility metadata
- Runner writes protocol details under `metrics.json -> protocol`.
- Trace includes per-question progress and per-item correctness markers.
