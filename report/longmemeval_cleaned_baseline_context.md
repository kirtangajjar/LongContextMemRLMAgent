# LongMemEval cleaned baseline-only run (no tools)

## Objective
Implement the baseline-focused portion of the replication plan without GEPA/Pro/tools.

## What changed
- Runner now supports `--task longmemeval` to build a context-aware prompt from:
  - `haystack_sessions`
  - `haystack_session_ids`
  - `haystack_dates`
  - `question_date`
- Scoring now reports:
  - `strict_accuracy` (basic normalized exact match)
  - `relaxed_accuracy` (punctuation-insensitive + simple acceptable-variant parsing)

## Full run command
```bash
GEMINI_API_KEY='<set>' python scripts/run_baseline.py \
  --task longmemeval \
  --solver gemini \
  --gemini-model gemini-2.0-flash \
  --hf-dataset xiaowu0162/longmemeval-cleaned \
  --hf-file longmemeval_oracle.json \
  --output-dir results/longmemeval_cleaned_baseline_context \
  --timeout 120 > trace/longmemeval_cleaned_baseline_context_run.log 2>&1
```

## Result
- Total: 500
- Strict correct: 70 / 500 (**0.1400**)
- Relaxed correct: 191 / 500 (**0.3820**)
- Avg latency: 0.5637 s

## Notes vs tweet target
- This is a substantial improvement over the previous one-shot no-context run (0.014 strict).
- It is still below the reported DSPy.rLM baseline (87.2%), which indicates additional gap from not running full DSPy.rLM agent behavior yet.

## Artifacts
- Metrics: `results/longmemeval_cleaned_baseline_context/metrics.json`
- Predictions: `results/longmemeval_cleaned_baseline_context/predictions.jsonl`
- Trace: `trace/longmemeval_cleaned_baseline_context_run.log`
