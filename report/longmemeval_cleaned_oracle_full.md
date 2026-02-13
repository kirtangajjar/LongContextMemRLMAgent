# LongMemEval-Cleaned Oracle Full Run Report

## Run summary
- Dataset: `xiaowu0162/longmemeval-cleaned`
- File: `longmemeval_oracle.json`
- Solver: Gemini (`gemini-2.0-flash`)
- Questions evaluated: 500
- Correct: 7
- Accuracy: **0.0140** (7/500)
- Average latency per question: 0.5438s

## Command used
```bash
GEMINI_API_KEY='<set>' python scripts/run_baseline.py \
  --solver gemini \
  --gemini-model gemini-2.0-flash \
  --hf-dataset xiaowu0162/longmemeval-cleaned \
  --hf-file longmemeval_oracle.json \
  --output-dir results/longmemeval_cleaned_oracle_full \
  --timeout 120 > trace/longmemeval_cleaned_oracle_full_run.log 2>&1
```

## Artifacts
- Metrics: `results/longmemeval_cleaned_oracle_full/metrics.json`
- Predictions: `results/longmemeval_cleaned_oracle_full/predictions.jsonl`
- Run trace: `trace/longmemeval_cleaned_oracle_full_run.log`
