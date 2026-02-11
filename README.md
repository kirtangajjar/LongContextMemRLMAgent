# LongContextMemRLMAgent

## Simple baseline runner

Use `scripts/run_baseline.py` to run a minimal baseline over a JSONL benchmark.

### Input format
Each JSONL row should include:
- `question` (required)
- `id` (optional)
- `answer` or `gold` or `target` (optional, used for exact-match scoring)

### Example
```bash
python scripts/run_baseline.py \
  --input data/benchmark.jsonl \
  --output-dir results/baseline \
  --solver-cmd "python my_solver.py --question {question}" \
  --timeout 60
```

Outputs:
- `predictions.jsonl`
- `metrics.json`

If no gold answers are present, the script still writes predictions and latency metrics.
