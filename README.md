# LongContextMemRLMAgent

## Baseline benchmark runner

Use `scripts/run_baseline.py` for simple, single-turn baseline evaluation.

### Supported inputs
- Local JSONL via `--input`
- Hugging Face dataset JSON file via `--hf-dataset` + `--hf-file`

Each row should include:
- `question` (required)
- `id` (optional)
- `answer` or `gold` or `target` (optional, used for exact-match scoring)

### Solver modes
- `--solver cmd` (default): runs `--solver-cmd` template with `{question}`
- `--solver gemini`: uses Gemini directly with `GEMINI_API_KEY`

### Example: local JSONL with command solver
```bash
python scripts/run_baseline.py \
  --input data/benchmark.jsonl \
  --output-dir results/baseline_cmd \
  --solver cmd \
  --solver-cmd "python my_solver.py --question {question}" \
  --timeout 60
```

### Example: LongMemEval from Hugging Face with Gemini
```bash
export GEMINI_API_KEY="<your_key>"
python scripts/run_baseline.py \
  --solver gemini \
  --gemini-model gemini-2.0-flash \
  --hf-dataset xiaowu0162/longmemeval-cleaned \
  --hf-file longmemeval_oracle.json \
  --output-dir results/longmemeval_cleaned_oracle_full \
  --timeout 120
```

### Outputs
- `predictions.jsonl`
- `metrics.json`

The runner logs question-level progress in stdout as:
- `[idx/total] start id=...`
- `[idx/total] done id=... latency_s=...`
