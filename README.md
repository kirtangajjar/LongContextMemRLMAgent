# LongContextMemRLMAgent

## Baseline benchmark runner

Use `scripts/run_baseline.py` for baseline evaluation.

### Supported inputs
- Local JSONL via `--input`
- Hugging Face dataset JSON file via `--hf-dataset` + `--hf-file`

Each row should include:
- `question` (required)
- `id` or `question_id` (optional)
- `answer` or `gold` or `target` (optional, used for scoring)

### Baseline mode for LongMemEval (no GEPA/Pro/tools)
Use `--task longmemeval` to include memory context (`haystack_sessions`, session IDs, dates, question date).

### Solver modes
- `--solver cmd` (default): command template mode
- `--solver gemini`: one-pass Gemini
- `--solver rlm`: iterative RLM-style loop (baseline focus)

### Example: baseline-focused RLM-style run
```bash
export GEMINI_API_KEY="<your_key>"
python scripts/run_baseline.py \
  --task longmemeval \
  --solver rlm \
  --gemini-model gemini-3-flash-preview \
  --temperature 0.0 \
  --rlm-max-steps 3 \
  --hf-dataset xiaowu0162/longmemeval-cleaned \
  --hf-file longmemeval_oracle.json \
  --output-dir results/longmemeval_cleaned_baseline_rlm \
  --timeout 120
```

### Outputs
- `predictions.jsonl`
- `metrics.json`

`metrics.json` includes:
- strict/relaxed accuracy
- average steps (for iterative solver)
- protocol metadata (`protocol` object)

Progress trace format:
- `[idx/total] start id=...`
- `[idx/total] done id=... latency_s=... steps=... strict=... relaxed=...`
