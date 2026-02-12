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

### Baseline mode for LongMemEval (no tools)
Use `--task longmemeval` to include provided memory sessions (`haystack_sessions`, dates, IDs) in the prompt for each question.

### Solver modes
- `--solver cmd` (default): runs `--solver-cmd` template with `{question}` / `{prompt}`
- `--solver gemini`: uses Gemini directly with `GEMINI_API_KEY`

### Example: LongMemEval baseline run with Gemini
```bash
export GEMINI_API_KEY="<your_key>"
python scripts/run_baseline.py \
  --task longmemeval \
  --solver gemini \
  --gemini-model gemini-2.0-flash \
  --hf-dataset xiaowu0162/longmemeval-cleaned \
  --hf-file longmemeval_oracle.json \
  --output-dir results/longmemeval_cleaned_baseline \
  --timeout 120
```

### Outputs
- `predictions.jsonl`
- `metrics.json`

`metrics.json` includes strict and relaxed accuracy:
- `strict_accuracy`: lowercase+whitespace normalized exact match
- `relaxed_accuracy`: punctuation-insensitive with basic acceptable-variant parsing

The runner logs question-level progress to stdout:
- `[idx/total] start id=...`
- `[idx/total] done id=... latency_s=... strict=... relaxed=...`
