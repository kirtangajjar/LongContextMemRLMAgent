# LongContextMemRLMAgent

## Quickstart: download a benchmark + run baseline agent

This repo now includes a minimal end-to-end path:
1. Download a benchmark split from Hugging Face into JSONL
2. Run baseline evaluation with one model call per question (no tools)

## 1) Install dependencies

```bash
pip install datasets google-genai
```

## 2) Download benchmark to JSONL

Use `scripts/download_benchmark.py`.

Example (GSM8K test subset of 100 rows):

```bash
python scripts/download_benchmark.py \
  --dataset gsm8k \
  --config main \
  --split test \
  --question-field question \
  --answer-field answer \
  --max-rows 100 \
  --output data/gsm8k_test_100.jsonl
```

## 3) Run baseline on Gemini Flash

Set your API key:

```bash
export GEMINI_API_KEY=...your_key...
```

Then run:

```bash
python scripts/run_baseline.py \
  --input data/gsm8k_test_100.jsonl \
  --output-dir results/gemini_flash_baseline \
  --solver-cmd "python scripts/gemini_solver.py --question {question} --model gemini-2.0-flash" \
  --timeout 90
```

## Scripts

### `scripts/download_benchmark.py`
Downloads any Hugging Face dataset split and writes a compact JSONL with:
- `id`
- `question`
- `answer` (if available)

### `scripts/run_baseline.py`
Runs minimal baseline eval over JSONL rows:
- one solver command per question
- exact-match scoring when `answer`/`gold`/`target` exists
- writes `predictions.jsonl` and `metrics.json`

### `scripts/gemini_solver.py`
Single-question Gemini client used by `run_baseline.py`.
- requires `GEMINI_API_KEY`
- defaults to `gemini-2.0-flash`
