# Baseline replication progress (no GEPA/Pro/tools)

Implemented items from the baseline gap list:

1. **RLM-style baseline path added**
   - `--solver rlm` now runs an iterative loop (plan/search/read/answer) over LongMemEval memory context.
2. **Model/version pinning**
   - Default model pinned to `gemini-3-flash-preview`.
   - Protocol metadata is emitted in `metrics.json` under `protocol`.
3. **Dataset context usage**
   - `--task longmemeval` builds prompt with question date + haystack sessions + dates + IDs.
4. **Benchmark-aligned scoring (improved)**
   - strict + relaxed scoring retained and enhanced (punctuation-insensitive, acceptable variants, simple range parsing).
5. **Fixed protocol lock**
   - Added `eval_protocol.md` and protocol metadata wiring in runner outputs.

Smoke verification command (completed):
```bash
GEMINI_API_KEY='<set>' python scripts/run_baseline.py \
  --task longmemeval \
  --solver rlm \
  --gemini-model gemini-3-flash-preview \
  --temperature 0.0 \
  --rlm-max-steps 1 \
  --hf-dataset xiaowu0162/longmemeval-cleaned \
  --hf-file longmemeval_oracle.json \
  --max-questions 1 \
  --output-dir /tmp/rlm_smoke \
  --timeout 120
```

Current status: runner capability is now in place for the baseline-focused replication pass.
