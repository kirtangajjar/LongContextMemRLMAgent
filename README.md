# LongContextMemRLMAgent

## LongMemEval benchmark run (Gemini 2.0 Flash)

This repository now contains a full rerun of LongMemEval oracle with benchmark trace logging.

### Artifacts

- Predictions: `outputs/longmemeval_oracle_gemini_flash_full.jsonl`
- Benchmark trace: `outputs/longmemeval_oracle_gemini_flash_full_trace.jsonl`
- Results summary: `outputs/longmemeval_oracle_gemini_flash_full_results.md`

### What was fixed

During earlier runs, intermittent `RESOURCE_EXHAUSTED (429)` errors occurred.
A direct curl test can still succeed because it uses a tiny prompt, while LongMemEval benchmark prompts are much larger and can hit per-minute/per-request quota limits.

The rerun was executed with:
- explicit retry/backoff on API errors,
- lower request pacing to reduce token burst rate,
- per-attempt trace logging for post-run diagnosis.
