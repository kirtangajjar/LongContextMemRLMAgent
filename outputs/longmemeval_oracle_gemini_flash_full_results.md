# LongMemEval oracle run results (Gemini 2.0 Flash, rerun with trace)

- Dataset: `data/longmemeval_oracle.json`
- Predictions file: `outputs/longmemeval_oracle_gemini_flash_full.jsonl`
- Trace file: `outputs/longmemeval_oracle_gemini_flash_full_trace.jsonl`
- Total questions run: 500
- Total API attempts: 511
- Questions needing retry: 11
- Max attempts for a single question: 2
- HTTP status counts: {200: 500, 429: 11}

## Metrics (string-based sanity checks)
- Exact match (normalized): 119/500 = 0.2380
- Relaxed contains match (normalized substring either direction): 217/500 = 0.4340
- Abstention heuristic accuracy (`_abs` only): 20/30 = 0.6667

## By question type

| question_type | n | exact_match | relaxed_contains |
|---|---:|---:|---:|
| knowledge-update | 78 | 0.3846 | 0.6154 |
| multi-session | 133 | 0.0977 | 0.1729 |
| single-session-assistant | 56 | 0.5536 | 0.8214 |
| single-session-preference | 30 | 0.0000 | 0.0000 |
| single-session-user | 70 | 0.4714 | 0.7857 |
| temporal-reasoning | 133 | 0.0902 | 0.3383 |

## 429 diagnosis
- Simple single-prompt curl calls can succeed while benchmark calls fail because benchmark prompts are much larger and consume far more tokens per minute/request.
- The trace confirms quota/rate behavior was mitigated by lower request pacing and retry backoff: only a small number of retries occurred in this rerun.
