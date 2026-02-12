# Plan to replicate the reported DSPy.rLM baseline results

## Goal
Reproduce the posted baseline ladder on the 500-question LongMemEval benchmark, starting with the no-tools DSPy.rLM Flash baseline near **87.2%**.

## Why current repo result is off
Current committed run (`7/500 = 0.014`) is not comparable because:
1. It runs one-shot Gemini QA on just the `question` field.
2. It does not run the DSPy.rLM agent loop (generate/execute/observe/iterate).
3. It ignores LongMemEval context fields such as `haystack_sessions`.
4. It uses naive exact-match scoring that misses acceptable variants.

## Replication plan

### 1) Lock evaluation protocol (before code changes)
- Freeze and document in the repo:
  - Dataset and split/file: `xiaowu0162/longmemeval-cleaned` + `longmemeval_oracle.json`
  - Exact model ID (Gemini Flash version used for replication)
  - Prompt/system instruction
  - Max steps/turns, timeout, retry policy
  - Tool availability by experiment stage
  - Scoring rubric and normalization rules
- Output: protocol section in this file and mirrored metadata in run outputs.

### 2) Implement a true DSPy.rLM baseline mode in `scripts/run_baseline.py`
- Add a solver path that executes DSPy.rLM behavior rather than single-turn completion.
- Keep this in one file (as requested in earlier instructions) unless a split is necessary.
- Preserve existing artifacts (`predictions.jsonl`, `metrics.json`) and progress logging.

### 3) Feed benchmark memory context into the run
- For each question, include needed context from LongMemEval entries:
  - `haystack_sessions`
  - `haystack_session_ids`
  - `haystack_dates`
  - `question_date`
- Ensure the runner can format this context deterministically.

### 4) Add benchmark-aligned scoring
- Replace pure lowercase/whitespace exact match with robust normalization:
  - punctuation/quote-insensitive matching
  - support for acceptable alternates listed in gold text (e.g., "X is also acceptable")
- Track both strict and relaxed metrics for debugging.

### 5) Reproduce baseline first, then ablations
Run in order:
1. **Baseline**: DSPy.rLM Flash, no tools
2. **+tools+prompt**: add `search_sessions`, `grep_sessions`, `get_session` and structured prompt
3. **+obs memory/temporal grounding**
4. **+GEPA**
5. **+Pro swap**

For each stage, save:
- `results/<exp>/metrics.json`
- `results/<exp>/predictions.jsonl`
- `trace/<exp>.log`
- brief `report/<exp>.md`

### 6) Validate deltas and stability
- Run at least 3 seeds if model/runtime permit.
- Compare:
  - absolute accuracy
  - delta vs baseline
  - delta vs previous stage
- Confirm trend matches tweet direction:
  - tools/prompt improves baseline
  - obs memory slightly improves further
  - GEPA/Pro regress relative to best Flash setup

### 7) Error analysis loop
- Diff changed questions between stages.
- Label likely causes (retrieval miss, temporal reasoning miss, scoring mismatch, formatting).
- Use this to tune prompt/tool behavior only after protocol parity is achieved.

## Immediate next actions
1. Add protocol metadata fields to `metrics.json`.
2. Add context-aware formatting path for LongMemEval rows.
3. Add improved scorer.
4. Re-run baseline and compare against 87.2% target.
