# Tweet context for LongMemEval replication

## Target tweet
- URL: https://x.com/raw_works/status/2021413002939871552?s=46
- Tweet ID: `2021413002939871552`
- Key claims from the tweet text:
  - dspy.RLM Flash baseline (no tools): **87.2%** (436/500)
  - dspy.RLM Flash + tools + prompt (exp 004): **89.2%** (449/500)
  - dspy.RLM Flash + obs memory + tools (exp 006): **89.8%**
  - + GEPA prompt optimization: **87.8%**
  - + Pro model swap: **88.4%**

## Parent tweet
- URL: https://twitter.com/raw_works/status/2021412999932612648
- Tweet ID: `2021412999932612648`
- Key context:
  - States Gemini Flash + DSPy.rlm reached **89.8%** on LongMemEval.
  - Says this was achieved without special preprocessing "memory system".

## Tweet quoted by parent tweet
- Parent's quoted tweet ID: `2021303970929479795` (by @raw_works)
- That tweet quotes: https://twitter.com/tylbar/status/2020924183979397512
- Quoted tweet ID: `2020924183979397512` (by @tylbar)
- Key context from quoted tweet:
  - Announces Observational Memory (OM) as a new memory approach and references LongMemEval SOTA claims.

## Retrieval notes
- Context fetched via vxtwitter API endpoints:
  - `https://api.vxtwitter.com/raw_works/status/2021413002939871552`
  - `https://api.vxtwitter.com/raw_works/status/2021412999932612648`
  - `https://api.vxtwitter.com/i/status/2020924183979397512`
