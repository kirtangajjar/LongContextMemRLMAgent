#!/usr/bin/env python3
"""Download a Hugging Face dataset split and export it to JSONL.

Keeps things simple: pick text fields for question/answer and write compact rows
for `scripts/run_baseline.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a benchmark split and save as JSONL.")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name, e.g. gsm8k")
    parser.add_argument("--config", default=None, help="Optional dataset config/subset")
    parser.add_argument("--split", default="test", help="Dataset split to export")
    parser.add_argument("--question-field", default="question", help="Column name for question text")
    parser.add_argument(
        "--answer-field",
        default="answer",
        help="Column name for gold answer text (optional if unavailable)",
    )
    parser.add_argument("--id-field", default=None, help="Optional ID column; defaults to row index")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap; 0 means all rows")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: install with `pip install datasets`.") from exc

    ds = load_dataset(args.dataset, args.config, split=args.split)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(ds)
    limit = total if args.max_rows <= 0 else min(args.max_rows, total)

    with output_path.open("w", encoding="utf-8") as f:
        for i in range(limit):
            row = ds[i]
            if args.question_field not in row:
                raise KeyError(f"question field '{args.question_field}' missing in row {i}")

            out = {
                "id": str(row[args.id_field]) if args.id_field and args.id_field in row else str(i),
                "question": str(row[args.question_field]),
            }

            if args.answer_field in row and row[args.answer_field] is not None:
                out["answer"] = str(row[args.answer_field])

            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote {limit} rows to {output_path}")


if __name__ == "__main__":
    main()
