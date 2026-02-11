#!/usr/bin/env python3
"""Simple baseline runner for 500-question evaluation sets.

Input format: JSONL where each row contains:
- id (optional)
- question (required)
- answer OR gold OR target (optional, for scoring)

This script intentionally keeps the baseline simple:
- no tools
- one model call per question via a shell command template
- exact-match scoring when gold answers are present
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Result:
    qid: str
    question: str
    prediction: str
    gold: str | None
    correct: bool | None
    latency_s: float
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple baseline evaluation.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument(
        "--output-dir",
        default="results/baseline",
        help="Directory for predictions.jsonl and metrics.json",
    )
    parser.add_argument(
        "--solver-cmd",
        default="python -c \"print('')\"",
        help=(
            "Shell command template used to answer each question. "
            "Use {question} placeholder, e.g. "
            "\"python my_solver.py --question {question}\""
        ),
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-question timeout in seconds")
    return parser.parse_args()


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def run_solver(command_template: str, question: str, timeout_s: float) -> tuple[str, float, str | None]:
    command = command_template.format(question=shlex.quote(question))
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            shell=True,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        latency = time.perf_counter() - start
        if completed.returncode != 0:
            return "", latency, completed.stderr.strip() or f"non-zero exit ({completed.returncode})"
        return completed.stdout.strip(), latency, None
    except subprocess.TimeoutExpired:
        latency = time.perf_counter() - start
        return "", latency, f"timeout after {timeout_s:.1f}s"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "question" not in row:
                raise ValueError(f"Missing 'question' on line {line_no}")
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_dataset(input_path)
    results: list[Result] = []

    for idx, row in enumerate(rows):
        qid = str(row.get("id", idx))
        question = str(row["question"])
        gold = row.get("answer", row.get("gold", row.get("target")))
        gold = None if gold is None else str(gold)

        prediction, latency_s, error = run_solver(args.solver_cmd, question, args.timeout)
        correct = None if gold is None else (normalize(prediction) == normalize(gold))

        results.append(
            Result(
                qid=qid,
                question=question,
                prediction=prediction,
                gold=gold,
                correct=correct,
                latency_s=latency_s,
                error=error,
            )
        )

    pred_path = output_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "id": r.qid,
                        "question": r.question,
                        "prediction": r.prediction,
                        "gold": r.gold,
                        "correct": r.correct,
                        "latency_s": round(r.latency_s, 4),
                        "error": r.error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    scored = [r for r in results if r.correct is not None]
    n_correct = sum(1 for r in scored if r.correct)
    metrics = {
        "total": len(results),
        "scored": len(scored),
        "correct": n_correct,
        "accuracy": (n_correct / len(scored)) if scored else None,
        "avg_latency_s": (sum(r.latency_s for r in results) / len(results)) if results else 0.0,
        "solver_cmd": args.solver_cmd,
        "timeout_s": args.timeout,
        "input": str(input_path),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote {pred_path}")
    print(f"Wrote {metrics_path}")
    if metrics["accuracy"] is None:
        print("Accuracy: n/a (no gold answer fields found)")
    else:
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['scored']})")


if __name__ == "__main__":
    main()
