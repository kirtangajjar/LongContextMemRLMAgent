#!/usr/bin/env python3
"""Simple baseline runner for benchmark evaluation.

Supports:
- Local JSONL input (`--input`)
- Hugging Face JSON file input (`--hf-dataset` + `--hf-file`)
- Solver via shell command template (`--solver cmd` + `--solver-cmd`)
- Direct Gemini API calls (`--solver gemini`)

Input rows must have `question` and may have `id` plus one of `answer|gold|target`.
"""

from __future__ import annotations

import argparse
import json
import os
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
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Path to input JSONL file")
    input_group.add_argument("--hf-dataset", help="Hugging Face dataset repo id (e.g., xiaowu0162/longmemeval-cleaned)")

    parser.add_argument("--hf-file", help="File name in HF dataset repo (required with --hf-dataset)")
    parser.add_argument(
        "--output-dir",
        default="results/baseline",
        help="Directory for predictions.jsonl and metrics.json",
    )
    parser.add_argument(
        "--solver",
        choices=["cmd", "gemini"],
        default="cmd",
        help="Answer generation backend",
    )
    parser.add_argument(
        "--solver-cmd",
        default="python -c \"print('')\"",
        help=(
            "Shell command template used to answer each question (solver=cmd). "
            "Use {question} placeholder, e.g. \"python my_solver.py --question {question}\""
        ),
    )
    parser.add_argument("--gemini-model", default="gemini-2.0-flash", help="Gemini model id for solver=gemini")
    parser.add_argument("--max-questions", type=int, default=None, help="Optional cap on number of questions")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-question timeout in seconds")
    args = parser.parse_args()

    if args.hf_dataset and not args.hf_file:
        parser.error("--hf-file is required when using --hf-dataset")
    return args


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def run_cmd_solver(command_template: str, question: str, timeout_s: float) -> tuple[str, float, str | None]:
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


def build_gemini_solver(model: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required when --solver gemini")
    from google import genai

    client = genai.Client(api_key=api_key)

    def _solve(question: str, timeout_s: float) -> tuple[str, float, str | None]:
        _ = timeout_s
        start = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model,
                contents=question,
                config={
                    "temperature": 0.0,
                    "system_instruction": "You are a concise QA assistant. Return only the final answer.",
                },
            )
            latency = time.perf_counter() - start
            return (response.text or "").strip(), latency, None
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - start
            return "", latency, str(exc)

    return _solve


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def load_hf_json_file(dataset_repo: str, filename: str) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    local_path = Path(hf_hub_download(repo_id=dataset_repo, repo_type="dataset", filename=filename))
    with local_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            records = data["data"]
        else:
            raise ValueError("Unsupported JSON object format in HF file")
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("HF file must be a JSON list or object with `data` list")

    rows: list[dict[str, Any]] = []
    for rec in records:
        if isinstance(rec, dict) and "question" in rec:
            rows.append(rec)
    if not rows:
        raise ValueError("No usable rows with `question` found in HF file")
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.hf_dataset:
        rows = load_hf_json_file(args.hf_dataset, args.hf_file)
        input_label = f"hf://{args.hf_dataset}/{args.hf_file}"
    else:
        input_path = Path(args.input)
        rows = load_jsonl(input_path)
        input_label = str(input_path)

    if args.max_questions is not None:
        rows = rows[: args.max_questions]

    if args.solver == "gemini":
        solve = build_gemini_solver(args.gemini_model)
    else:
        solve = lambda q, t: run_cmd_solver(args.solver_cmd, q, t)

    results: list[Result] = []
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        qid = str(row.get("id", idx))
        question = str(row["question"])
        gold = row.get("answer", row.get("gold", row.get("target")))
        gold = None if gold is None else str(gold)

        print(f"[{idx}/{total}] start id={qid}", flush=True)
        prediction, latency_s, error = solve(question, args.timeout)
        correct = None if gold is None else (normalize(prediction) == normalize(gold))
        status = "ok" if error is None else f"error={error}"
        print(f"[{idx}/{total}] done id={qid} latency_s={latency_s:.2f} {status}", flush=True)

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
        "solver": args.solver,
        "solver_cmd": args.solver_cmd if args.solver == "cmd" else None,
        "gemini_model": args.gemini_model if args.solver == "gemini" else None,
        "timeout_s": args.timeout,
        "input": input_label,
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
