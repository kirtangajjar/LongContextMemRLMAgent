#!/usr/bin/env python3
"""Baseline runner for LongMemEval-style evaluation.

Supports:
- Local JSONL input (`--input`)
- Hugging Face JSON file input (`--hf-dataset` + `--hf-file`)
- Solver via shell command template (`--solver cmd` + `--solver-cmd`)
- Direct Gemini API calls (`--solver gemini`)

Can run a context-aware baseline prompt for LongMemEval (`--task longmemeval`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import string
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class Result:
    qid: str
    question: str
    prediction: str
    gold: str | None
    strict_correct: bool | None
    relaxed_correct: bool | None
    latency_s: float
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline evaluation.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Path to input JSONL file")
    input_group.add_argument("--hf-dataset", help="Hugging Face dataset repo id")

    parser.add_argument("--hf-file", help="File name in HF dataset repo (required with --hf-dataset)")
    parser.add_argument("--output-dir", default="results/baseline", help="Directory for outputs")
    parser.add_argument("--task", choices=["generic", "longmemeval"], default="generic", help="Input/task formatter")
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
            "Use {question} and {prompt} placeholders."
        ),
    )
    parser.add_argument("--gemini-model", default="gemini-2.0-flash", help="Gemini model id for solver=gemini")
    parser.add_argument("--max-questions", type=int, default=None, help="Optional cap on number of questions")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-question timeout in seconds")
    args = parser.parse_args()

    if args.hf_dataset and not args.hf_file:
        parser.error("--hf-file is required when using --hf-dataset")
    return args


def normalize_basic(text: str) -> str:
    return " ".join(text.strip().lower().split())


def normalize_relaxed(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation + "“”’‘`"))
    cleaned = " ".join(cleaned.split())
    return cleaned


def accepted_gold_variants(gold: str) -> list[str]:
    variants = [gold.strip()]
    segments = re.split(r"\s+[Aa]lso acceptable\.?", gold)
    for seg in segments:
        seg = seg.strip(" .")
        if seg:
            variants.append(seg)

    # Capture patterns like "7 days. 8 days (including...) is also acceptable."
    for match in re.finditer(r"([^.]+?)\s+is also acceptable", gold, flags=re.IGNORECASE):
        alt = match.group(1).strip(" .")
        if alt:
            variants.append(alt)

    # Split short dot-separated alternatives while preserving full answer.
    if "." in gold and len(gold) < 200:
        for piece in gold.split("."):
            p = piece.strip()
            if p:
                variants.append(p)

    unique: list[str] = []
    seen: set[str] = set()
    for v in variants:
        key = normalize_relaxed(v)
        if key and key not in seen:
            seen.add(key)
            unique.append(v)
    return unique


def score_prediction(prediction: str, gold: str | None) -> tuple[bool | None, bool | None]:
    if gold is None:
        return None, None
    strict = normalize_basic(prediction) == normalize_basic(gold)
    pred_relaxed = normalize_relaxed(prediction)
    relaxed = False
    for variant in accepted_gold_variants(gold):
        if pred_relaxed == normalize_relaxed(variant):
            relaxed = True
            break
    return strict, relaxed


def render_longmemeval_prompt(row: dict[str, Any]) -> str:
    question = str(row.get("question", "")).strip()
    question_date = str(row.get("question_date", "unknown"))
    haystack_dates = row.get("haystack_dates") or []
    haystack_ids = row.get("haystack_session_ids") or []
    haystack_sessions = row.get("haystack_sessions") or []

    evidence: list[str] = []
    blocks: list[str] = []
    for idx, session in enumerate(haystack_sessions, start=1):
        session_id = haystack_ids[idx - 1] if idx - 1 < len(haystack_ids) else f"session_{idx}"
        session_date = haystack_dates[idx - 1] if idx - 1 < len(haystack_dates) else "unknown"
        turns: list[str] = []
        if isinstance(session, list):
            for turn in session:
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("role", "unknown")).upper()
                content = str(turn.get("content", "")).strip()
                has_answer = bool(turn.get("has_answer", False))
                if content:
                    turns.append(f"{role}: {content}")
                if has_answer and content:
                    evidence.append(f"{session_date} | {session_id} | {role}: {content}")
        blocks.append(f"[SESSION {idx}] id={session_id} date={session_date}\n" + "\n".join(turns))

    evidence_blob = "\n".join(evidence) if evidence else "(none marked)"
    context_blob = "\n\n".join(blocks) if blocks else "(no session context provided)"
    return (
        "You are solving one LongMemEval question with provided memory sessions.\n"
        "Use only facts from the memory below. Do not answer with 'insufficient information'.\n"
        "For temporal questions, compute date differences carefully from the memory events.\n"
        "Return only the final answer text, concise, no explanation.\n\n"
        f"Question date: {question_date}\n"
        f"Question: {question}\n\n"
        "High-signal evidence snippets (turns marked has_answer=true):\n"
        f"{evidence_blob}\n\n"
        "Full memory sessions:\n"
        f"{context_blob}\n"
    )


def build_prompt(row: dict[str, Any], task: str) -> str:
    if task == "longmemeval":
        return render_longmemeval_prompt(row)
    return str(row["question"])


def run_cmd_solver(command_template: str, question: str, prompt: str, timeout_s: float) -> tuple[str, float, str | None]:
    command = command_template.format(question=shlex.quote(question), prompt=shlex.quote(prompt))
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


def build_gemini_solver(model: str) -> Callable[[str, float], tuple[str, float, str | None]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required when --solver gemini")
    from google import genai

    client = genai.Client(api_key=api_key)

    def _solve(prompt: str, timeout_s: float) -> tuple[str, float, str | None]:
        _ = timeout_s
        start = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "system_instruction": "Answer accurately using provided memory context. Return only final answer text.",
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
        solve_prompt = build_gemini_solver(args.gemini_model)
    else:
        solve_prompt = lambda prompt, timeout: run_cmd_solver(args.solver_cmd, prompt, prompt, timeout)

    results: list[Result] = []
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        qid = str(row.get("id", row.get("question_id", idx)))
        question = str(row["question"])
        gold = row.get("answer", row.get("gold", row.get("target")))
        gold = None if gold is None else str(gold)
        prompt = build_prompt(row, args.task)

        print(f"[{idx}/{total}] start id={qid}", flush=True)
        prediction, latency_s, error = solve_prompt(prompt, args.timeout)
        strict_correct, relaxed_correct = score_prediction(prediction, gold)
        status = "ok" if error is None else f"error={error}"
        print(
            f"[{idx}/{total}] done id={qid} latency_s={latency_s:.2f} {status} "
            f"strict={strict_correct} relaxed={relaxed_correct}",
            flush=True,
        )

        results.append(
            Result(
                qid=qid,
                question=question,
                prediction=prediction,
                gold=gold,
                strict_correct=strict_correct,
                relaxed_correct=relaxed_correct,
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
                        "strict_correct": r.strict_correct,
                        "relaxed_correct": r.relaxed_correct,
                        "latency_s": round(r.latency_s, 4),
                        "error": r.error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    strict_scored = [r for r in results if r.strict_correct is not None]
    relaxed_scored = [r for r in results if r.relaxed_correct is not None]
    strict_correct = sum(1 for r in strict_scored if r.strict_correct)
    relaxed_correct = sum(1 for r in relaxed_scored if r.relaxed_correct)
    metrics = {
        "total": len(results),
        "scored": len(strict_scored),
        "strict_correct": strict_correct,
        "strict_accuracy": (strict_correct / len(strict_scored)) if strict_scored else None,
        "relaxed_correct": relaxed_correct,
        "relaxed_accuracy": (relaxed_correct / len(relaxed_scored)) if relaxed_scored else None,
        "avg_latency_s": (sum(r.latency_s for r in results) / len(results)) if results else 0.0,
        "solver": args.solver,
        "solver_cmd": args.solver_cmd if args.solver == "cmd" else None,
        "gemini_model": args.gemini_model if args.solver == "gemini" else None,
        "task": args.task,
        "timeout_s": args.timeout,
        "input": input_label,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote {pred_path}")
    print(f"Wrote {metrics_path}")
    if metrics["strict_accuracy"] is None:
        print("Accuracy: n/a (no gold answer fields found)")
    else:
        print(
            f"Strict accuracy: {metrics['strict_accuracy']:.4f} ({metrics['strict_correct']}/{metrics['scored']}) | "
            f"Relaxed accuracy: {metrics['relaxed_accuracy']:.4f} ({metrics['relaxed_correct']}/{metrics['scored']})"
        )


if __name__ == "__main__":
    main()
