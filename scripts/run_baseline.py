#!/usr/bin/env python3
"""Baseline runner for LongMemEval-style evaluation.

Supports:
- Local JSONL input (`--input`)
- Hugging Face JSON file input (`--hf-dataset` + `--hf-file`)
- Solver via shell command template (`--solver cmd` + `--solver-cmd`)
- Direct Gemini API calls (`--solver gemini`)
- RLM-style iterative baseline loop (`--solver rlm`)
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
from typing import Any


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
    steps: int | None


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
        choices=["cmd", "gemini", "rlm"],
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
    parser.add_argument(
        "--gemini-model",
        default="gemini-3-flash-preview",
        help="Gemini model id for solver=gemini|rlm; pinned to Flash preview by default",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--max-questions", type=int, default=None, help="Optional cap on number of questions")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-question timeout in seconds")
    parser.add_argument("--rlm-max-steps", type=int, default=3, help="Max iterative steps for solver=rlm")
    parser.add_argument(
        "--protocol",
        default="raw_works_baseline_v1",
        help="Protocol profile label recorded in metrics/reporting metadata",
    )
    args = parser.parse_args()

    if args.hf_dataset and not args.hf_file:
        parser.error("--hf-file is required when using --hf-dataset")
    if args.solver != "cmd" and not args.gemini_model:
        parser.error("--gemini-model is required for gemini/rlm solver")
    return args


def normalize_basic(text: str) -> str:
    return " ".join(text.strip().lower().split())


def normalize_relaxed(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation + "“”’‘`"))
    cleaned = " ".join(cleaned.split())
    return cleaned


def parse_range_variants(gold: str) -> list[str]:
    variants: list[str] = []
    range_match = re.search(r"ranging\s+from\s+(\d+)\s+\w+\s+to\s+(\d+)\s+\w+", gold, flags=re.IGNORECASE)
    if range_match:
        lo = int(range_match.group(1))
        hi = int(range_match.group(2))
        unit_match = re.search(r"(day|week|month|year|hour|minute)s?", gold, flags=re.IGNORECASE)
        unit = unit_match.group(1).lower() if unit_match else ""
        for n in range(lo, hi + 1):
            variants.append(f"{n} {unit}".strip())
            variants.append(str(n))
    return variants


def accepted_gold_variants(gold: str) -> list[str]:
    variants = [gold.strip()]

    for match in re.finditer(r"([^.]+?)\s+is also acceptable", gold, flags=re.IGNORECASE):
        alt = match.group(1).strip(" .")
        if alt:
            variants.append(alt)

    for piece in re.split(r"[.;]", gold):
        p = piece.strip()
        if p and len(p) < 80:
            variants.append(p)

    variants.extend(parse_range_variants(gold))

    # quoted forms
    for match in re.finditer(r"['\"]([^'\"]+)['\"]", gold):
        q = match.group(1).strip()
        if q:
            variants.append(q)

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
    relaxed = any(pred_relaxed == normalize_relaxed(variant) for variant in accepted_gold_variants(gold))
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
        "Use only facts from the memory below.\n"
        "For temporal questions, compute date differences carefully from memory events.\n"
        "Return only final answer text, concise, no explanation.\n\n"
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


def run_cmd_solver(command_template: str, question: str, prompt: str, timeout_s: float) -> tuple[str, float, str | None, int | None]:
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
            return "", latency, completed.stderr.strip() or f"non-zero exit ({completed.returncode})", None
        return completed.stdout.strip(), latency, None, None
    except subprocess.TimeoutExpired:
        latency = time.perf_counter() - start
        return "", latency, f"timeout after {timeout_s:.1f}s", None


class GeminiAdapter:
    def __init__(self, model: str, temperature: float):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required when --solver gemini|rlm")
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def complete(self, prompt: str, system_instruction: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": self.temperature,
                "system_instruction": system_instruction,
            },
        )
        return (response.text or "").strip()


def build_gemini_solver(model: str, temperature: float):
    adapter = GeminiAdapter(model=model, temperature=temperature)

    def _solve(prompt: str, timeout_s: float, row: dict[str, Any]) -> tuple[str, float, str | None, int | None]:
        _ = timeout_s
        _ = row
        start = time.perf_counter()
        try:
            answer = adapter.complete(
                prompt,
                system_instruction="Answer accurately using provided memory context. Return only final answer text.",
            )
            latency = time.perf_counter() - start
            return answer, latency, None, 1
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - start
            return "", latency, str(exc), 1

    return _solve


def _build_memory_index(row: dict[str, Any]) -> dict[str, list[str]]:
    ids = row.get("haystack_session_ids") or []
    dates = row.get("haystack_dates") or []
    sessions = row.get("haystack_sessions") or []
    out: dict[str, list[str]] = {}
    for i, sess in enumerate(sessions, start=1):
        sid = ids[i - 1] if i - 1 < len(ids) else f"session_{i}"
        date = dates[i - 1] if i - 1 < len(dates) else "unknown"
        lines: list[str] = [f"id={sid}", f"date={date}"]
        if isinstance(sess, list):
            for turn in sess:
                if isinstance(turn, dict):
                    lines.append(f"{turn.get('role','unknown')}: {turn.get('content','')}")
        out[sid] = lines
    return out


def _search_memory(mem_index: dict[str, list[str]], query: str, limit: int = 2) -> list[str]:
    q = query.lower().strip()
    scored: list[tuple[int, str]] = []
    for sid, lines in mem_index.items():
        text = "\n".join(lines).lower()
        score = sum(1 for tok in q.split() if tok and tok in text)
        if score > 0:
            scored.append((score, sid))
    scored.sort(reverse=True)
    return [sid for _, sid in scored[:limit]]


def build_rlm_solver(model: str, temperature: float, max_steps: int):
    adapter = GeminiAdapter(model=model, temperature=temperature)

    def _solve(prompt: str, timeout_s: float, row: dict[str, Any]) -> tuple[str, float, str | None, int | None]:
        _ = timeout_s
        start = time.perf_counter()
        try:
            question = str(row.get("question", ""))
            mem_index = _build_memory_index(row)
            scratch: list[str] = []
            selected_sessions: list[str] = []

            for step in range(1, max_steps + 1):
                planner_prompt = (
                    "You are in an iterative memory-reasoning loop.\n"
                    "Given question + memory state, output JSON with fields:\n"
                    "action in {search,read,answer}, query, session_id, answer.\n"
                    "- action=search: provide query\n"
                    "- action=read: provide session_id from known sessions\n"
                    "- action=answer: provide final answer\n"
                    "JSON only.\n\n"
                    f"Question: {question}\n"
                    f"Known session ids: {', '.join(mem_index.keys())}\n"
                    f"Scratchpad:\n{chr(10).join(scratch) or '(empty)'}\n"
                )
                raw = adapter.complete(planner_prompt, "Return strict JSON only.")
                try:
                    decision = json.loads(raw)
                except json.JSONDecodeError:
                    decision = {"action": "answer", "answer": raw}

                action = str(decision.get("action", "answer")).lower()
                if action == "search":
                    query = str(decision.get("query", question))
                    hits = _search_memory(mem_index, query=query)
                    selected_sessions = hits or selected_sessions
                    scratch.append(f"step {step} search query={query} hits={hits}")
                    continue

                if action == "read":
                    session_id = str(decision.get("session_id", ""))
                    if not session_id and selected_sessions:
                        session_id = selected_sessions[0]
                    if session_id in mem_index:
                        snippet = "\n".join(mem_index[session_id][:12])
                        scratch.append(f"step {step} read {session_id}:\n{snippet}")
                    else:
                        scratch.append(f"step {step} read failed session_id={session_id}")
                    continue

                if action == "answer":
                    final = str(decision.get("answer", "")).strip()
                    if final:
                        latency = time.perf_counter() - start
                        return final, latency, None, step

            final_prompt = (
                f"Question: {question}\n\n"
                "Use the memory context and scratchpad to answer with final short answer only.\n"
                f"Scratchpad:\n{chr(10).join(scratch)}\n\n"
                f"Full context:\n{prompt}"
            )
            final = adapter.complete(final_prompt, "Return only final answer text.")
            latency = time.perf_counter() - start
            return final, latency, None, max_steps
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - start
            return "", latency, str(exc), None

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


def build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "name": args.protocol,
        "dataset": args.hf_dataset if args.hf_dataset else "local_jsonl",
        "dataset_file": args.hf_file if args.hf_dataset else args.input,
        "solver": args.solver,
        "model": args.gemini_model if args.solver in {"gemini", "rlm"} else None,
        "temperature": args.temperature if args.solver in {"gemini", "rlm"} else None,
        "task": args.task,
        "rlm_max_steps": args.rlm_max_steps if args.solver == "rlm" else None,
        "timeout_s": args.timeout,
        "notes": "Baseline-only replication pass (no GEPA/Pro/tools).",
    }


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
        solve = build_gemini_solver(args.gemini_model, args.temperature)
    elif args.solver == "rlm":
        solve = build_rlm_solver(args.gemini_model, args.temperature, args.rlm_max_steps)
    else:
        solve = lambda prompt, timeout, row: run_cmd_solver(args.solver_cmd, str(row["question"]), prompt, timeout)

    protocol = build_protocol(args)
    results: list[Result] = []
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        qid = str(row.get("id", row.get("question_id", idx)))
        question = str(row["question"])
        gold = row.get("answer", row.get("gold", row.get("target")))
        gold = None if gold is None else str(gold)
        prompt = build_prompt(row, args.task)

        print(f"[{idx}/{total}] start id={qid}", flush=True)
        prediction, latency_s, error, steps = solve(prompt, args.timeout, row)
        strict_correct, relaxed_correct = score_prediction(prediction, gold)
        status = "ok" if error is None else f"error={error}"
        print(
            f"[{idx}/{total}] done id={qid} latency_s={latency_s:.2f} steps={steps} {status} "
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
                steps=steps,
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
                        "steps": r.steps,
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
        "avg_steps": (sum(r.steps for r in results if r.steps is not None) / len([r for r in results if r.steps is not None])) if any(r.steps is not None for r in results) else None,
        "solver": args.solver,
        "solver_cmd": args.solver_cmd if args.solver == "cmd" else None,
        "gemini_model": args.gemini_model if args.solver in {"gemini", "rlm"} else None,
        "task": args.task,
        "timeout_s": args.timeout,
        "input": input_label,
        "protocol": protocol,
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
