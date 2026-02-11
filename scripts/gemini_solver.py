#!/usr/bin/env python3
"""Minimal Gemini Flash solver for baseline runs.

Reads one question from --question and prints a single-line answer.
Requires GEMINI_API_KEY in env.
"""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer a question with Gemini.")
    parser.add_argument("--question", required=True, help="Question string")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model name")
    parser.add_argument(
        "--system",
        default="You are a concise QA assistant. Return only the final answer.",
        help="System instruction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set")

    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit("Missing dependency: install with `pip install google-genai`.") from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=args.model,
        config={"system_instruction": args.system, "temperature": 0.0},
        contents=args.question,
    )

    text = (response.text or "").strip()
    print(text)


if __name__ == "__main__":
    main()
