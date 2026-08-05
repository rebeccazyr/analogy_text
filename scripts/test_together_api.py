#!/usr/bin/env python3
"""Send one small request to verify Together API access."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from dotenv import load_dotenv
from together import Together


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser


def usage_value(usage: Any, name: str) -> int | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(name)
    else:
        value = getattr(usage, name, None)
    return int(value) if value is not None else None


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv()

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("ERROR: TOGETHER_API_KEY is not set.", file=sys.stderr)
        return 2

    try:
        client = Together(api_key=api_key)
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly API_OK and nothing else.",
                }
            ],
            reasoning_effort="low",
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
    except Exception as error:
        print(f"ERROR: Together request failed: {error}", file=sys.stderr)
        return 1

    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    print("Together API request succeeded.")
    print(f"Model: {args.model}")
    print(f"Response: {content.strip() or '<empty>'}")
    print(
        "Tokens: "
        f"input={usage_value(usage, 'prompt_tokens')}, "
        f"output={usage_value(usage, 'completion_tokens')}, "
        f"total={usage_value(usage, 'total_tokens')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
