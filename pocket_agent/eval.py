from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference import run
from .core import ToolDecisionEngine


ENGINE = ToolDecisionEngine()


def _load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}. Try --path starter/shadow_eval.jsonl")

    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def evaluate(path: Path) -> dict[str, float]:
    rows = _load_rows(path)
    if not rows:
        raise SystemExit(f"no examples found in {path}")

    exact = 0
    tool_ok = 0
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        user_message = messages[0].get("content", "") if isinstance(messages[0], dict) else ""
        expected = messages[1].get("content", "") if isinstance(messages[1], dict) else ""
        if not isinstance(user_message, str) or not isinstance(expected, str):
            continue

        result = run(user_message, [])
        if result == expected:
            exact += 1

        expected_call = ENGINE.extract_tool_call(expected)
        result_call = ENGINE.extract_tool_call(result)
        if expected_call and result_call and expected_call.get("tool") == result_call.get("tool"):
            tool_ok += 1
        elif expected_call is None and result_call is None and expected.strip() == result.strip():
            tool_ok += 1

    total = max(1, len(rows))
    return {
        "exact_match": exact / total,
        "tool_match": tool_ok / total,
        "count": float(total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pocket-Agent on a local JSONL set.")
    parser.add_argument("--path", default="starter/shadow_eval.jsonl")
    args = parser.parse_args()
    try:
        metrics = evaluate(Path(args.path))
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
