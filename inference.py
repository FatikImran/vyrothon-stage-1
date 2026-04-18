from __future__ import annotations

import json
from typing import Any

from pocket_agent.core import ToolDecisionEngine

_ENGINE = ToolDecisionEngine()
_MODEL_BACKEND = None


def _load_backend() -> Any:
    global _MODEL_BACKEND
    if _MODEL_BACKEND is not None:
        return _MODEL_BACKEND

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch  # type: ignore
    except Exception:
        _MODEL_BACKEND = False
        return _MODEL_BACKEND

    model_path = "artifacts/quantized"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=getattr(torch, "float16", None),
            device_map="auto",
        )
        _MODEL_BACKEND = (tokenizer, model)
    except Exception:
        _MODEL_BACKEND = False

    return _MODEL_BACKEND


def _format_tool_call(decision: dict[str, Any]) -> str:
    return f"<tool_call>{json.dumps(decision, separators=(',', ':'), ensure_ascii=False)}</tool_call>"


def _coerce_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    if not history:
        return []

    sanitized: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user")).strip().lower()
        if role not in {"user", "assistant", "system"}:
            role = "user"
        content = item.get("content", "")
        if isinstance(content, str):
            sanitized.append({"role": role, "content": content})
    return sanitized


def _model_generate(prompt: str, history: list[dict[str, str]]) -> str | None:
    backend = _load_backend()
    if not backend:
        return None

    tokenizer, model = backend
    messages = history + [{"role": "user", "content": prompt}]
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.strip()
    except Exception:
        return None


def run(prompt: str, history: list[dict] | None) -> str:
    cleaned_prompt = prompt.strip() if isinstance(prompt, str) else ""
    cleaned_history = _coerce_history(history)
    if not cleaned_prompt:
        return "I need a request before I can help."

    model_text = _model_generate(cleaned_prompt, cleaned_history)
    if model_text:
        parsed = _ENGINE.extract_tool_call(model_text)
        if parsed is not None:
            return _format_tool_call(parsed)
        if _ENGINE.looks_like_refusal(model_text):
            return _ENGINE.normalize_refusal(model_text)

    decision = _ENGINE.decide(cleaned_prompt, cleaned_history)
    if decision["kind"] == "tool_call":
        return _format_tool_call(decision["payload"])
    return decision["payload"]


def main() -> None:
    import sys

    raw = sys.stdin.read().strip()
    if not raw:
        raise SystemExit("Provide a prompt on stdin or import run() directly.")
    print(run(raw, []))


if __name__ == "__main__":
    main()
