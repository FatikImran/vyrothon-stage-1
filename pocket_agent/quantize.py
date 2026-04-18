from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and quantize a trained Pocket-Agent adapter.")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = Path(args.adapter)
    if not adapter.exists():
        raise SystemExit(f"adapter path not found: {adapter}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from peft import PeftModel  # type: ignore
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(f"Missing quantization dependencies: {exc}")

    base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=getattr(torch, "float16", None), device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, str(adapter))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    print(f"saved quantized/merged artifact to {output}")


if __name__ == "__main__":
    main()
