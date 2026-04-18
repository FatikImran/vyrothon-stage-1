from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_model_with_quantization(base_model: str):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # type: ignore
    import torch  # type: ignore

    # transformers v5 expects quantization_config instead of load_in_4bit.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    return AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=bnb_config,
    )


def _load_model_fallback(base_model: str):
    from transformers import AutoModelForCausalLM  # type: ignore

    # CPU-only fallback when 4-bit cannot initialize in the current runtime.
    return AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")


def _build_trainer(
    trainer_cls,
    model,
    tokenizer,
    dataset,
    training_args,
):
    try:
        return trainer_cls(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            args=training_args,
            packing=True,
            max_seq_length=512,
        )
    except TypeError:
        # Newer TRL versions renamed tokenizer -> processing_class.
        return trainer_cls(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=training_args,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning entrypoint for Pocket-Agent.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"training data not found: {data_path}")

    try:
        from datasets import Dataset  # type: ignore
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments  # type: ignore
        from trl import SFTTrainer  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(f"Missing training dependencies: {exc}")

    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("training dataset is empty")

    def format_row(row: dict[str, object]) -> dict[str, str]:
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise SystemExit("invalid dataset row: missing messages")
        text = ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            text += f"<{role}>\n{content}\n"
        return {"text": text.strip()}

    dataset = Dataset.from_list([format_row(row) for row in rows])
    try:
        model = _load_model_with_quantization(args.base_model)
    except Exception as exc:
        print(f"4-bit load failed, retrying without 4-bit quantization: {exc}")
        model = _load_model_fallback(args.base_model)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        bf16=False,
        fp16=False,
        max_steps=args.max_steps,
        report_to=[],
    )

    trainer = _build_trainer(
        trainer_cls=SFTTrainer,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_args=training_args,
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    print(f"saved LoRA adapter to {output}")


if __name__ == "__main__":
    main()
