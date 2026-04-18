from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning entrypoint for Pocket-Agent.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-steps", type=int, default=1000)
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
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        bf16=False,
        fp16=True,
        max_steps=args.max_steps,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=training_args,
        packing=True,
        max_seq_length=512,
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    print(f"saved LoRA adapter to {output}")


if __name__ == "__main__":
    main()
