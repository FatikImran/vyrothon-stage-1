# Pocket-Agent

Offline tool-calling assistant for the Pocket-Agent hackathon.

This repository is designed to be reproducible end to end on a free Colab T4 or a local Python environment. The core inference path is a deterministic, offline parser that emits exact tool calls when the intent is unambiguous, with optional model-backed generation hooks for training and experimentation.

This project now follows a synthetic-first strategy. That means the training set is generated locally, the parser is used as the stable baseline, and any public datasets are optional augmentation only.

## What is included

- Synthetic data generation for the five final tool schemas.
- Leakage checks against the provided public test set when available.
- A local shadow eval set under `starter/` so you can regression-test without the official pack.
- A LoRA fine-tuning script for an open-weight Qwen2.5 base model.
- Quantization and merge scripts.
- A demo app that launches in Gradio or from the CLI.
- `inference.py` with the grader contract `run(prompt: str, history: list[dict]) -> str`.
- A small test suite that exercises tool selection, context resolution, and refusals.

## Base model choice

The training scripts default to `Qwen/Qwen2.5-1.5B-Instruct`, matching the competition brief. The shipped inference path does not depend on a network connection and will still work if the model artifacts are absent by falling back to the local parser.

## Quick start

```bash
python -m pip install -r requirements.txt
make test
python -m pocket_agent.data --output data/generated/train.jsonl --count 2400
python -m pocket_agent.train --data data/generated/train.jsonl --output artifacts/lora --base-model Qwen/Qwen2.5-1.5B-Instruct
python -m pocket_agent.quantize --adapter artifacts/lora --output artifacts/quantized
python -m pocket_agent.demo
make shadow-eval
```

## Recommended workflow

1. Generate synthetic data.
2. Use the shadow eval set in `starter/shadow_eval.jsonl` to sanity-check the parser and prompt templates.
3. If an official `starter/public_test.jsonl` is ever provided, point the generator at it for leakage checks.
4. Fine-tune with LoRA in Colab.
5. Merge and quantize the adapter.
6. Run `make shadow-eval` locally to catch regressions.
7. Launch the demo and inspect visible tool-call output.

## Manual steps you need to do

1. Create a public GitHub repo and push this workspace into it.
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Open Colab and run the training pipeline there if your local machine cannot handle the model.
4. Generate the synthetic dataset with `python -m pocket_agent.data --output data/generated/train.jsonl --count 2400`.
5. Run the training and quantization scripts in Colab.
6. Use the demo or CLI to smoke-test the final artifacts before submission.

## Synthetic-first strategy

- Synthetic data is the primary training signal because the final task is schema-specific and the public evaluation wording is hidden.
- Public datasets, if used at all, should only be normalized into the exact five-tool schema.
- The shadow eval set is only for local regression. It is not a substitute for the hidden grader set.
- Keep the deterministic parser in place even after training, because it protects refusal behavior and latency.

## Design decisions

- The inference layer is deterministic first. That keeps latency low and makes the output format stable for structured tool calls.
- The parser resolves common follow-up references like “that” and “it” by carrying forward the most recent actionable user context.
- Synthetic data includes clean, paraphrased, code-switched, typo-heavy, and refusal examples so the fine-tuning set is not narrow.
- Validation is strict: malformed inputs, missing context, or unsupported requests become plain-text refusals instead of bad tool calls.

## What worked well

- Template diversity for tool intents makes the model see the same schema in many surfaces.
- Deterministic validation of tool payloads is much more reliable than free-form generation for a small model.
- Context carry-over is essential for multi-turn examples and improves the apparent intelligence of the assistant.

## What did not

- Pure generation without a validator is too brittle for adversarial prompts.
- Large base models make the size gate harder to satisfy once quantized.
- Relative date parsing is easy to get subtly wrong, so the code is conservative when the intent is not clear.

## Notes on inference

`inference.py` intentionally avoids network imports. It first tries to load a local model artifact if one exists, then falls back to the offline parser. The parser is what makes the repo usable in a clean evaluation environment even if the trained weights are not present.
