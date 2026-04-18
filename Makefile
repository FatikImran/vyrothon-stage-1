PYTHON = c:/python314/python.exe

.PHONY: all test generate train quantize demo shadow-eval

all: generate test

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

generate:
	$(PYTHON) -m pocket_agent.data --output data/generated/train.jsonl --count 2400

train:
	$(PYTHON) -m pocket_agent.train --data data/generated/train.jsonl --output artifacts/lora --base-model Qwen/Qwen2.5-1.5B-Instruct

quantize:
	$(PYTHON) -m pocket_agent.quantize --adapter artifacts/lora --output artifacts/quantized

demo:
	$(PYTHON) -m pocket_agent.demo

shadow-eval:
	$(PYTHON) -m pocket_agent.eval --path starter/shadow_eval.jsonl
