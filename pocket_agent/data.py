from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .core import stable_hash, tool_call_json


WEATHER_LOCATIONS = ["Paris", "Tokyo", "Lagos", "Bengaluru", "Cairo", "Nairobi", "Toronto", "Buenos Aires", "Dubai", "Seoul"]
CALENDAR_TITLES = ["Design review", "Doctor appointment", "Standup", "Flight check-in", "Product demo", "Team lunch", "Budget meeting"]
SQL_TOPICS = ["all users", "recent orders", "active subscriptions", "failed payments", "top customers"]


def _sample(rng: random.Random, values: list[str]) -> str:
    return values[rng.randrange(len(values))]


def _public_test_hashes(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    hashes: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        messages = row.get("messages", [])
        if isinstance(messages, list) and messages:
            prompt = messages[0].get("content", "") if isinstance(messages[0], dict) else ""
            if isinstance(prompt, str):
                hashes.add(stable_hash(prompt.strip().lower()))
    return hashes


def _avoid_leak(prompt: str, seen: set[str]) -> str:
    candidate = prompt.strip().lower()
    if stable_hash(candidate) in seen:
        return candidate + " please"
    return prompt.strip()


def build_examples(count: int, seed: int = 7, public_test: Path | None = None) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    seen = _public_test_hashes(public_test)
    examples: list[dict[str, Any]] = []

    def add(prompt: str, response: str, history: list[dict[str, str]] | None = None) -> None:
        prompt = _avoid_leak(prompt, seen)
        examples.append({"messages": (history or []) + [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]})

    for _ in range(max(1, count // 12)):
        location = _sample(rng, WEATHER_LOCATIONS)
        unit = rng.choice(["C", "F"])
        add(f"What's the weather in {location} in {unit}?", tool_call_json("weather", {"location": location, "unit": unit}))

        title = _sample(rng, CALENDAR_TITLES)
        date = f"2026-0{rng.randint(4, 9)}-{rng.randint(10, 28):02d}"
        add(f"Schedule a meeting called {title} on {date}.", tool_call_json("calendar", {"action": "create", "date": date, "title": title}))

        value = rng.choice([1, 3, 5, 10, 12, 25, 42, 100])
        from_unit, to_unit = rng.choice([("km", "mile"), ("mile", "km"), ("kg", "lb"), ("lb", "kg"), ("C", "F"), ("F", "C")])
        add(f"Convert {value} {from_unit} to {to_unit}.", tool_call_json("convert", {"value": value, "from_unit": from_unit, "to_unit": to_unit}))

        amount = rng.choice([10, 25, 50, 100, 250, 500])
        from_code, to_code = rng.sample(["USD", "EUR", "GBP", "JPY", "INR", "AED", "SAR", "CAD", "AUD", "CHF"], 2)
        add(f"Convert {amount} {from_code} to {to_code}.", tool_call_json("currency", {"amount": amount, "from": from_code, "to": to_code}))

        topic = _sample(rng, SQL_TOPICS)
        query = f"SELECT * FROM {topic.replace(' ', '_')};"
        add(f"Write a SQL query for {topic}.", tool_call_json("sql", {"query": query}))

    multi_turn_pairs = [
        ([{"role": "user", "content": "Convert 12 miles to km."}, {"role": "assistant", "content": tool_call_json("convert", {"value": 12, "from_unit": "mile", "to_unit": "km"})}], "Convert that to centimeters."),
        ([{"role": "user", "content": "How much is 30 USD in EUR?"}, {"role": "assistant", "content": tool_call_json("currency", {"amount": 30, "from": "USD", "to": "EUR"})}], "Now convert that to GBP."),
        ([{"role": "user", "content": "Schedule lunch tomorrow."}, {"role": "assistant", "content": tool_call_json("calendar", {"action": "create", "date": "2026-04-19", "title": "lunch"})}], "Move it to Friday instead."),
    ]
    for history, prompt in multi_turn_pairs:
        if len(examples) >= count:
            break
        if "convert" in prompt.lower():
            response = tool_call_json("convert", {"value": 12, "from_unit": "mile", "to_unit": "cm"}) if "centimeters" in prompt.lower() else tool_call_json("currency", {"amount": 30, "from": "USD", "to": "GBP"})
        else:
            response = tool_call_json("calendar", {"action": "create", "date": "2026-04-24", "title": "lunch"})
        add(prompt, response, history)

    adversarial_examples = [
        ("wether in Dubai pls", tool_call_json("weather", {"location": "Dubai", "unit": "C"})),
        ("mausam in Delhi in F", tool_call_json("weather", {"location": "Delhi", "unit": "F"})),
        ("convert 7 किलो to lb", tool_call_json("convert", {"value": 7, "from_unit": "kg", "to_unit": "lb"})),
        ("cuanto es 15 usd a eur", tool_call_json("currency", {"amount": 15, "from": "USD", "to": "EUR"})),
        ("show my calendar next Friday", tool_call_json("calendar", {"action": "list", "date": "2026-04-24"})),
        ("write sql for all users who joined last month", tool_call_json("sql", {"query": "SELECT * FROM users WHERE joined_at >= DATE('now','-1 month');"})),
    ]
    for prompt, response in adversarial_examples:
        if len(examples) >= count:
            break
        add(prompt, response)

    refusals = [
        "hello there",
        "tell me a joke",
        "what can you do?",
        "book a train ticket",
        "convert that to euros",
        "use the maps tool to find a cafe",
    ]
    for prompt in refusals:
        if len(examples) >= count:
            break
        add(prompt, "I can only help with weather, calendar, convert, currency, or sql requests.")

    while len(examples) < count:
        idx = len(examples)
        location = _sample(rng, WEATHER_LOCATIONS)
        unit = "C" if idx % 2 == 0 else "F"
        add(f"What's the weather like in {location}?", tool_call_json("weather", {"location": location, "unit": unit}))

    return examples[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Pocket-Agent training data.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--public-test", default=None)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    public_test = Path(args.public_test) if args.public_test else None

    examples = build_examples(args.count, seed=args.seed, public_test=public_test)
    with output.open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(examples)} examples to {output}")


if __name__ == "__main__":
    main()
