from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import re
from typing import Any


TOOL_NAMES = {"weather", "calendar", "convert", "currency", "sql"}
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

WEATHER_TERMS = {
    "weather",
    "forecast",
    "temperature",
    "climate",
    "mausam",
    "clima",
    "طقس",
    "موسم",
    "tiempo",
}

CONVERT_TERMS = {"convert", "change", "turn", "how many", "equivalent", "equals", "equals to"}
CURRENCY_TERMS = {"currency", "dollars", "bucks", "euros", "pounds", "yen", "rupees", "aed", "sar", "usd", "eur", "gbp", "jpy", "inr", "dirham", "peso"}
CALENDAR_TERMS = {"calendar", "meeting", "event", "appointment", "schedule", "remind", "reminder", "list events", "create event", "add event"}
SQL_TERMS = {"sql", "query", "select", "insert", "update", "delete", "table", "database", "join", "where"}

UNIT_ALIASES = {
    "c": "C",
    "celsius": "C",
    "centigrade": "C",
    "f": "F",
    "fahrenheit": "F",
    "km": "km",
    "kilometer": "km",
    "kilometers": "km",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "mm": "mm",
    "mile": "mile",
    "miles": "mile",
    "inch": "inch",
    "inches": "inch",
    "ft": "ft",
    "feet": "ft",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "l": "l",
    "liter": "l",
    "liters": "l",
    "ml": "ml",
}

CURRENCY_ALIASES = {
    "usd": "USD",
    "us dollar": "USD",
    "us dollars": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "eur": "EUR",
    "euro": "EUR",
    "euros": "EUR",
    "gbp": "GBP",
    "pound": "GBP",
    "pounds": "GBP",
    "sterling": "GBP",
    "jpy": "JPY",
    "yen": "JPY",
    "inr": "INR",
    "rupee": "INR",
    "rupees": "INR",
    "aed": "AED",
    "dirham": "AED",
    "dirhams": "AED",
    "sar": "SAR",
    "riyal": "SAR",
    "riyals": "SAR",
    "cny": "CNY",
    "yuan": "CNY",
    "renminbi": "CNY",
    "mxn": "MXN",
    "peso": "MXN",
    "pesos": "MXN",
    "cad": "CAD",
    "aud": "AUD",
    "chf": "CHF",
}

REFUSAL_TEXT = "I can only help with weather, calendar, convert, currency, or sql requests."
REFERENCE_DATE = dt.date(2026, 4, 18)


@dataclasses.dataclass(slots=True)
class ConversationState:
    location: str | None = None
    date: str | None = None
    title: str | None = None
    amount: float | None = None
    source_unit: str | None = None
    target_unit: str | None = None
    currency_amount: float | None = None
    currency_from: str | None = None
    currency_to: str | None = None
    sql_query: str | None = None


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _safe_float(value: str) -> float | None:
    try:
        return float(value.replace(",", ""))
    except Exception:
        return None


def _extract_number_unit(text: str) -> tuple[float | None, str | None]:
    matches = re.findall(r"(-?\d+(?:\.\d+)?)\s*([a-zA-Z°]+)?", text)
    if not matches:
        return None, None
    number_text, unit_text = matches[0]
    number = _safe_float(number_text)
    unit = None
    if unit_text:
        unit = UNIT_ALIASES.get(unit_text.lower(), unit_text)
    return number, unit


def _extract_location(text: str) -> str | None:
    patterns = [
        r"\b(?:in|for|at|near|around)\s+([a-zA-Z][a-zA-Z\s.'-]{1,60})",
        r"\bweather\s+in\s+([a-zA-Z][a-zA-Z\s.'-]{1,60})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" ?!.,")
            candidate = re.split(
                r"\b(?:in\s+(?:c|f|celsius|fahrenheit)|to\s+[a-z]+|on\s+\d|today|tomorrow|next\s+[a-z]+|please|now)\b",
                candidate,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" ?!.,")
            if candidate:
                return candidate
    return None


def _extract_title(text: str) -> str | None:
    match = re.search(r"(?:called|titled|named|for)\s+(.+)$", text, flags=re.IGNORECASE)
    if match:
        title = re.split(r"\b(?:on|at|for|tomorrow|today|next\s+[a-z]+|,|\.|!)\b", match.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
        title = title.strip(" ?!.,")
        return title or None
    return None


def _resolve_date(text: str, today: dt.date | None = None) -> str | None:
    today = today or REFERENCE_DATE
    raw = _normalize_text(text)

    direct = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", raw)
    if direct:
        return direct.group(1)

    compact = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", raw)
    if compact:
        month = int(compact.group(1))
        day = int(compact.group(2))
        year = int(compact.group(3))
        if year < 100:
            year += 2000
        try:
            return dt.date(year, month, day).isoformat()
        except ValueError:
            return None

    relative_map = {"today": 0, "tomorrow": 1, "day after tomorrow": 2}
    for phrase, delta in relative_map.items():
        if phrase in raw:
            return (today + dt.timedelta(days=delta)).isoformat()

    for weekday, index in WEEKDAYS.items():
        if weekday in raw:
            days_ahead = (index - today.weekday()) % 7
            if "next" in raw:
                days_ahead = days_ahead or 7
            return (today + dt.timedelta(days=days_ahead)).isoformat()

    month_match = re.search(r"\b(" + "|".join(MONTHS.keys()) + r")\s+(\d{1,2})(?:,\s*(\d{4}))?\b", raw)
    if month_match:
        month = MONTHS[month_match.group(1)]
        day = int(month_match.group(2))
        year = int(month_match.group(3)) if month_match.group(3) else today.year
        try:
            return dt.date(year, month, day).isoformat()
        except ValueError:
            return None

    return None


def _extract_currency_code(text: str) -> str | None:
    raw = _normalize_text(text)
    for token, code in CURRENCY_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", raw):
            return code
    return None


def _extract_sql_query(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    if re.match(r"(?is)^\s*(select|insert|update|delete|with)\b", stripped):
        return stripped.rstrip(".")
    match = re.search(r"(?:sql|query)\s*[:\-]\s*(.+)$", stripped, flags=re.IGNORECASE | re.DOTALL)
    if match:
        query = match.group(1).strip()
        if query:
            return query.rstrip(".")
    return None


def _parse_history_message(message: dict[str, str], state: ConversationState) -> None:
    content = message.get("content", "")
    if message.get("role") == "assistant":
        parsed = ToolDecisionEngine().extract_tool_call(content)
        if parsed:
            tool = parsed.get("tool")
            args = parsed.get("args", {}) if isinstance(parsed, dict) else {}
            if tool == "weather":
                state.location = args.get("location") or state.location
            elif tool == "calendar":
                state.date = args.get("date") or state.date
                state.title = args.get("title") or state.title
            elif tool == "convert":
                state.amount = args.get("value", state.amount)
                state.source_unit = args.get("from_unit") or state.source_unit
                state.target_unit = args.get("to_unit") or state.target_unit
            elif tool == "currency":
                state.currency_amount = args.get("amount", state.currency_amount)
                state.currency_from = args.get("from") or state.currency_from
                state.currency_to = args.get("to") or state.currency_to
            elif tool == "sql":
                state.sql_query = args.get("query") or state.sql_query
        return

    lowered = _normalize_text(content)
    if state.location is None:
        location = _extract_location(lowered)
        if location:
            state.location = location
    if state.date is None:
        date = _resolve_date(lowered)
        if date:
            state.date = date
    if state.title is None:
        title = _extract_title(content)
        if title:
            state.title = title

    amount, unit = _extract_number_unit(lowered)
    if amount is not None:
        if unit and unit in {"C", "F", "km", "m", "cm", "mm", "mile", "inch", "ft", "kg", "g", "lb", "l", "ml"}:
            state.amount = amount
            state.source_unit = unit
        elif unit is None and state.amount is None:
            state.amount = amount

    currency_code = _extract_currency_code(lowered)
    if currency_code:
        if state.currency_from is None:
            state.currency_from = currency_code
        elif state.currency_to is None:
            state.currency_to = currency_code
    if state.currency_amount is None and amount is not None:
        state.currency_amount = amount

    sql_query = _extract_sql_query(content)
    if sql_query:
        state.sql_query = sql_query


def _history_state(history: list[dict[str, str]]) -> ConversationState:
    state = ConversationState()
    for message in history:
        if isinstance(message, dict):
            _parse_history_message(message, state)
    return state


class ToolDecisionEngine:
    def __init__(self) -> None:
        self.now = REFERENCE_DATE

    def looks_like_refusal(self, text: str) -> bool:
        lowered = _normalize_text(text)
        return not any(token in lowered for token in ("<tool_call>", "weather", "calendar", "convert", "currency", "sql")) and len(lowered) > 0

    def normalize_refusal(self, text: str) -> str:
        cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
        return cleaned or REFUSAL_TEXT

    def extract_tool_call(self, text: str) -> dict[str, Any] | None:
        match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(1))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        tool = payload.get("tool")
        args = payload.get("args")
        if tool not in TOOL_NAMES or not isinstance(args, dict):
            return None
        return payload

    def decide(self, prompt: str, history: list[dict[str, str]]) -> dict[str, Any]:
        prompt = prompt.strip()
        normalized = _normalize_text(prompt)
        state = _history_state(history)

        if self._is_chitchat(normalized):
            return {"kind": "refusal", "payload": REFUSAL_TEXT}

        if self._should_weather(normalized):
            location = _extract_location(prompt) or state.location
            unit = "C" if re.search(r"\bc(?:elsius)?\b", normalized) else "F" if re.search(r"\bf(?:ahrenheit)?\b", normalized) else None
            if not location:
                return {"kind": "refusal", "payload": REFUSAL_TEXT}
            if unit is None:
                unit = "C"
            return {"kind": "tool_call", "payload": {"tool": "weather", "args": {"location": location, "unit": unit}}}

        if self._should_calendar(normalized):
            action = "list" if any(term in normalized for term in ("list", "show", "what's on", "what is on", "agenda")) else "create"
            date = _resolve_date(prompt) or state.date
            if not date:
                return {"kind": "refusal", "payload": REFUSAL_TEXT}
            args: dict[str, Any] = {"action": action, "date": date}
            if action == "create":
                title = _extract_title(prompt) or state.title
                if title:
                    args["title"] = title
            return {"kind": "tool_call", "payload": {"tool": "calendar", "args": args}}

        if self._should_currency(normalized):
            amount, _ = _extract_number_unit(normalized)
            from_code = _extract_currency_code(normalized) or state.currency_from
            to_code = self._extract_target_currency(normalized) or state.currency_to
            if amount is None:
                amount = state.currency_amount
            if amount is None or from_code is None or to_code is None:
                return {"kind": "refusal", "payload": REFUSAL_TEXT}
            return {"kind": "tool_call", "payload": {"tool": "currency", "args": {"amount": amount, "from": from_code, "to": to_code}}}

        if self._should_convert(normalized):
            amount, unit = _extract_number_unit(normalized)
            if amount is None:
                amount = state.amount
            if unit is None:
                unit = state.source_unit
            to_unit = self._extract_target_unit(normalized) or state.target_unit
            if amount is None or unit is None or to_unit is None:
                return {"kind": "refusal", "payload": REFUSAL_TEXT}
            return {"kind": "tool_call", "payload": {"tool": "convert", "args": {"value": amount, "from_unit": unit, "to_unit": to_unit}}}

        if self._should_sql(normalized):
            sql_query = _extract_sql_query(prompt) or state.sql_query or self._synthesize_sql_query(normalized)
            if not sql_query:
                return {"kind": "refusal", "payload": REFUSAL_TEXT}
            if not re.search(r"\b(select|insert|update|delete|with)\b", sql_query, flags=re.IGNORECASE):
                sql_query = f"SELECT {sql_query}"
            return {"kind": "tool_call", "payload": {"tool": "sql", "args": {"query": sql_query}}}

        return {"kind": "refusal", "payload": REFUSAL_TEXT}

    def _is_chitchat(self, text: str) -> bool:
        return any(phrase in text for phrase in ("how are you", "tell me a joke", "what can you do", "hello", "hi ", "hey ", "good morning", "good evening")) and not any(term in text for term in WEATHER_TERMS | CALENDAR_TERMS | CONVERT_TERMS | CURRENCY_TERMS | SQL_TERMS)

    def _should_weather(self, text: str) -> bool:
        return any(term in text for term in WEATHER_TERMS) or ("temperature" in text and "convert" not in text)

    def _should_calendar(self, text: str) -> bool:
        return any(term in text for term in CALENDAR_TERMS) or bool(re.search(r"\b(today|tomorrow|next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)|\d{4}-\d{2}-\d{2})\b", text))

    def _should_convert(self, text: str) -> bool:
        if any(term in text for term in CONVERT_TERMS):
            return True
        if re.search(r"\b(?:same|that|it|this|again|instead)\b", text):
            return True
        return bool(re.search(r"\b\d+(?:\.\d+)?\s*(km|m|cm|mm|mile|miles|inch|inches|ft|feet|kg|g|lb|lbs|l|ml|c|f|celsius|fahrenheit)\b", text))

    def _should_currency(self, text: str) -> bool:
        return any(term in text for term in CURRENCY_TERMS) or bool(re.search(r"\b(?:usd|eur|gbp|jpy|inr|aed|sar|cny|cad|aud|chf|mxn)\b", text))

    def _should_sql(self, text: str) -> bool:
        return any(term in text for term in SQL_TERMS) or text.strip().startswith(("select", "insert", "update", "delete", "with"))

    def _extract_target_unit(self, text: str) -> str | None:
        match = re.search(r"\bto\s+([a-zA-Z°]+)\b", text)
        if not match:
            match = re.search(r"\bin\s+([a-zA-Z°]+)\b", text)
        if not match:
            return None
        return UNIT_ALIASES.get(match.group(1).lower(), match.group(1))

    def _extract_target_currency(self, text: str) -> str | None:
        match = re.search(r"\bto\s+([a-zA-Z]{3}|[a-zA-Z]+)\b", text)
        if not match:
            return None
        token = match.group(1).lower()
        return CURRENCY_ALIASES.get(token) or (token.upper() if len(token) == 3 else None)

    def _synthesize_sql_query(self, text: str) -> str | None:
        if "joined last month" in text and "user" in text:
            return "SELECT * FROM users WHERE joined_at >= DATE('now','-1 month');"
        if "failed payment" in text or "failed payments" in text:
            return "SELECT * FROM payments WHERE status = 'failed';"
        if "active subscription" in text or "active subscriptions" in text:
            return "SELECT * FROM subscriptions WHERE status = 'active';"
        if "recent order" in text or "recent orders" in text:
            return "SELECT * FROM orders ORDER BY created_at DESC LIMIT 20;"
        if "top customer" in text or "top customers" in text:
            return "SELECT customer_id, SUM(total) AS total_spend FROM orders GROUP BY customer_id ORDER BY total_spend DESC LIMIT 10;"
        if "all user" in text or "list user" in text:
            return "SELECT * FROM users;"
        return None


def tool_call_json(tool: str, args: dict[str, Any]) -> str:
    return f"<tool_call>{json.dumps({'tool': tool, 'args': args}, separators=(',', ':'), ensure_ascii=False)}</tool_call>"


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
