"""LLM-based ticket triage with validation and a deterministic fallback.

Flow:
  1. Ask the LLM for strict JSON (summary, category, priority, confidence).
  2. Extract + validate the JSON against the LLMTriage schema.
  3. Reject low-confidence or malformed output; retry once.
  4. If the LLM is unavailable or unreliable, fall back to keyword rules.
Every attempt's outcome is returned so it can be stored for review.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Protocol

import httpx
from pydantic import ValidationError

from .schemas import CATEGORIES, PRIORITIES, LLMTriage

log = logging.getLogger("triage")

MIN_CONFIDENCE = float(os.getenv("TRIAGE_MIN_CONFIDENCE", "0.5"))
MAX_ATTEMPTS = 2

SYSTEM_PROMPT = f"""You are a customer-support triage assistant.
Read the ticket and respond with ONLY a JSON object, no prose, with keys:
  "summary": one or two sentences (max 300 chars) describing the issue,
  "category": one of {list(CATEGORIES)},
  "priority": one of {list(PRIORITIES)},
  "confidence": number 0-1 for how sure you are of category and priority.
Priority guide: urgent = outage, security issue, or data loss; high = customer blocked
or charged incorrectly; medium = degraded but workable; low = questions and requests."""


@dataclass
class TriageOutcome:
    source: str  # "llm" | "fallback"
    model: str | None
    summary: str
    category: str
    priority: str
    confidence: float | None
    raw_output: str | None = None
    error: str | None = None


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient:
    def __init__(self, api_key: str, model: str | None = None, timeout: float = 20.0):
        self.api_key = api_key
        self.model = model or os.getenv("LLM_MODEL", "claude-haiku-5-5")
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 400,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return "".join(b.get("text", "") for b in resp.json()["content"])


class OpenAIClient:
    def __init__(self, api_key: str, model: str | None = None, timeout: float = 20.0):
        self.api_key = api_key
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def default_client() -> LLMClient | None:
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if provider in ("", "anthropic") and os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicClient(os.environ["ANTHROPIC_API_KEY"])
    if provider in ("", "openai") and os.getenv("OPENAI_API_KEY"):
        return OpenAIClient(os.environ["OPENAI_API_KEY"])
    return None


# ---------- validation ----------

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_llm_output(text: str) -> LLMTriage:
    """Pull the first JSON object out of the model text and validate it."""
    match = _JSON_BLOCK.search(text or "")
    if not match:
        raise ValueError("no JSON object in model output")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e.msg}") from e
    try:
        result = LLMTriage.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"schema validation failed: {e.errors()[0]['msg']}") from e
    if result.confidence < MIN_CONFIDENCE:
        raise ValueError(f"confidence {result.confidence:.2f} below {MIN_CONFIDENCE}")
    return result


# ---------- deterministic fallback ----------

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing": ("invoice", "charge", "charged", "refund", "payment", "billing", "subscription", "price"),
    "account": ("password", "login", "log in", "sign in", "locked", "account", "2fa", "email change"),
    "bug": ("bug", "error", "crash", "broken", "doesn't work", "not working", "exception", "500"),
    "technical": ("api", "integration", "install", "configure", "setup", "timeout", "slow", "sync"),
    "feature_request": ("feature", "would be nice", "request", "suggestion", "add support", "wish"),
}
URGENT_KEYWORDS = ("outage", "down for everyone", "security", "breach", "hacked", "data loss", "all users")
HIGH_KEYWORDS = ("urgent", "asap", "cannot", "can't", "blocked", "double charged", "charged twice", "locked out")
LOW_KEYWORDS = ("question", "how do i", "wondering", "feature", "suggestion", "when will")


def rule_based_triage(subject: str, body: str) -> tuple[str, str, str]:
    text = f"{subject} {body}".lower()

    scores = {cat: sum(kw in text for kw in kws) for cat, kws in CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    category = best if scores[best] > 0 else "general"

    if any(k in text for k in URGENT_KEYWORDS):
        priority = "urgent"
    elif any(k in text for k in HIGH_KEYWORDS):
        priority = "high"
    elif any(k in text for k in LOW_KEYWORDS):
        priority = "low"
    else:
        priority = "medium"

    first_sentence = re.split(r"(?<=[.!?])\s+", body.strip())[0]
    summary = f"{subject.strip()}: {first_sentence}"[:300]
    return summary, category, priority


# ---------- entry point ----------


def triage_ticket(subject: str, body: str, client: LLMClient | None = None) -> TriageOutcome:
    client = client if client is not None else default_client()
    errors: list[str] = []
    last_raw: str | None = None

    if client is not None:
        user_msg = f"Subject: {subject}\n\nBody:\n{body}"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                last_raw = client.complete(SYSTEM_PROMPT, user_msg)
                result = parse_llm_output(last_raw)
                return TriageOutcome(
                    source="llm",
                    model=client.model,
                    summary=result.summary,
                    category=result.category,
                    priority=result.priority,
                    confidence=result.confidence,
                    raw_output=last_raw,
                )
            except (httpx.HTTPError, KeyError, ValueError) as e:
                msg = f"attempt {attempt}: {type(e).__name__}: {e}"
                log.warning("LLM triage failed, %s", msg)
                errors.append(msg)
    else:
        errors.append("no LLM configured")

    summary, category, priority = rule_based_triage(subject, body)
    return TriageOutcome(
        source="fallback",
        model=getattr(client, "model", None),
        summary=summary,
        category=category,
        priority=priority,
        confidence=None,
        raw_output=last_raw,
        error="; ".join(errors),
    )
