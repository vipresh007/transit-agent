"""Provider configuration and failover.

Every provider here speaks the OpenAI chat-completions format, which is why
we use the OpenAI SDK rather than any vendor's native one. When a free tier
runs dry mid-run we move to the next and carry the conversation with us.

ACCESSORS, NOT GLOBALS. `current()`, `model()` and `client()` are functions
because failover REASSIGNS the active provider at runtime. When this was a
module-level `provider` variable, `from providers import provider` bound the
old value forever and plan.py cheerfully reported "gemini" after failing over
to Groq. A function can't go stale.
"""

import os
import sys
import time

from openai import OpenAI

PROVIDERS = [
    {
        "name": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": os.getenv("MODEL", "gemini-3.6-flash"),
    },
    {
        "name": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    },
    {
        "name": "cerebras",
        "base_url": "https://api.cerebras.ai/v1",
        "key_env": "CEREBRAS_API_KEY",
        "model": os.getenv("CEREBRAS_MODEL", "llama-3.3-70b"),
    },
    {
        "name": "mistral",
        "base_url": "https://api.mistral.ai/v1",
        "key_env": "MISTRAL_API_KEY",
        "model": os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
        # Free tier throttles per second and its 429 carries no retry delay,
        # so backing off after the fact is guesswork. Pacing requests avoids
        # the limit instead of reacting to it.
        "min_interval": 1.5,
    },
    {
        "name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": os.getenv("OPENROUTER_MODEL", "qwen/qwen3-235b-a22b:free"),
    },
    {
        "name": "ollama",
        "base_url": os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
        "key_env": "OLLAMA_ENABLED",
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    },
]

AVAILABLE = [p for p in PROVIDERS if os.getenv(p["key_env"])]
if not AVAILABLE:
    sys.exit("No provider configured. Set GEMINI_API_KEY or GROQ_API_KEY in .env")

# PROVIDER=groq forces a specific one and disables failover.
_forced = os.getenv("PROVIDER")
if _forced:
    AVAILABLE = [p for p in AVAILABLE if p["name"] == _forced] or sys.exit(
        f"PROVIDER={_forced} has no key configured"
    )


_active = 0
_client = OpenAI(
    api_key=os.getenv(AVAILABLE[0]["key_env"]),
    base_url=AVAILABLE[0]["base_url"],
)


def current() -> dict:
    """The provider serving requests right now."""
    return AVAILABLE[_active]


def model() -> str:
    return current()["model"]


def client() -> OpenAI:
    return _client


def describe() -> str:
    return f"{current()['name']}:{model()}"


def is_gemini() -> bool:
    """Gemini needs special handling: thought signatures, thinking_config."""
    return current()["name"] == "gemini"


def switch(on_switch=None) -> bool:
    """Move to the next configured provider. False if none are left."""
    global _active, _client
    if _active + 1 >= len(AVAILABLE):
        return False
    if on_switch:
        on_switch()
    _active += 1
    _client = OpenAI(
        api_key=os.getenv(current()["key_env"]), base_url=current()["base_url"]
    )
    return True


def sanitize(messages: list) -> list:
    """Strip provider-specific extras before sending history elsewhere.

    Gemini attaches thought_signature to tool calls. Groq has never seen that
    field and may reject it. Failing over mid-conversation means handing one
    vendor's history to another, so scrub anything vendor-specific first.
    """
    if is_gemini():
        return messages

    cleaned = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m = dict(m)
            m["tool_calls"] = [
                {k: v for k, v in tc.items() if k != "extra_content"}
                for tc in m["tool_calls"]
            ]
        cleaned.append(m)
    return cleaned


# --- pacing ----------------------------------------------------------------
# Some providers cap requests per second and return a bare 429 with no
# guidance, making reactive backoff pure guesswork. Spacing requests out is
# cheaper than being throttled.
_last_request_at = {"t": 0.0}


def throttle() -> None:
    interval = current().get("min_interval", 0.0) or float(
        os.getenv("MIN_REQUEST_INTERVAL", "0")
    )
    if interval <= 0:
        return
    elapsed = time.monotonic() - _last_request_at["t"]
    if elapsed < interval:
        time.sleep(interval - elapsed)
    _last_request_at["t"] = time.monotonic()


# --- Gemini-only request knobs ---------------------------------------------
# Gemini's thinking models sign their reasoning and attach it to tool calls in
# a non-standard field. Set THINKING_BUDGET=0 to disable reasoning entirely.
THINKING_BUDGET = os.getenv("THINKING_BUDGET")

EXTRA_BODY = {}
if THINKING_BUDGET is not None:
    EXTRA_BODY = {
        "extra_body": {
            "google": {"thinking_config": {"thinking_budget": int(THINKING_BUDGET)}}
        }
    }


def extra_body():
    """Gemini-only knobs. Sending these to Groq would 400."""
    return (EXTRA_BODY or None) if is_gemini() else None


TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))
MAX_WAIT_SECONDS = float(os.getenv("MAX_WAIT_SECONDS", "420"))
