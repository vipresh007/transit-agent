"""Shared test scaffolding: fake the model, keep everything else real.

These tests never touch the network. The OpenAI SDK is stubbed before any
project module imports it, so `providers`, `llm` and `agent` load normally but
every request is answered by a script you write per test.

That's deliberate. The loop's guardrails — retries, failover, duplicate
blocking, the times pushback — all exist because of specific model behaviour,
and the only way to test them is to reproduce that behaviour on demand. You
cannot make a real provider return a malformed tool call when you want one.
"""

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class FakeCompletion:
    """Stands in for openai's ChatCompletion, including cache round-tripping."""

    def __init__(self, content):
        self._content = content

    @property
    def choices(self):
        m = MagicMock()
        m.message.content = self._content
        m.message.tool_calls = None
        return [m]

    usage = None

    def model_dump_json(self):
        return json.dumps({"c": self._content})

    @classmethod
    def model_validate_json(cls, raw):
        return cls(json.loads(raw)["c"])


def install_fake_openai():
    """Stub the SDK. Must run before importing any project module."""
    om = types.ModuleType("openai")
    for name in (
        "InternalServerError", "RateLimitError", "APIConnectionError",
        "APITimeoutError", "NotFoundError", "BadRequestError",
    ):
        setattr(om, name, type(name, (Exception,), {}))
    om.OpenAI = lambda **kw: MagicMock()

    types_mod = types.ModuleType("openai.types")
    chat_mod = types.ModuleType("openai.types.chat")
    chat_mod.ChatCompletion = FakeCompletion

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *a, **k: None

    sys.modules.update({
        "openai": om,
        "openai.types": types_mod,
        "openai.types.chat": chat_mod,
        "dotenv": dotenv,
    })
    return om


def clean_env(**overrides):
    """A predictable environment: two providers, no cache, temp traces."""
    for key in (
        "PROVIDER", "OLLAMA_ENABLED", "CEREBRAS_API_KEY", "MISTRAL_API_KEY",
        "OPENROUTER_API_KEY", "THINKING_BUDGET",
    ):
        os.environ.pop(key, None)
    os.environ.update({
        "GEMINI_API_KEY": "test-gemini",
        "GROQ_API_KEY": "test-groq",
        "CACHE": "0",
        "TEMPERATURE": "0",
        "TRACE_DIR": "/tmp/test-traces",
    })
    os.environ.update(overrides)


# --- building fake model responses ------------------------------------------

def tool_call(call_id, name, args):
    tc = MagicMock()
    tc.id = str(call_id)
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


class _ToolMessage:
    def __init__(self, calls):
        self.content = None
        self.tool_calls = calls

    def model_dump(self, exclude_none=False):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": t.id,
                    "type": "function",
                    "function": {
                        "name": t.function.name,
                        "arguments": t.function.arguments,
                    },
                }
                for t in self.tool_calls
            ],
        }


def calls(*specs):
    """A model turn that requests tools: calls(("geocode", {"place": "X"}))."""
    made = [tool_call(i, n, a) for i, (n, a) in enumerate(specs, 1)]
    return MagicMock(choices=[MagicMock(message=_ToolMessage(made))])


def says(text):
    """A model turn that answers in text."""
    m = MagicMock()
    m.content = text
    m.tool_calls = None
    m.model_dump = lambda exclude_none=False: {"role": "assistant", "content": text}
    return MagicMock(choices=[MagicMock(message=m)])


# --- tiny assertion helpers -------------------------------------------------

PASSED = {"n": 0}


def check(label: str, actual, expected=True):
    ok = actual == expected
    PASSED["n"] += ok
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}" + ("" if ok else f"\n         want {expected!r}, got {actual!r}"))
    if not ok:
        raise AssertionError(label)


def section(title: str):
    print(f"\n{title}")
