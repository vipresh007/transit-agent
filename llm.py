"""Calling the model: retries, rate limits, quota, failover, accounting.

The interesting content here is not "wrap it in a retry" — it's knowing which
failures are worth retrying, which are worth waiting out, and which mean stop:

  transient 5xx / timeout   retry with exponential backoff + jitter
  per-minute 429            wait the delay the server states, then retry
  daily quota 429           waiting won't help; fail over to another provider
  tool_use_failed 400       the MODEL emitted bad JSON; resample
  any other 400             OUR request is malformed; retrying fails slower
  404 model not found       config error; say so and stop

Every one of those distinctions came from a real failure. Getting them wrong
costs either quota (retrying a daily cap) or a whole run (giving up on a
five-minute window).
"""

import os
import random
import re
import sys
import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

import cache
import providers
from tools import TOOL_SCHEMAS

RETRIABLE = (InternalServerError, RateLimitError, APIConnectionError, APITimeoutError)

# Requests are the obvious thing to count and the wrong one: Groq caps tokens
# per day, and a single run can spend 15k because every turn resends the whole
# conversation. Track both; the token number is the one that surprises you.
USAGE = {"n": 0, "prompt_tokens": 0, "completion_tokens": 0}


class DailyQuotaExhausted(RuntimeError):
    """Out of requests for the day. Waiting will not help."""


def _is_daily_quota(exc: Exception) -> bool:
    """Distinguish 'slow down' from 'come back tomorrow'.

    Both arrive as 429. A per-minute limit clears in seconds; a per-day quota
    does not, so retrying it just spends more of a budget already gone.
    """
    text = str(exc)
    return "PerDay" in text or "per day" in text.lower()


def server_retry_delay(exc: Exception) -> float | None:
    """Providers usually state how long to wait. Believe them over our guess.

      Google: "Please retry in 28.14s" / "'retryDelay': '28s'"
      Groq:   "Please try again in 5m4.128s"
    """
    text = str(exc)
    compound = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s", text)
    if compound:
        h, m, s = compound.groups()
        return int(h or 0) * 3600 + int(m or 0) * 60 + float(s)
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", text)
    if match:
        return float(match.group(1))
    match = re.search(r"'retryDelay': '(\d+(?:\.\d+)?)s'", text)
    return float(match.group(1)) if match else None


def usage_line() -> str:
    p, c = USAGE["prompt_tokens"], USAGE["completion_tokens"]
    parts = [
        f"{USAGE['n']} requests",
        f"{p + c:,} tokens ({p:,} in / {c:,} out)",
        providers.describe(),
    ]
    if cache.summary():
        parts.append(cache.summary())
    return " | ".join(parts)


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(f"  {msg}", file=sys.stderr)


def _backoff(exc, attempt: int, attempts: int, delay: float, verbose: bool) -> None:
    if attempt == attempts - 1:
        raise exc
    # Jitter matters: without it, everything that failed together retries
    # together and recreates the spike you were backing off from.
    wait = server_retry_delay(exc) or (delay + random.uniform(0, 1))
    _log(f"! {type(exc).__name__} — retrying in {wait:.1f}s "
         f"(attempt {attempt + 2}/{attempts})", verbose)
    time.sleep(wait)


def call_model(messages, attempts: int = 5, verbose: bool = True, use_tools: bool = True):
    """Send a request, surviving the ways free tiers fail."""
    sent = providers.sanitize(messages)

    if cache.ENABLED:
        key = cache.key_for(providers.model(), sent, use_tools, providers.TEMPERATURE)
        hit = cache.read(key)
        if hit is not None:
            cache.STATS["hits"] += 1
            _log(f". cache hit ({key[:8]})", verbose)
            return hit
        cache.STATS["misses"] += 1

    delay = 2.0
    last_exc: Exception | None = None
    waits = 0

    for attempt in range(attempts):
        try:
            providers.throttle()
            USAGE["n"] += 1
            response = providers.client().chat.completions.create(
                model=providers.model(),
                messages=sent,
                temperature=providers.TEMPERATURE,
                # Omitting tools entirely is what forces a text answer. Asking
                # nicely in the prompt is not reliable; removing the option is.
                tools=TOOL_SCHEMAS if use_tools else None,
                extra_body=providers.extra_body(),
            )
            usage = getattr(response, "usage", None)
            if usage:
                USAGE["prompt_tokens"] += usage.prompt_tokens or 0
                USAGE["completion_tokens"] += usage.completion_tokens or 0
            if cache.ENABLED:
                cache.write(key, response)
            return response

        except RateLimitError as exc:
            last_exc = exc
            stated = server_retry_delay(exc)

            # A stated delay beats our classification: Groq's "tokens per day"
            # limit is a ROLLING window that can clear in minutes, and treating
            # it as terminal threw away a run that needed to wait five.
            if stated is not None and stated <= providers.MAX_WAIT_SECONDS:
                waits += 1
                # Throttled twice means saturated, not momentarily busy.
                if waits >= 2 and providers.switch(_switch_note(verbose)):
                    _log(f"~ still rate limited — switching to "
                         f"{providers.describe()}", verbose)
                    waits = 0
                    continue
                _log(f"~ rate limited; server says retry in {stated:.0f}s "
                     f"— waiting (attempt {attempt + 1}/{attempts})", verbose)
                time.sleep(stated + 1)
                continue

            if _is_daily_quota(exc):
                spent = providers.describe()
                if providers.switch(_switch_note(verbose)):
                    _log(f"~ {spent} daily quota exhausted — failing over to "
                         f"{providers.describe()}", verbose)
                    continue
                # Say WHY there's no fallback. "No other provider is
                # configured" was actively misleading when the real cause was
                # PROVIDER= pinning one and filtering the rest out — the user
                # had three other keys sitting in .env.
                pinned = os.getenv("PROVIDER")
                if pinned:
                    reason = (
                        f"PROVIDER={pinned} is set in .env, which pins this "
                        f"provider and disables failover.\n"
                        f"Comment it out to fall through to the rest of the "
                        f"chain, or point it at another provider."
                    )
                else:
                    reason = (
                        "No other provider has a key configured.\n"
                        "  - Add GROQ_API_KEY / MISTRAL_API_KEY / "
                        "CEREBRAS_API_KEY to .env\n"
                        "  - or set OLLAMA_ENABLED=1 for unlimited local"
                    )
                raise DailyQuotaExhausted(
                    f"Daily quota exhausted for {spent}.\n"
                    f"Used {USAGE['n']} requests this run.\n\n{reason}"
                ) from exc

            _backoff(exc, attempt, attempts, max(delay, 15.0), verbose)
            delay = min(delay * 2, 60.0)

        except BadRequestError as exc:
            # Most 400s mean OUR request is malformed. But `tool_use_failed`
            # means the MODEL emitted tool arguments that aren't valid JSON --
            # a sampling accident. The provider rejects it before we see the
            # message, so feeding the error back isn't possible; resample.
            if "tool_use_failed" not in str(exc) or attempt == attempts - 1:
                raise
            last_exc = exc
            _log("! model emitted malformed tool-call JSON — resampling", verbose)
            time.sleep(1 + random.uniform(0, 1))

        except RETRIABLE as exc:
            last_exc = exc
            if attempt == attempts - 1:
                raise
            _backoff(exc, attempt, attempts, delay, verbose)
            delay *= 2

    # Every `continue` above falls through to here once attempts run out.
    # Without this the function returns None and the caller dies on
    # `response.choices` with an AttributeError naming the wrong problem.
    # A retry loop must end in a value or an exception, never off the bottom.
    raise last_exc or RuntimeError(
        f"call_model exhausted {attempts} attempts without a response"
    )


def _switch_note(verbose: bool):
    """Warn that failover invalidates the cache for the rest of the run."""
    def note():
        if cache.ENABLED:
            _log(". note: provider switch invalidates the cache for the rest "
                 "of this run — pin PROVIDER= in .env for reproducible replays",
                 verbose)
    return note
