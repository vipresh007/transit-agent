"""Calling the model: retries, rate limits, quota, failover, accounting.

The interesting content here is not "wrap it in a retry" — it's knowing which
failures are worth retrying, which are worth waiting out, and which mean stop:

  transient 5xx / timeout   retry with exponential backoff + jitter
  per-minute 429            wait the delay the server states, then retry
  daily quota 429           waiting won't help; fail over to another provider
  tool_use_failed 400       the MODEL emitted bad JSON; resample
  output_parse_failed 400   the MODEL wrote prose where a tool call belonged;
                            resample
  413 request too large     the conversation exceeds this provider's
                            per-minute token budget; waiting cannot shrink it
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
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from transit.core import cache
from transit.core import providers
from transit.tools import TOOL_SCHEMAS

RETRIABLE = (InternalServerError, RateLimitError, APIConnectionError, APITimeoutError)

# Requests are the obvious thing to count and the wrong one: Groq caps tokens
# per day, and a single run can spend 15k because every turn resends the whole
# conversation. Track both; the token number is the one that surprises you.
USAGE = {"n": 0, "prompt_tokens": 0, "completion_tokens": 0}

# "The run was slow" is three different problems with three different fixes,
# and one number cannot tell them apart:
#
#   model_seconds   time the provider spent generating. Fix with a smaller
#                   prompt, fewer tools in scope, THINKING_BUDGET=0.
#   wait_seconds    time WE spent asleep on rate limits and backoff. Fix by
#                   pinning a provider, pacing, or waiting for quota to reset.
#   throttle_seconds  our own deliberate pacing between requests.
#   failed_seconds  round trips that came back an error. Real time, but
#                   neither generation nor sleeping — a rejected request still
#                   crosses the network twice.
#
# A 60-second run that was 55s generating and a 60-second run that was 55s
# sleeping look identical from outside and share no remedy. Measuring the
# aggregate would have sent us optimising the prompt when the real answer was
# "you burned Groq's daily quota an hour ago".
TIMING = {"model_seconds": 0.0, "wait_seconds": 0.0, "throttle_seconds": 0.0,
          "failed_seconds": 0.0, "latencies": []}


def reset_timing() -> None:
    TIMING.update(model_seconds=0.0, wait_seconds=0.0, throttle_seconds=0.0,
                  failed_seconds=0.0, latencies=[])


def reset_run() -> None:
    """Zero every per-run counter. Call once at the start of a pipeline.

    USAGE, TIMING and the cache stats are module globals, which is fine for a
    command-line program: the process dies after one run. Streamlit's process
    lives for hours across many runs, so without this the second question
    reports the first one's requests, tokens and seconds added to its own —
    the totals only ever climb.

    Same shape as every other global-state bug in this project: state scoped
    to the process, used as though it were scoped to the operation.
    """
    USAGE.update(n=0, prompt_tokens=0, completion_tokens=0)
    reset_timing()
    cache.STATS.update(hits=0, misses=0)


def _slept(seconds: float, bucket: str = "wait_seconds") -> None:
    """Sleep, and remember that we did. Every sleep site routes through here
    so no waiting can go unaccounted for."""
    TIMING[bucket] += seconds
    time.sleep(seconds)


def timing_summary() -> dict:
    """Numbers for the trace. Percentiles need more than a handful of calls,
    so report the shape that's actually meaningful: total, worst, typical."""
    lat = sorted(TIMING["latencies"])
    return {
        "model_seconds": round(TIMING["model_seconds"], 1),
        "wait_seconds": round(TIMING["wait_seconds"], 1),
        "throttle_seconds": round(TIMING["throttle_seconds"], 1),
        "failed_seconds": round(TIMING["failed_seconds"], 1),
        "calls": len(lat),
        "slowest_call": lat[-1] if lat else 0.0,
        "median_call": lat[len(lat) // 2] if lat else 0.0,
        "latencies": lat,
    }


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

    # Two numbers, because they mean different things and have different
    # fixes. Waiting is only shown when there was some — a clean run
    # shouldn't carry a "0.0s waiting" that trains you to ignore the field.
    timing = [f"{TIMING['model_seconds']:.0f}s model"]
    if TIMING["wait_seconds"] >= 1:
        timing.append(f"{TIMING['wait_seconds']:.0f}s waiting")
    if TIMING["throttle_seconds"] >= 1:
        timing.append(f"{TIMING['throttle_seconds']:.0f}s paced")
    if TIMING["failed_seconds"] >= 1:
        timing.append(f"{TIMING['failed_seconds']:.0f}s rejected")
    if TIMING["latencies"]:
        timing.append(f"slowest {max(TIMING['latencies']):.0f}s")
    parts.append(", ".join(timing))

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
    _slept(wait)


def call_model(messages, attempts: int = 5, verbose: bool = True, use_tools: bool = True):
    """Send a request, surviving the ways free tiers fail."""
    delay = 2.0
    last_exc: Exception | None = None
    waits = 0
    counted_miss = False
    last_stated: float | None = None

    for attempt in range(attempts):
        # Sanitize and key INSIDE the loop. Both depend on which provider is
        # active, and a mid-loop failover changes that. Computing them once up
        # front resent a Gemini-sanitized payload — thought_signature and all —
        # to Mistral, which 422'd on a field it has never heard of. The
        # sanitizer was right; it just ran before the thing it depended on
        # changed.
        sent = providers.sanitize(messages)
        key = None

        if cache.ENABLED:
            key = cache.key_for(providers.model(), sent, use_tools,
                                providers.TEMPERATURE)
            hit = cache.read(key)
            if hit is not None:
                cache.STATS["hits"] += 1
                _log(f". cache hit ({key[:8]})", verbose)
                return hit
            if not counted_miss:
                cache.STATS["misses"] += 1
                counted_miss = True

        try:
            paced = time.perf_counter()
            providers.throttle()
            TIMING["throttle_seconds"] += time.perf_counter() - paced

            USAGE["n"] += 1
            started = time.perf_counter()
            attempted = started
            response = providers.client().chat.completions.create(
                model=providers.model(),
                messages=sent,
                temperature=providers.TEMPERATURE,
                # Omitting tools entirely is what forces a text answer. Asking
                # nicely in the prompt is not reliable; removing the option is.
                tools=TOOL_SCHEMAS if use_tools else None,
                extra_body=providers.extra_body(),
            )
            # Only a request that RETURNED counts as model time. A call that
            # 429s spent real seconds too, but attributing those to generation
            # would make a rate-limited run look like a slow model.
            elapsed = time.perf_counter() - started
            TIMING["model_seconds"] += elapsed
            TIMING["latencies"].append(round(elapsed, 2))

            usage = getattr(response, "usage", None)
            if usage:
                USAGE["prompt_tokens"] += usage.prompt_tokens or 0
                USAGE["completion_tokens"] += usage.completion_tokens or 0
            if cache.ENABLED and key:
                cache.write(key, response)
            return response

        except RateLimitError as exc:
            # A rejected request still crossed the network twice. Excluding it
            # entirely left 52 seconds of a 325-second run filed under "our own
            # python", which reads like a performance bug in our code rather
            # than 14 requests bouncing off a rate limit.
            TIMING["failed_seconds"] += time.perf_counter() - attempted
            last_exc = exc
            stated = server_retry_delay(exc)

            # IS THE WAIT GETTING US ANYWHERE? A rolling window DRAINS: each
            # wait leaves less to wait. A hard daily cap does not. Gemini's
            # free tier is 20 requests/day for this model and it answered
            # "retry in 57s" four times running — we spent nearly four minutes
            # asleep to learn nothing, then raised a raw traceback.
            #
            # A delay that stops shrinking is the signal, and unlike parsing
            # each vendor's quota vocabulary it works for vendors we haven't
            # met. Still gated on _is_daily_quota so a genuinely busy minute
            # isn't mistaken for an exhausted day.
            stalled = (_is_daily_quota(exc) and stated is not None
                       and last_stated is not None and stated >= last_stated - 1)
            if stalled:
                _log(f"~ retry delay is not shrinking ({last_stated:.0f}s -> "
                     f"{stated:.0f}s) — this is a daily cap, not a busy minute",
                     verbose)
            last_stated = stated

            # A stated delay beats our classification: Groq's "tokens per day"
            # limit is a ROLLING window that can clear in minutes, and treating
            # it as terminal threw away a run that needed to wait five.
            if (not stalled and stated is not None
                    and stated <= providers.MAX_WAIT_SECONDS):
                waits += 1
                # Throttled twice means saturated, not momentarily busy.
                if waits >= 2 and providers.switch(_switch_note(verbose)):
                    _log(f"~ still rate limited — switching to "
                         f"{providers.describe()}", verbose)
                    waits = 0
                    continue
                _log(f"~ rate limited; server says retry in {stated:.0f}s "
                     f"— waiting (attempt {attempt + 1}/{attempts})", verbose)
                _slept(stated + 1)
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
            TIMING["failed_seconds"] += time.perf_counter() - attempted
            # Most 400s mean OUR request is malformed, and retrying just fails
            # slower. Two of Groq's do not — they mean the MODEL produced
            # something the provider's own parser rejected, which is a sampling
            # accident and very likely to come out fine next time:
            #
            #   tool_use_failed      tool arguments that aren't valid JSON
            #   output_parse_failed  reasoning text where a tool call belonged.
            #                        Seen mid-plan: the model wrote a paragraph
            #                        of "we need to find stop IDs..." and never
            #                        emitted the call.
            #
            # Both are rejected server-side, so we never receive the message
            # and cannot feed the error back the way a tool error is fed back.
            # Resampling is the only move. Distinguishing these from a genuinely
            # malformed request matters: treating all 400s as retriable hides
            # our own bugs behind five slow attempts, and treating none as
            # retriable kills a run over one bad sample.
            resamplable = ("tool_use_failed", "output_parse_failed")
            if not any(code in str(exc) for code in resamplable) \
                    or attempt == attempts - 1:
                raise
            last_exc = exc
            _log("! provider could not parse the model's output — resampling",
                 verbose)
            _slept(1 + random.uniform(0, 1))

        except RETRIABLE as exc:
            TIMING["failed_seconds"] += time.perf_counter() - attempted
            last_exc = exc
            if attempt == attempts - 1:
                raise
            _backoff(exc, attempt, attempts, delay, verbose)
            delay *= 2

        # LAST, deliberately. RateLimitError, BadRequestError AND
        # InternalServerError all subclass APIStatusError, so placing this
        # earlier swallowed 400s that should resample and 503s that should
        # retry. Python picks the FIRST matching handler, not the most
        # specific one — with an exception hierarchy, handler order IS the
        # logic. Catching a base class is only safe at the bottom.
        #
        # The test harness stubbed these as flat Exception subclasses, so it
        # passed either way. A double simpler than the real thing stops
        # testing the part that's actually hard; it now mirrors the SDK.
        except APIStatusError as exc:
            # 413: the request is larger than the provider's per-minute token
            # budget. Groq free tier is 8,000 TPM and our conversation reached
            # 9,489 — no amount of waiting makes a single request smaller than
            # a cap it already exceeds, so this is a hard "not here", not a
            # "not yet". Distinguishing the two is the same call as daily
            # quota vs busy minute, one layer down.
            if getattr(exc, "status_code", None) != 413:
                raise
            TIMING["failed_seconds"] += time.perf_counter() - attempted
            last_exc = exc
            over = providers.describe()
            if providers.switch(_switch_note(verbose)):
                _log(f"~ conversation too large for {over}'s per-minute "
                     f"budget — switching to {providers.describe()}", verbose)
                continue
            raise DailyQuotaExhausted(
                f"The conversation no longer fits {over}'s per-minute token "
                f"budget, and no other provider is available.\n"
                f"{USAGE['prompt_tokens']:,} prompt tokens used this run.\n\n"
                f"This is a SIZE problem, not a quota one — waiting won't "
                f"help. Shorten the conversation: fewer tools in scope, a "
                f"smaller MAX_RESULT_CHARS, or a provider with a larger "
                f"per-minute allowance."
            ) from exc

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
