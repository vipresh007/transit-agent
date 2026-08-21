"""
Stage 1: the agent loop, written out by hand.

No framework. ~60 lines of real logic. Read it once and you will understand
what LangGraph and CrewAI are doing underneath, which makes choosing between
them a lot easier later.

The loop:
    1. Send the conversation + tool schemas to the model.
    2. Model replies with either text (done) or tool calls (not done).
    3. If tool calls: run them, append the results, go back to 1.

That is the entire idea. Everything else in agent frameworks is ergonomics,
persistence, and error handling layered on top of this.

Run:  python agent.py "what's the weather like in Toronto this week?"
"""

import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from openai.types.chat import ChatCompletion
from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

from tools import SCHEDULE_TOOLS, TOOL_FUNCTIONS, TOOL_SCHEMAS

load_dotenv()

# ---------------------------------------------------------------------------
# Providers.
#
# Every one of these speaks the OpenAI chat-completions format, which is why
# we used the OpenAI SDK instead of Google's native one. When a free tier runs
# dry mid-run, we move to the next provider and CARRY THE CONVERSATION WITH US
# rather than starting over. Failing over to a backup is standard practice for
# anything that depends on someone else's rate limit.
#
# Order matters: first one with a key present is used first.
# ---------------------------------------------------------------------------
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
provider = AVAILABLE[0]
MODEL = provider["model"]
client = OpenAI(
    api_key=os.getenv(provider["key_env"]), base_url=provider["base_url"]
)

MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))

# Temperature 0 by default. Two reasons, both practical rather than about
# answer quality:
#   1. Reproducibility. Chasing a bug through a loop that takes a different
#      path each run is miserable.
#   2. The cache only works if runs are deterministic. One differing response
#      changes the message history for every later step, so every later key
#      misses. A stochastic multi-turn loop is effectively uncacheable.
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))


def switch_provider() -> bool:
    """Move to the next configured provider. False if none are left."""
    global _active, provider, client, MODEL
    if _active + 1 >= len(AVAILABLE):
        return False
    if CACHE_ENABLED:
        # Worth saying out loud: the model name is part of the cache key, so
        # switching mid-run means every subsequent step is a guaranteed miss,
        # and the next run will diverge at a different point. Caching and
        # failover fight each other; pin PROVIDER when you want replays.
        print(
            "  . note: provider switch invalidates the cache for the rest of "
            "this run — pin PROVIDER= in .env for reproducible replays",
            file=sys.stderr,
        )
    _active += 1
    provider = AVAILABLE[_active]
    MODEL = provider["model"]
    client = OpenAI(
        api_key=os.getenv(provider["key_env"]), base_url=provider["base_url"]
    )
    return True


def sanitize(messages: list) -> list:
    """Strip provider-specific extras before sending history elsewhere.

    Gemini attaches thought_signature to tool calls. Groq has never seen that
    field and may reject it. When you fail over mid-conversation you're handing
    one vendor's history to another, so scrub anything vendor-specific first.
    """
    if provider["name"] == "gemini":
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

# Gemini's thinking models sign their reasoning and attach the signature to
# each tool call in a NON-STANDARD field:
#     tool_calls[0].extra_content.google.thought_signature
# If you don't echo it back on the next turn, Gemini 400s with
# "Function call is missing a thought_signature".
#
# The OpenAI SDK keeps unknown fields in .model_extra, so the fix is to hand
# the whole message object back rather than rebuilding it field by field.
# Lesson: an OpenAI-compatible endpoint is compatible, not identical.
#
# Set THINKING_BUDGET=0 in .env to turn thinking off instead — no reasoning,
# no signatures, no problem. Cheaper and faster, but weaker at multi-step
# planning, which is most of what this project is about.
THINKING_BUDGET = os.getenv("THINKING_BUDGET")

EXTRA_BODY = {}
if THINKING_BUDGET is not None:
    EXTRA_BODY = {
        "extra_body": {
            "google": {"thinking_config": {"thinking_budget": int(THINKING_BUDGET)}}
        }
    }

# A model has no clock. Left to itself it assumes a date from its training
# data -- this one filtered the schedule on 2024-06-10 against a feed covering
# July-September 2026, got zero rows three times, and reported estimates.
# Anything time-relative ("this morning", "today", "next Monday") is
# unanswerable without this line.
TODAY = date.today()

SYSTEM_PROMPT = f"""You are a travel planning assistant for Toronto.

Today's date is {TODAY:%A, %d %B %Y} ({TODAY:%Y%m%d} in GTFS format).

Tools available: geocoding, weather, points of interest, and a SQL database
of the real TTC schedule.

Use them rather than guessing. You do not know today's weather, whether a
museum is open, or when the last streetcar runs, and inventing those details
makes you useless.

For a journey between two places, do exactly this:
  1. geocode each place to get coordinates
  2. plan_journey(origin_lat, origin_lon, dest_lat, dest_lon, after_time)

That's it. plan_journey searches nearby stops at both ends, finds a direct
ride or computes a single-transfer option with a real interchange, and returns
verified times for every leg. Two tool calls answer the whole question.

Do NOT hand-pick stops, guess an interchange, or write journey SQL yourself.
Every attempt at that produced empty results that were then misread as "no
service". find_direct_trips and query_transit exist for follow-ups
plan_journey doesn't cover — not as a substitute for it.

A stop_id is ONE PLATFORM serving ONE direction. College St at Augusta Ave
is stop 809 eastbound and stop 12338 westbound. Check the 'serves' headsigns
from find_nearby_stops to pick the platform pointing your way — querying the
wrong one returns zero rows, which reads exactly like "no service".

Never search the stops table by neighbourhood or landmark name. Stop names
are intersections ('College St at Augusta Ave'); 'Kensington Market' and
'Distillery District' do not appear in the data at all. Coordinates are the
only bridge between a place and its stops.

ONLY pair a route with stops that find_nearby_stops listed as serving it.
That tool returns a "routes" field per stop. Querying route 504 at a stop
whose routes are ["306","506"] returns nothing, and the empty result looks
identical to "no service" — you will misread it as a scheduling fact.

Do NOT filter on calendar.start_date or calendar.end_date. This feed has one
active service period, and a date outside it silently returns zero rows.
Selecting the right service_id is sufficient.

Vague times need a concrete departure time before you can query anything.
Use these unless the user gives one, and say which you assumed:
    morning 08:00 | midday 12:00 | afternoon 14:00
    evening 19:00 | night 22:00  | "now" the current time
Then ask for departures AFTER that time, not MIN() over the whole day —
MIN() gives you the 05:29 first run, which nobody means by "morning".

If a query returns an error, read it and write a corrected query rather than
giving up. If a query returns zero rows, suspect your filters before
concluding the service does not exist.

MINIMUM WORK BEFORE ANSWERING — this comes first:
Never state a departure or arrival time you did not retrieve from a tool.
For a journey question you MUST have called find_direct_trips (or
query_transit) and received actual clock times before you answer. Stopping
early and filling in plausible times is a failure, not efficiency — an
itinerary of invented times is worse than no itinerary.

If find_direct_trips reports that a transfer is required, that is the middle
of the job, not the end: pick an interchange and call it again for each leg.

BUDGET AND STOPPING — once you have real data:
Don't explore aimlessly. Before each ADDITIONAL query, ask: "will this change
my answer?" If it only adds confidence or detail, skip it. Never keep
querying to remove the last bit of uncertainty — state it as a caveat.

If a detail is ambiguous, pick the most reasonable reading, say which one
you picked, and move on. Do not run queries to resolve ambiguity.

Answer concisely and concretely, with real place names and real times. If
the data doesn't support a confident answer, say so plainly instead of
filling the gap."""


# Errors worth retrying: the server was busy, rate-limited us, or the network
# hiccuped. A 400 (bad request) is NOT here — retrying a malformed request just
# fails again more slowly.
RETRIABLE = (InternalServerError, RateLimitError, APIConnectionError, APITimeoutError)

# Requests are the obvious thing to count and the wrong one. Groq caps tokens
# per day (200k), and a single agent run can spend 15k of them, because every
# turn resends the entire conversation. Track both; the token number is the
# one that will surprise you.
REQUEST_COUNT = {"n": 0, "prompt_tokens": 0, "completion_tokens": 0}


class DailyQuotaExhausted(RuntimeError):
    """Out of requests for the day. Waiting will not help."""


def _is_daily_quota(exc: Exception) -> bool:
    """Distinguish 'slow down' from 'come back tomorrow'.

    Both arrive as 429. A per-minute limit clears in seconds and is worth
    retrying; a per-day quota does not clear until midnight Pacific, so
    retrying it just spends more of a budget you've already exhausted.
    """
    text = str(exc)
    return "PerDay" in text or "per day" in text.lower()


# How long we're willing to sit and wait for a rate limit to clear before
# giving up or failing over. Five minutes is annoying; an hour is a hang.
MAX_WAIT_SECONDS = float(os.getenv("MAX_WAIT_SECONDS", "420"))


def _server_retry_delay(exc: Exception) -> float | None:
    """Providers usually tell you exactly how long to wait. Believe them.

    Handles the formats seen in practice:
      Google: "Please retry in 28.142564072s" / "'retryDelay': '28s'"
      Groq:   "Please try again in 5m4.128s"
    """
    text = str(exc)

    # Groq's compound form: 5m4.128s, or 1h2m3s.
    compound = re.search(
        r"try again in (?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s", text
    )
    if compound:
        h, m, s = compound.groups()
        return int(h or 0) * 3600 + int(m or 0) * 60 + float(s)

    match = re.search(r"retry in (\d+(?:\.\d+)?)s", text)
    if match:
        return float(match.group(1))
    match = re.search(r"'retryDelay': '(\d+(?:\.\d+)?)s'", text)
    return float(match.group(1)) if match else None


def call_model(messages, attempts: int = 5, verbose: bool = True, use_tools: bool = True):
    """Call the model, retrying transient failures with exponential backoff.

    Free tiers are noisy: 503s under load and per-minute 429s are routine.
    Without retries, one blip discards every tool call made so far.

    But retries are not free -- each one spends quota. Retrying a daily-quota
    429 is strictly worse than failing fast, which is a mistake this code
    originally made. Knowing which errors are worth retrying is the actual
    skill; "wrap it in a retry" is not.
    """
    sent = sanitize(messages)

    if CACHE_ENABLED:
        key = _cache_key(MODEL, sent, use_tools)
        hit = _cache_read(key)
        if hit is not None:
            CACHE_STATS["hits"] += 1
            if verbose:
                print(f"  . cache hit ({key[:8]})", file=sys.stderr)
            return hit
        CACHE_STATS["misses"] += 1

    delay = 2.0
    last_exc: Exception | None = None
    waits = 0
    for attempt in range(attempts):
        try:
            throttle(verbose)
            REQUEST_COUNT["n"] += 1
            response = client.chat.completions.create(
                model=MODEL,
                messages=sent,
                temperature=TEMPERATURE,
                # Omitting tools entirely is what forces a text answer. Asking
                # nicely in the prompt is not reliable; removing the option is.
                tools=TOOL_SCHEMAS if use_tools else None,
                # Gemini-only knob; sending it to Groq would 400.
                extra_body=(EXTRA_BODY or None)
                if provider["name"] == "gemini"
                else None,
            )
            usage = getattr(response, "usage", None)
            if usage:
                REQUEST_COUNT["prompt_tokens"] += usage.prompt_tokens or 0
                REQUEST_COUNT["completion_tokens"] += usage.completion_tokens or 0
            if CACHE_ENABLED:
                _cache_write(key, response)
            return response
        except RateLimitError as exc:
            # A stated retry delay beats our own classification. Groq's
            # "tokens per day" limit is a ROLLING window that can clear in
            # minutes; treating it as terminal because the words "per day"
            # appear threw away a run that only needed to wait five minutes.
            # Trust the server's number over our guess about its semantics.
            last_exc = exc
            stated = _server_retry_delay(exc)
            if stated is not None and stated <= MAX_WAIT_SECONDS:
                waits += 1
                # Waiting twice and still being throttled means this provider
                # is saturated, not momentarily busy. Sitting through five
                # 57-second waits burns four minutes to reach the same place.
                # If we have a backup, use it.
                if waits >= 2 and switch_provider():
                    if verbose:
                        print(
                            f"  ~ still rate limited — switching to "
                            f"{provider['name']} ({MODEL})",
                            file=sys.stderr,
                        )
                    waits = 0
                    continue
                if verbose:
                    print(
                        f"  ~ rate limited; server says retry in {stated:.0f}s "
                        f"— waiting (attempt {attempt + 1}/{attempts})",
                        file=sys.stderr,
                    )
                time.sleep(stated + 1)
                continue

            if _is_daily_quota(exc):
                exhausted = provider["name"]
                # Waiting won't help, but another provider might.
                if switch_provider():
                    if verbose:
                        print(
                            f"  ~ {exhausted} daily quota exhausted — "
                            f"failing over to {provider['name']} ({MODEL})",
                            file=sys.stderr,
                        )
                    continue  # same request, new provider, history intact
                raise DailyQuotaExhausted(
                    f"Daily quota exhausted for {exhausted} ({MODEL}), and no "
                    f"other provider is configured.\n"
                    f"Used {REQUEST_COUNT['n']} requests this run.\n\n"
                    "Options:\n"
                    "  1. Add GROQ_API_KEY to .env — https://console.groq.com/keys\n"
                    "  2. `python list_models.py`, switch to a flash-lite model\n"
                    "  3. Install Ollama and set OLLAMA_ENABLED=1 for unlimited local\n"
                    "  4. Wait for reset (midnight Pacific for Gemini)"
                ) from exc
            # A rate limit with no stated delay is usually per-minute, so the
            # 2/4/8/16s schedule used for transient 5xx is too impatient --
            # it gives up after ~30s on a window that needs 60. Start higher.
            _backoff(exc, attempt, attempts, max(delay, 15.0), verbose)
            delay = min(delay * 2, 60.0)
        except BadRequestError as exc:
            # Most 400s mean OUR request is malformed — retrying is pointless.
            # But `tool_use_failed` means the MODEL emitted tool arguments
            # that aren't valid JSON (e.g. a brace inside a SQL string).
            # That's a sampling accident, and resampling usually fixes it.
            # The provider rejects it before we see the message, so feeding
            # the error back isn't an option — retrying is all we have.
            if "tool_use_failed" not in str(exc):
                raise
            if attempt == attempts - 1:
                raise
            if verbose:
                print(
                    "  ! model emitted malformed tool-call JSON — resampling",
                    file=sys.stderr,
                )
            last_exc = exc
            time.sleep(1 + random.uniform(0, 1))
        except RETRIABLE as exc:
            last_exc = exc
            if attempt == attempts - 1:
                raise
            _backoff(exc, attempt, attempts, delay, verbose)
            delay *= 2

    # Every `continue` above can fall through to here once attempts run out.
    # Without this the function returns None and the caller dies on
    # `response.choices` with an AttributeError that says nothing about the
    # rate limit that actually caused it. A retry loop must always end in a
    # value or an exception — never off the bottom.
    raise last_exc or RuntimeError(
        f"call_model exhausted {attempts} attempts without a response"
    )


def _backoff(exc, attempt: int, attempts: int, delay: float, verbose: bool) -> None:
    if attempt == attempts - 1:
        raise exc
    # Jitter matters: without it, every client that failed together retries
    # together and recreates the spike you were backing off from.
    wait = _server_retry_delay(exc) or (delay + random.uniform(0, 1))
    if verbose:
        print(
            f"  ! {type(exc).__name__} — retrying in {wait:.1f}s "
            f"(attempt {attempt + 2}/{attempts})",
            file=sys.stderr,
        )
    time.sleep(wait)


# Set by run(). Callers need to know whether an answer came from a completed
# investigation or from a truncated one — the text alone doesn't say, and a
# model that ran out of budget will still answer confidently.
# `truncated` was not enough. A run can stop voluntarily having learned
# nothing -- every query errored, and the model answered anyway. So we also
# record which tools actually RETURNED something usable. "Did the agent finish?"
# and "did the agent find anything?" are different questions, and only the
# second one tells you whether to trust the answer.
LAST_RUN = {
    "truncated": False,
    "steps": 0,
    "repeats": 0,
    "productive": set(),   # tools that returned real data
    "barren": set(),       # tools that only ever errored or came back empty
    "times_retrieved": 0,  # tool results that actually contained clock times
}

# "Did query_transit return rows?" turned out to be the wrong question. A run
# that only looked up `SELECT DISTINCT service_id FROM calendar` satisfies it
# while establishing no schedule whatsoever -- and then reports invented
# departure times with no warning. What we actually care about is whether a
# CLOCK TIME ever came back from the data. Matches '8:03:00' and '25:23:17'.
GTFS_TIME_IN_RESULT = re.compile(r"\b\d{1,2}:[0-5]\d:[0-5]\d\b")

# Imported from tools.py so it can't drift from the tool definitions.

# Result prefixes that mean "you learned nothing". Kept as a list rather than
# inlined so it's obvious what counts as progress and easy to adjust.
BARREN_MARKERS = (
    "error",
    "sql error",
    "no location found",
    "query returned no rows",
    "not found",
    "unknown category",
)


# Every tool result stays in the conversation for the rest of the run, and
# every subsequent request resends the whole thing. A 50-row JSON result isn't
# expensive once -- it's expensive eight more times. Token budgets (Groq bills
# 200k tokens/day) run out long before request counts do, so this is the lever
# that actually matters.
MAX_RESULT_CHARS = int(os.getenv("MAX_RESULT_CHARS", "2500"))


# ---------------------------------------------------------------------------
# Response cache.
#
# The dominant cost of iterating on an agent is not money, it's the wait. A
# 12-step run takes minutes and burns quota, and most code changes -- fixing
# a parser, a validator, a render function -- do not change what we send to
# the model at all. Replaying those responses makes the loop instant.
#
# Keyed on the exact request. Change a prompt and you get a miss, correctly:
# the cache never hides a real change in what the model was asked.
#
# Enable with CACHE=1. Off by default, because a cache that's silently on
# will eventually convince you a bug is fixed when you're reading a replay.
# ---------------------------------------------------------------------------
CACHE_ENABLED = os.getenv("CACHE") == "1"
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))
CACHE_STATS = {"hits": 0, "misses": 0}


# The tool schemas are part of every request, so they must be part of the key.
# Keying on a bare `use_tools` boolean meant editing a tool's description or
# parameters left the cache happily replaying answers from the old tools —
# I changed plan_journey's signature, reran, and got a byte-identical stale
# result. A cache must be keyed on the ENTIRE request, not the parts that
# were convenient to hash.
TOOLS_FINGERPRINT = hashlib.sha256(
    json.dumps(TOOL_SCHEMAS, sort_keys=True).encode()
).hexdigest()[:12]


def _cache_key(model: str, messages: list, use_tools: bool) -> str:
    blob = json.dumps(
        {
            "model": model,
            "messages": messages,
            "tools": TOOLS_FINGERPRINT if use_tools else None,
            "temperature": TEMPERATURE,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _cache_read(key: str):
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return ChatCompletion.model_validate_json(path.read_text("utf-8"))
    except Exception:
        # A corrupt or stale-schema entry should never break a run.
        path.unlink(missing_ok=True)
        return None


def _cache_write(key: str, response) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        (CACHE_DIR / f"{key}.json").write_text(
            response.model_dump_json(), encoding="utf-8"
        )
    except Exception:
        pass  # caching is an optimisation; never let it break the run


def cache_line() -> str:
    h, m = CACHE_STATS["hits"], CACHE_STATS["misses"]
    return f"cache {h} hit / {m} miss" if (h or m) else ""


# Pacing. Some providers cap requests per second and return a bare 429 with
# no guidance, which makes reactive backoff pure guesswork. Spacing requests
# out is cheaper than being throttled: a 1.5s pause costs less than a failed
# request plus an escalating backoff that may still be too short.
_last_request_at = {"t": 0.0}


def throttle(verbose: bool = False) -> None:
    interval = provider.get("min_interval", 0.0) or float(
        os.getenv("MIN_REQUEST_INTERVAL", "0")
    )
    if interval <= 0:
        return
    elapsed = time.monotonic() - _last_request_at["t"]
    if elapsed < interval:
        time.sleep(interval - elapsed)
    _last_request_at["t"] = time.monotonic()


# ---------------------------------------------------------------------------
# Tracing.
#
# Every run writes a full JSON record to traces/. Reading stderr scrollback is
# fine for a 3-step run and useless for a 12-step one -- and it loses the
# thing you most want later: the exact tool results the model was reasoning
# over. A trace file is greppable, diffable between runs, and shareable.
# ---------------------------------------------------------------------------
TRACE_DIR = Path(os.getenv("TRACE_DIR", "traces"))
TRACE = {"events": []}


def trace_event(kind: str, **fields) -> None:
    TRACE["events"].append({"kind": kind, "t": round(time.time(), 3), **fields})


def write_trace(question: str, answer: str = "", extra: dict | None = None) -> Path:
    TRACE_DIR.mkdir(exist_ok=True)
    path = TRACE_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    payload = {
        "question": question,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": provider["name"],
        "model": MODEL,
        "usage": dict(REQUEST_COUNT),
        "cache": dict(CACHE_STATS),
        "flags": {
            k: (sorted(v) if isinstance(v, set) else v)
            for k, v in LAST_RUN.items()
        },
        "events": TRACE["events"],
        "answer": answer,
        **(extra or {}),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # Also keep a stable filename so tooling can always find the newest run.
    (TRACE_DIR / "latest.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return path


def usage_line() -> str:
    """One-line cost summary. Prompt tokens dominate — that's the point."""
    p = REQUEST_COUNT["prompt_tokens"]
    c = REQUEST_COUNT["completion_tokens"]
    parts = [
        f"{REQUEST_COUNT['n']} requests",
        f"{p + c:,} tokens ({p:,} in / {c:,} out)",
        f"{provider['name']}:{MODEL}",
    ]
    if cache_line():
        parts.append(cache_line())
    return " | ".join(parts)


def clip(result: str) -> str:
    """Truncate an oversized tool result, telling the model it was truncated.

    Silent truncation is dangerous -- the model would treat a cut-off list as
    complete. Saying so lets it narrow the query instead.
    """
    if len(result) <= MAX_RESULT_CHARS:
        return result
    kept = result[:MAX_RESULT_CHARS]
    return (
        f"{kept}\n\n[TRUNCATED: {len(result) - MAX_RESULT_CHARS} more chars. "
        f"This result was too large. Re-run with a tighter WHERE clause, "
        f"fewer columns, or a smaller LIMIT to see the rest.]"
    )


def _is_barren(result: str) -> bool:
    low = result.strip().lower()
    return (
        not low
        or low.startswith(BARREN_MARKERS)
        or low.startswith("no ")          # "No museum found within 1500m."
    )


def run(user_message: str, verbose: bool = True, require_times: bool = False) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    LAST_RUN.update(
        truncated=False, steps=0, repeats=0, productive=set(), barren=set(),
        times_retrieved=0,
    )
    seen_calls: dict[str, int] = {}   # exact (tool, args) -> times requested
    barren: dict[str, int] = {}       # tool -> consecutive useless results
    pushed_back = False               # allow exactly one "go do the work"

    for step in range(MAX_STEPS):
        remaining = MAX_STEPS - step

        # The model has no idea how many steps it has left, so it explores as
        # if the budget were infinite and gets truncated mid-thought. Telling
        # it the budget converts a hard cutoff into a deadline it can plan for.
        if remaining == 3:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have 2 tool calls left. Stop exploring. Make at "
                        "most one more query if it's essential, then give your "
                        "final answer with whatever you have."
                    ),
                }
            )

        response = call_model(messages, verbose=verbose)
        message = response.choices[0].message

        # No tool calls means the model thinks it's finished.
        if not message.tool_calls:
            # ...but "finished" and "did the work" are different claims.
            # Prompting alone didn't hold: after the anti-thrash guidance told
            # it to answer quickly, it started stopping after two steps and
            # inventing times. A precondition checked in code is a guarantee;
            # a precondition described in a prompt is a suggestion.
            # One pushback only — if it still won't, the caller gets the
            # honest unverified answer rather than an infinite loop.
            if (
                require_times
                and LAST_RUN["times_retrieved"] == 0
                and not pushed_back
                and step < MAX_STEPS - 2
            ):
                pushed_back = True
                if verbose:
                    print(
                        "  [!] tried to answer with no retrieved times — "
                        "pushing back",
                        file=sys.stderr,
                    )
                trace_event("pushback", step=step, draft=message.content)
                messages.append(message.model_dump(exclude_none=True))
                messages.append({
                    "role": "user",
                    "content": (
                        "Stop. You have not retrieved a single scheduled time "
                        "from the data, so every time in that answer is "
                        "invented. Use find_direct_trips with the stop_ids you "
                        "already have. If it reports a transfer is required, "
                        "pick an interchange and call it once per leg. Do not "
                        "answer again until you have real times."
                    ),
                })
                continue

            trace_event("final", step=step, content=message.content)
            return message.content or "(empty response)"

        # The assistant turn must go into history before the tool results,
        # or the next request will 400 on a dangling tool_call_id.
        #
        # model_dump() rather than a hand-built dict: it round-trips provider
        # extras like Gemini's thought_signature, which a manual rebuild
        # silently drops. See the THINKING_BUDGET note above.
        messages.append(message.model_dump(exclude_none=True))

        LAST_RUN["steps"] = step + 1
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            # Anti-thrash. A model that gets an unhelpful result often retries
            # the same tool with a trivially different argument -- five
            # find_pois calls at radii 500/800/1000/1500/2000 -- burning the
            # budget without gaining information. Detect the pattern and say
            # so, instead of dutifully executing the same lookup again.
            signature = f"{name}:{json.dumps(args, sort_keys=True)}"
            seen_calls[signature] = seen_calls.get(signature, 0) + 1

            if seen_calls[signature] > 1:
                LAST_RUN["repeats"] += 1
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": (
                        f"You already called {name} with these exact arguments "
                        f"and got a result. Repeating it will not help. Use "
                        f"what you have, or try a different tool."
                    ),
                })
                if verbose:
                    print(f"  [{step}] {name}(...) — duplicate, blocked",
                          file=sys.stderr)
                continue

            # Throttle on FRUITLESS calls only. Counting every call punished
            # query_transit -- the workhorse -- for doing legitimate stepwise
            # exploration, which is worse than the thrashing it was meant to
            # prevent. What actually signals a stuck agent is repeatedly
            # getting nothing back, so that's what we count.
            if barren.get(name, 0) >= 3:
                LAST_RUN["repeats"] += 1
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": (
                        f"{name} has returned no useful result {barren[name]} "
                        f"times in a row. Varying the arguments further is "
                        f"unlikely to help. Use a different tool, or proceed "
                        f"with what you have and note the gap."
                    ),
                })
                if verbose:
                    print(f"  [{step}] {name}(...) — {barren[name]} barren "
                          f"results, blocked", file=sys.stderr)
                continue

            if verbose:
                print(f"  [{step}] {name}({args})", file=sys.stderr)

            started = time.time()
            fn = TOOL_FUNCTIONS.get(name)
            if fn is None:
                result = f"Error: no tool named {name!r}"
            else:
                try:
                    result = fn(**args)
                except Exception as exc:
                    # Handing the error back to the model instead of crashing is
                    # what lets it self-correct. This one line is most of the
                    # difference between a demo and something that survives use.
                    result = f"Error calling {name}: {type(exc).__name__}: {exc}"

            # Did that call teach us anything? Errors and empty results are
            # how a stuck agent looks from the outside. Consecutive, so one
            # bad query among good ones doesn't count against the tool.
            if _is_barren(str(result)):
                barren[name] = barren.get(name, 0) + 1
                LAST_RUN["barren"].add(name)
            else:
                barren[name] = 0
                LAST_RUN["productive"].add(name)
                # Any schedule tool counts, not just query_transit. When
                # find_direct_trips was added it became the primary source of
                # times and this check silently stopped seeing them — the
                # hazard of naming a specific tool instead of a capability.
                if name in SCHEDULE_TOOLS and GTFS_TIME_IN_RESULT.search(str(result)):
                    LAST_RUN["times_retrieved"] += 1

            trace_event(
                "tool_call",
                step=step,
                tool=name,
                args=args,
                seconds=round(time.time() - started, 2),
                barren=_is_barren(str(result)),
                # Full result, not the clipped version: the whole point of a
                # trace is seeing what the model was actually looking at.
                result=str(result),
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": clip(str(result)),
                }
            )

    # Out of steps. Never return "I gave up" to a user when we're holding a
    # pile of successfully gathered data — make one final call with no tools
    # available, so the only thing the model can produce is an answer.
    LAST_RUN["truncated"] = True
    if verbose:
        print(f"  [!] step budget spent — forcing a final answer", file=sys.stderr)

    messages.append(
        {
            "role": "user",
            "content": (
                "You are out of tool calls. Answer the original question now "
                "using only what you have already gathered.\n\n"
                "CRITICAL: you must not invent any value you did not retrieve. "
                "If you never queried a departure time, do not state one — say "
                "explicitly that times were not retrieved. An answer that "
                "admits a gap is useful; an answer with plausible invented "
                "numbers is worse than no answer at all."
            ),
        }
    )
    final = call_model(messages, verbose=verbose, use_tools=False)
    return final.choices[0].message.content or (
        f"Stopped after {MAX_STEPS} steps without reaching an answer."
    )


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What should I do in Toronto tomorrow?"
    answer = ""
    try:
        answer = run(question)
        print(answer)
    except DailyQuotaExhausted as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(1)
    except NotFoundError as exc:
        # Models get retired for new users without warning.
        print(
            f"\nModel {MODEL!r} is not available to your key.\n"
            f"Run `python list_models.py` to see what is, then update MODEL "
            f"in .env.\n\nOriginal error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    finally:
        path = write_trace(question, answer)
        print(f"\n[{usage_line()}]", file=sys.stderr)
        print(f"[trace: {path}]", file=sys.stderr)
