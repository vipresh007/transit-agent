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

import json
import os
import random
import re
import sys
import time

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

load_dotenv()

# We use the OpenAI SDK pointed at Google's OpenAI-compatible endpoint.
# Why: the tool-calling format is the de-facto standard, so this same code
# runs against Groq, OpenRouter, Ollama, or OpenAI by changing two lines.
client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

MODEL = os.getenv("MODEL", "gemini-3.6-flash")
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))

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

SYSTEM_PROMPT = """You are a travel planning assistant for Toronto.

Tools available: geocoding, weather, points of interest, and a SQL database
of the real TTC schedule.

Use them rather than guessing. You do not know today's weather, whether a
museum is open, or when the last streetcar runs, and inventing those details
makes you useless.

For anything about transit — routes, departure times, which days a service
runs — call describe_transit_schema first, then query_transit. If a query
returns an error, read it and write a corrected query rather than giving up.

BUDGET AND STOPPING — this matters as much as correctness:
You have a limited number of tool calls. Plan to answer within about five.
Before each query, ask: "will this change my answer?" If it only adds
confidence or detail, skip it and answer now.

Answer as soon as you can answer *usefully*. A good answer with one caveat
beats a perfect answer you never reach. Never keep querying to remove the
last bit of uncertainty — state the uncertainty in your answer instead.

If a detail is ambiguous, pick the most reasonable reading, say which one
you picked, and move on. Do not run queries to resolve ambiguity.

Answer concisely and concretely, with real place names and real times. If
the data doesn't support a confident answer, say so plainly instead of
filling the gap."""


# Errors worth retrying: the server was busy, rate-limited us, or the network
# hiccuped. A 400 (bad request) is NOT here — retrying a malformed request just
# fails again more slowly.
RETRIABLE = (InternalServerError, RateLimitError, APIConnectionError, APITimeoutError)

# Counts requests for the whole process, so you can see your quota burn rate.
REQUEST_COUNT = {"n": 0}


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


def _server_retry_delay(exc: Exception) -> float | None:
    """Google often tells you exactly how long to wait. Believe it."""
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc))
    if match:
        return float(match.group(1))
    match = re.search(r"'retryDelay': '(\d+(?:\.\d+)?)s'", str(exc))
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
    delay = 2.0
    for attempt in range(attempts):
        try:
            REQUEST_COUNT["n"] += 1
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                # Omitting tools entirely is what forces a text answer. Asking
                # nicely in the prompt is not reliable; removing the option is.
                tools=TOOL_SCHEMAS if use_tools else None,
                extra_body=EXTRA_BODY or None,
            )
        except RateLimitError as exc:
            if _is_daily_quota(exc):
                raise DailyQuotaExhausted(
                    f"Daily free-tier quota exhausted for {MODEL}.\n"
                    f"Used {REQUEST_COUNT['n']} requests this run.\n\n"
                    "Options:\n"
                    "  - Switch MODEL in .env to gemini-2.5-flash-lite (~1000/day)\n"
                    "  - Wait for the reset (midnight Pacific)\n"
                    "  - Create a second key in a different Google Cloud project"
                ) from exc
            _backoff(exc, attempt, attempts, delay, verbose)
            delay *= 2
        except RETRIABLE as exc:
            if attempt == attempts - 1:
                raise
            _backoff(exc, attempt, attempts, delay, verbose)
            delay *= 2


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


def run(user_message: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

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
            return message.content or "(empty response)"

        # The assistant turn must go into history before the tool results,
        # or the next request will 400 on a dangling tool_call_id.
        #
        # model_dump() rather than a hand-built dict: it round-trips provider
        # extras like Gemini's thought_signature, which a manual rebuild
        # silently drops. See the THINKING_BUDGET note above.
        messages.append(message.model_dump(exclude_none=True))

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if verbose:
                print(f"  [{step}] {name}({args})", file=sys.stderr)

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

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

    # Out of steps. Never return "I gave up" to a user when we're holding a
    # pile of successfully gathered data — make one final call with no tools
    # available, so the only thing the model can produce is an answer.
    if verbose:
        print(f"  [!] step budget spent — forcing a final answer", file=sys.stderr)

    messages.append(
        {
            "role": "user",
            "content": (
                "You are out of tool calls. Answer the original question now "
                "using only what you have already gathered. State clearly "
                "what you could not verify."
            ),
        }
    )
    final = call_model(messages, verbose=verbose, use_tools=False)
    return final.choices[0].message.content or (
        f"Stopped after {MAX_STEPS} steps without reaching an answer."
    )


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What should I do in Toronto tomorrow?"
    try:
        print(run(question))
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
        print(
            f"\n[{REQUEST_COUNT['n']} model requests used this run]",
            file=sys.stderr,
        )
