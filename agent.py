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
import sys

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

load_dotenv()

# We use the OpenAI SDK pointed at Google's OpenAI-compatible endpoint.
# Why: the tool-calling format is the de-facto standard, so this same code
# runs against Groq, OpenRouter, Ollama, or OpenAI by changing two lines.
client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

MODEL = os.getenv("MODEL", "gemini-2.5-flash")
MAX_STEPS = 10  # a runaway loop is the classic first bug; cap it

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

You have tools for geocoding, weather, and finding points of interest.
Use them rather than guessing — you do not know today's weather or whether
a museum is open, and inventing those details makes you useless.

Work step by step: get coordinates first, then use them for other lookups.
When you have what you need, answer concisely and concretely, with real
place names and real times."""


def run(user_message: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            extra_body=EXTRA_BODY or None,
        )
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

    return f"Stopped after {MAX_STEPS} steps without a final answer."


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What should I do in Toronto tomorrow?"
    print(run(question))
