"""
List the models each configured provider will actually serve you.

We have now been burned three times by guessing model names from
documentation: gemini-flash-latest (20/day quota), gemini-2.5-flash (retired
for new users), llama-3.3-70b-versatile (deprecated by Groq in June 2026).

Model catalogues change monthly and differ per account. Don't guess — ask.
Every OpenAI-compatible provider exposes GET /models, so one function covers
all of them.

    python list_models.py           # all configured providers
    python list_models.py groq      # just one
"""

import re
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

# Imported after load_dotenv so the provider list reflects your .env.
from agent import AVAILABLE  # noqa: E402

import os  # noqa: E402

TIMEOUT = 30


def list_provider(prov: dict) -> None:
    name = prov["name"]
    base = prov["base_url"].rstrip("/")
    key = os.getenv(prov["key_env"], "")

    print(f"\n{'=' * 60}\n{name}  ({base})\n{'=' * 60}")

    try:
        r = requests.get(
            f"{base}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
    except requests.RequestException as exc:
        print(f"  could not reach provider: {exc}")
        return

    models = r.json().get("data", [])
    if not models:
        print("  no models returned")
        return

    names = sorted(m["id"].removeprefix("models/") for m in models)

    # Filter out things that can't run an agent loop. Embedding, speech, and
    # image models all show up in these lists and none of them call tools.
    skip = ("embed", "embedding", "whisper", "tts", "imagen", "veo",
            "image", "aqa", "guard", "moderation")
    usable = [n for n in names if not any(s in n.lower() for s in skip)]

    for n in usable:
        note = ""
        low = n.lower()
        if "latest" in low or low.endswith("-preview"):
            note = "   <- alias/preview: unstable, avoid"
        # Match on token boundaries. Naive `"mini" in name` tagged every
        # single Gemini model as small, because "gemini" contains "mini".
        elif re.search(r"(?:^|[-_/])(lite|mini|small|\d+b)(?:$|[-_/])", low):
            note = "   <- small: biggest free quota, weakest at tool use"
        print(f"  {n}{note}")

    print(f"\n  {len(usable)} usable of {len(names)} total")
    if prov is AVAILABLE[0]:
        print("  (this is your primary provider)")


def main() -> None:
    wanted = sys.argv[1:] if len(sys.argv) > 1 else None
    providers = [p for p in AVAILABLE if not wanted or p["name"] in wanted]

    if not providers:
        print(f"No configured provider matches {wanted}.")
        print(f"Configured: {[p['name'] for p in AVAILABLE]}")
        sys.exit(1)

    for prov in providers:
        list_provider(prov)

    print(
        "\nSet MODEL (Gemini) or GROQ_MODEL (Groq) in .env to a stable,\n"
        "versioned name from the list above. Then confirm the real daily cap\n"
        'with one cheap call:  python agent.py "hello"'
    )


if __name__ == "__main__":
    main()
