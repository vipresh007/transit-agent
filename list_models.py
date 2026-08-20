"""
List the models your API key can actually use.

Model availability differs by account and changes often — models get retired
for new users without warning, and aliases like `-latest` point at whatever
is newest, which is usually the one with the tightest free quota.

Guessing model names from blog posts wastes requests. Ask the API.

    python list_models.py
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

KEY = os.environ["GEMINI_API_KEY"]

# The native endpoint returns more metadata than the OpenAI-compatible one:
# token limits and supported methods, which tell you if a model can do
# function calling at all.
URL = "https://generativelanguage.googleapis.com/v1beta/models"


def main() -> None:
    r = requests.get(URL, headers={"x-goog-api-key": KEY}, timeout=30)
    r.raise_for_status()
    models = r.json().get("models", [])

    usable = []
    for m in models:
        name = m["name"].removeprefix("models/")
        methods = m.get("supportedGenerationMethods", m.get("supportedActions", []))
        if "generateContent" not in methods:
            continue  # embedding / image / TTS models can't run an agent loop
        usable.append((name, m.get("inputTokenLimit", 0)))

    print(f"{len(usable)} models support generateContent:\n")
    for name, limit in sorted(usable):
        alias = "  <- alias, avoid: resolves to newest = smallest quota" if "latest" in name else ""
        print(f"  {name:<42} {limit:>10,} in{alias}")

    print(
        "\nPick a stable, versioned name (not an alias). Set it as MODEL in .env,\n"
        "then run one cheap request to discover the real daily cap:\n"
        '  python agent.py "hello"\n'
        "If it 429s immediately, that model's free quota is tiny — try the\n"
        "flash-lite variant, which usually has the most headroom."
    )


if __name__ == "__main__":
    main()
