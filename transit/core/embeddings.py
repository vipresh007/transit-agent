"""Turning text into vectors, without paying for it.

An embedding maps text to a list of ~768 numbers such that similar meanings
land near each other. That's the entire trick behind semantic search: embed
the question, embed every chunk of the corpus, return the chunks whose vectors
point in roughly the same direction.

Provider order mirrors providers.py — first one available wins:

  ollama   nomic-embed-text, runs locally, unlimited and free. Embedding is
           far cheaper than generation (no autoregressive decoding), so even
           on CPU this is fast. The right default.
  mistral  mistral-embed, generous free tier, if Ollama isn't installed.
  gemini   text-embedding-004 as a last resort.

Note what is NOT here: sentence-transformers. It's the usual recommendation
and it drags in ~2.5GB of PyTorch to run an 80MB model. Ollama is already
installed and already serving an OpenAI-compatible endpoint.
"""

import os
import sys

import requests

TIMEOUT = 120

PROVIDERS = [
    {
        "name": "ollama",
        "url": os.getenv("OLLAMA_URL", "http://localhost:11434/v1") + "/embeddings",
        "model": os.getenv("EMBED_MODEL_OLLAMA", "nomic-embed-text"),
        "key_env": None,          # local, no auth
    },
    {
        "name": "mistral",
        "url": "https://api.mistral.ai/v1/embeddings",
        "model": os.getenv("EMBED_MODEL_MISTRAL", "mistral-embed"),
        "key_env": "MISTRAL_API_KEY",
    },
    {
        "name": "gemini",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/embeddings",
        "model": os.getenv("EMBED_MODEL_GEMINI", "text-embedding-004"),
        "key_env": "GEMINI_API_KEY",
    },
]

_chosen = None


def _reachable(prov: dict) -> bool:
    if prov["key_env"]:
        return bool(os.getenv(prov["key_env"]))
    # Local: only usable if the server is actually up.
    try:
        base = prov["url"].rsplit("/v1/", 1)[0]
        requests.get(base, timeout=2)
        return True
    except requests.RequestException:
        return False


def provider() -> dict:
    """Pick an embedding provider once, and keep it.

    Mixing providers within one index would be silently broken: vectors from
    different models aren't comparable, so search would return nonsense
    rather than an error. The model name is stored alongside the vectors and
    checked at query time.
    """
    global _chosen
    if _chosen is None:
        forced = os.getenv("EMBED_PROVIDER")
        candidates = [p for p in PROVIDERS if not forced or p["name"] == forced]
        for prov in candidates:
            if _reachable(prov):
                _chosen = prov
                break
        else:
            sys.exit(
                "No embedding provider available.\n"
                "  - Start Ollama and run: ollama pull nomic-embed-text\n"
                "  - or set MISTRAL_API_KEY / GEMINI_API_KEY in .env"
            )
    return _chosen


def model_name() -> str:
    p = provider()
    return f"{p['name']}:{p['model']}"


def embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a list of strings. Batched, because per-item HTTP is slow."""
    prov = provider()
    headers = {"Content-Type": "application/json"}
    if prov["key_env"]:
        headers["Authorization"] = f"Bearer {os.getenv(prov['key_env'])}"

    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        r = requests.post(
            prov["url"],
            json={"model": prov["model"], "input": batch},
            headers=headers,
            timeout=TIMEOUT,
        )
        if r.status_code >= 400:
            raise RuntimeError(
                f"{prov['name']} embeddings failed ({r.status_code}): "
                f"{r.text[:300]}"
            )
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        out.extend(d["embedding"] for d in data)

    return out


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
