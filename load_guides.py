"""Stage 5: build a searchable index of Wikivoyage's Toronto guides.

    python load_guides.py

Downloads the Toronto article and its district subpages, splits them on
section boundaries, embeds each chunk, and stores everything in guides.db.
Takes a few minutes, mostly embedding.

CHUNKING ON SECTIONS, NOT CHARACTER COUNTS.
The usual advice is "split every 500 characters with 50 overlap", which is
what you do when your documents have no structure. Wikivoyage articles have
excellent structure -- "Get around", "See", "Eat", "Sleep" -- and those
headings are exactly the retrieval units a traveller wants. A fixed-size
splitter would cut the middle out of a restaurant listing and staple it to
an unrelated one. Respecting existing structure beats any overlap tuning.

Long sections still get split, but on paragraph boundaries, and every piece
keeps its article + heading as a prefix so a retrieved chunk carries the
context needed to interpret it.
"""

import json
import os
import re
import sqlite3
import struct
import sys
import time

import requests

import embeddings

API = "https://en.wikivoyage.org/w/api.php"
DB_PATH = "guides.db"
RAW_CACHE = "guides_raw.json"
ROOT_ARTICLE = os.getenv("GUIDE_ROOT", "Toronto")

# Wikimedia's user-agent policy asks for a descriptive agent with a contact
# address. Anonymous clients with a generic UA get throttled harder.
HEADERS = {
    "User-Agent": os.getenv(
        "WIKI_USER_AGENT",
        "transit-agent-learning-project/0.1 (personal learning project)",
    ),
    "Accept-Encoding": "gzip",
}

# Sections that are navigation furniture, not travel content.
SKIP_SECTIONS = {
    "see also", "external links", "references", "notes", "gallery",
    "further reading", "citations",
}

MAX_CHARS = 1800   # roughly 450 tokens; comfortably inside any context window
MIN_CHARS = 120    # below this a chunk is a stub heading, not content


def api(**params) -> dict:
    """One API call, with polite backoff on 429.

    Wikimedia runs this for free and rate-limits anonymous clients. A 429 here
    is a request to slow down, not an error to crash on — and they send
    Retry-After, so there's nothing to guess.
    """
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")

    delay = 2.0
    for attempt in range(5):
        r = requests.get(API, params=params, headers=HEADERS, timeout=60)
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", delay))
            print(f"\n  rate limited by Wikimedia — waiting {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Wikimedia kept rate limiting us; try again later.")


def article_titles() -> list[str]:
    """The root article plus every 'Toronto/District' subpage."""
    titles = [ROOT_ARTICLE]
    data = api(
        action="query", list="allpages",
        apprefix=f"{ROOT_ARTICLE}/", apnamespace=0, aplimit=100,
    )
    titles += [p["title"] for p in data.get("query", {}).get("allpages", [])]
    return titles


def fetch_plaintext(titles: list[str]) -> dict[str, str]:
    """Plain text for each article, with '== Heading ==' markers preserved.

    ONE ARTICLE PER REQUEST. The extracts API silently enforces exlimit=1 for
    full-text extracts (batching only works with exintro), so asking for five
    titles returned one extract and discarded the rest — the run reported
    "fetched 2/49" while hammering the API for nothing.

    exsectionformat=wiki is the other key parameter: without it the extract
    arrives as an undifferentiated wall of text and the structure we chunk on
    is gone.
    """
    out = {}
    for i, title in enumerate(titles, 1):
        data = api(
            action="query", prop="extracts", explaintext=1,
            exsectionformat="wiki", titles=title,
            # Follow redirects. Without this, 34 of 49 Toronto subpages
            # returned nothing: many district names ("Boytown") are redirects
            # to the canonical article, and a redirect has no extract of its
            # own. The API reports success and hands back an empty page.
            redirects=1,
        )
        for page in data.get("query", {}).get("pages", []):
            if page.get("extract", "").strip():
                out[page["title"]] = page["extract"]
        print(f"\r  {i}/{len(titles)} requested, {len(out)} with content",
              end="", flush=True)
        time.sleep(0.4)   # ~2.5 req/s: comfortably under Wikimedia's limits
    print()
    return out


HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split on == Heading == markers, keeping the full heading path.

    Uses finditer rather than re.split: splitting consumed the newline between
    a heading and an immediately-following subheading, so "== Get around ==\\n
    === By public transit ===" produced one section with the subheading left
    embedded as literal text. Adjacent delimiters are the classic case where
    split() quietly does the wrong thing.

    The heading PATH matters, not just the leaf. "Get around > By bike" tells
    you what a chunk is about; "By bike" alone could be any city.
    """
    marks = list(HEADING.finditer(text))
    sections = []

    intro = text[: marks[0].start()] if marks else text
    if intro.strip():
        sections.append(("Introduction", intro.strip()))

    trail: dict[int, str] = {}
    for i, m in enumerate(marks):
        level, heading = len(m.group(1)), m.group(2)
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end].strip()

        trail[level] = heading
        for deeper in [k for k in trail if k > level]:
            trail.pop(deeper)

        if body:
            sections.append((" > ".join(trail[k] for k in sorted(trail)), body))

    return sections


def split_long(body: str, limit: int = MAX_CHARS) -> list[str]:
    """Break an oversized section at the best available boundary.

    Tries progressively weaker separators. The first version only split on
    blank lines, and MediaWiki plaintext extracts use SINGLE newlines between
    listings — so it found no split points and emitted a 7,767-character chunk
    while claiming a 1,800 limit. A splitter that silently gives up produces
    exactly the giant chunks it was written to prevent.

    Sentence splitting is the last resort and still respects boundaries; we
    never cut mid-sentence, because half a sentence embeds to nothing useful.
    """
    if len(body) <= limit:
        return [body]

    for separator in ("\n\n", "\n", ". "):
        parts = body.split(separator)
        if len(parts) < 2:
            continue

        chunks, current = [], ""
        for part in parts:
            piece = part + separator
            if len(current) + len(piece) > limit and current:
                chunks.append(current.strip())
                current = ""
            current += piece
        if current.strip():
            chunks.append(current.strip())

        # If one part alone busts the limit, try a finer separator.
        if all(len(c) <= limit for c in chunks):
            return chunks

    # Nothing worked (one enormous unbroken run of text): hard-cut it rather
    # than emit something that won't fit a context window.
    return [body[i:i + limit] for i in range(0, len(body), limit)]


def build_chunks(articles: dict[str, str]) -> list[dict]:
    chunks = []
    for title, text in articles.items():
        for heading, body in split_sections(text):
            if heading.split(" > ")[-1].lower() in SKIP_SECTIONS:
                continue
            for piece in split_long(body):
                if len(piece) < MIN_CHARS:
                    continue
                # The heading path is prepended to the embedded text so the
                # vector encodes WHERE this came from, not just what it says.
                # "Open until 2am" is useless without knowing it's a bar in
                # Kensington Market.
                context = f"{title} > {heading}"
                chunks.append({
                    "article": title,
                    "heading": heading,
                    "text": piece,
                    "embed_text": f"{context}\n\n{piece}",
                })
    return chunks


def pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def main() -> None:
    print(f"Listing articles under '{ROOT_ARTICLE}'...")
    titles = article_titles()
    print(f"  {len(titles)} articles: {', '.join(titles[:6])}"
          f"{' ...' if len(titles) > 6 else ''}")

    # Cache the raw text. Wikimedia rate-limits hard enough that refetching
    # 49 articles takes minutes, and most iteration here is on CHUNKING, not
    # on the download. Separating "fetch" from "process" means you only pay
    # the network cost once. Delete guides_raw.json to force a refresh.
    if os.path.exists(RAW_CACHE) and "--refetch" not in sys.argv:
        articles = json.load(open(RAW_CACHE, encoding="utf-8"))
        print(f"\nUsing cached text for {len(articles)} articles "
              f"(--refetch to re-download)")
    else:
        print("\nFetching text...")
        articles = fetch_plaintext(titles)
        json.dump(articles, open(RAW_CACHE, "w", encoding="utf-8"))
    total_chars = sum(len(t) for t in articles.values())
    print(f"  {len(articles)} articles, {total_chars:,} characters")

    print("\nChunking on section boundaries...")
    chunks = build_chunks(articles)
    print(f"  {len(chunks)} chunks "
          f"(avg {total_chars // max(1, len(chunks)):,} chars)")
    if not chunks:
        sys.exit("No chunks produced — the API response shape may have changed.")

    print(f"\nEmbedding with {embeddings.model_name()}...")
    t0 = time.time()
    vectors = []
    for i in range(0, len(chunks), 32):
        batch = [c["embed_text"] for c in chunks[i:i + 32]]
        vectors.extend(embeddings.embed(batch))
        print(f"\r  {len(vectors)}/{len(chunks)}", end="", flush=True)
    print(f"\n  done in {time.time() - t0:.0f}s, {len(vectors[0])} dimensions")

    print(f"\nWriting {DB_PATH}...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            article TEXT, heading TEXT, text TEXT,
            vector BLOB, dims INTEGER
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        -- Full-text index alongside the vectors: hybrid search beats either
        -- alone, and exact names ("Distillery District") are precisely where
        -- embeddings are weakest.
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text, article, heading, content='chunks', content_rowid='id'
        );
    """)
    conn.executemany(
        "INSERT INTO chunks (article, heading, text, vector, dims) "
        "VALUES (?, ?, ?, ?, ?)",
        [(c["article"], c["heading"], c["text"], pack(v), len(v))
         for c, v in zip(chunks, vectors)],
    )
    conn.execute(
        "INSERT INTO chunks_fts(rowid, text, article, heading) "
        "SELECT id, text, article, heading FROM chunks"
    )
    conn.executemany(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        [("embed_model", embeddings.model_name()),
         ("dims", str(len(vectors[0]))),
         ("built", time.strftime("%Y-%m-%d %H:%M:%S")),
         ("chunks", str(len(chunks)))],
    )
    conn.commit()

    print(f"\nArticles indexed:")
    for article, n in conn.execute(
        "SELECT article, COUNT(*) FROM chunks GROUP BY article ORDER BY 2 DESC LIMIT 8"
    ):
        print(f"  {n:>4} chunks  {article}")
    conn.close()
    print(f"\nWrote {DB_PATH} ({os.path.getsize(DB_PATH) >> 20}MB)")


if __name__ == "__main__":
    main()
