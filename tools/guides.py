"""Retrieval over the Wikivoyage guides: the RAG half of the agent.

HYBRID SEARCH, not pure vectors. Two retrievers with opposite weaknesses:

  vector search   understands meaning. "somewhere to eat late" finds a
                  section about bars open past midnight that shares no words
                  with the query. But it's fuzzy on exact names — ask for
                  "Distillery District" and it may return generic
                  neighbourhood prose that's *about* the same topic.

  keyword (FTS5)  nails exact names and fails completely on paraphrase.

Travel questions contain both: proper nouns AND vague intent. Running both
and fusing the rankings is strictly better than choosing, and it costs one
extra SQLite query.

Fusion is Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across
retrievers. It combines RANKINGS rather than SCORES, which matters because
cosine similarity and BM25 aren't on comparable scales and normalising them
against each other is guesswork.
"""

import json
import math
import os
import sqlite3
import struct

DB_PATH = "guides.db"

# RRF constant. 60 is the value from the original paper and works fine; it
# damps the influence of top ranks so one retriever can't dominate.
RRF_K = 60

# Below this cosine, the corpus simply doesn't cover the question. Asked for
# "best ramen in Osaka" the index happily returned three Toronto restaurant
# sections at ~0.49 — semantically the nearest thing it had, and completely
# useless. Without a floor, a retriever ALWAYS returns its best guess, and an
# agent reading confident-looking passages has no way to know they're noise.
#
# 0.55 is tuned to this corpus and embedding model, not a universal constant:
# on-topic probes here score 0.59-0.83, off-topic ones 0.48-0.50. Re-measure
# if you change either.
MIN_RELEVANCE = float(os.getenv("GUIDES_MIN_RELEVANCE", "0.55"))


def _unpack(blob: bytes, dims: int) -> list[float]:
    return list(struct.unpack(f"{dims}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _meta(conn) -> dict:
    return dict(conn.execute("SELECT key, value FROM meta").fetchall())


def guides_status() -> str:
    """What's indexed, or how to build it."""
    if not os.path.exists(DB_PATH):
        return (
            f"{DB_PATH} not found. Run `python load_guides.py` to download and "
            f"index the Wikivoyage Toronto guides."
        )
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        m = _meta(conn)
        arts = conn.execute(
            "SELECT article, COUNT(*) FROM chunks GROUP BY article ORDER BY 2 DESC"
        ).fetchall()
    finally:
        conn.close()
    lines = [f"  {n:>4} chunks  {a}" for a, n in arts[:12]]
    return (
        f"{m.get('chunks')} chunks from {len(arts)} articles, embedded with "
        f"{m.get('embed_model')} ({m.get('dims')} dims), built {m.get('built')}.\n"
        + "\n".join(lines)
    )


def search_guides(query: str, limit: int = 4) -> str:
    """Search the travel guides for prose about places, food, and character."""
    if not os.path.exists(DB_PATH):
        return (
            f"{DB_PATH} not found — the travel guides are not indexed. "
            f"Run `python load_guides.py`. Schedule questions do not need this "
            f"tool; use plan_journey or query_transit instead."
        )

    import embeddings  # deferred: only needed when actually searching

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        meta = _meta(conn)
        # Vectors from different models are not comparable. Mixing them
        # wouldn't error, it would just return nonsense — so check.
        if meta.get("embed_model") != embeddings.model_name():
            return (
                f"Index was built with {meta.get('embed_model')} but the "
                f"current embedding provider is {embeddings.model_name()}. "
                f"Vectors from different models aren't comparable. Re-run "
                f"`python load_guides.py`, or set EMBED_PROVIDER to match."
            )

        dims = int(meta["dims"])
        qvec = embeddings.embed_one(query)

        # --- dense retrieval ------------------------------------------------
        # Brute force over a few hundred chunks: ~1ms, and no dependency on a
        # vector database. Approximate indexes (HNSW, IVF) matter at millions
        # of vectors, not hundreds. Reach for one when you measure a problem.
        rows = conn.execute(
            "SELECT id, article, heading, text, vector FROM chunks"
        ).fetchall()
        scored = [
            (cid, _cosine(qvec, _unpack(vec, dims)))
            for cid, _a, _h, _t, vec in rows
        ]
        scored.sort(key=lambda r: -r[1])
        dense_rank = {cid: i for i, (cid, _) in enumerate(scored[:20])}

        # --- sparse retrieval -------------------------------------------------
        sparse_rank = {}
        try:
            fts = conn.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY rank LIMIT 20",
                (" OR ".join(_fts_terms(query)),),
            ).fetchall()
            sparse_rank = {rid: i for i, (rid,) in enumerate(fts)}
        except sqlite3.OperationalError:
            pass  # odd punctuation can upset FTS5; dense results still stand

        # --- fuse -------------------------------------------------------------
        fused: dict[int, float] = {}
        for ranks in (dense_rank, sparse_rank):
            for cid, rank in ranks.items():
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)

        ranked = sorted(fused, key=lambda c: -fused[c])
        cosines = dict(scored)

        # Apply the relevance floor to the BEST hit, not to each one. If the
        # top result is on-topic, a weaker third result is still useful
        # context; if even the best is below the floor, nothing here is.
        best = max((cosines.get(c, 0.0) for c in ranked[:limit]), default=0.0)
        if best < MIN_RELEVANCE:
            return (
                f"Nothing relevant in the Toronto guides for {query!r} "
                f"(best match scored {best:.2f}, below the {MIN_RELEVANCE} "
                f"relevance floor). The guides cover Toronto neighbourhoods, "
                f"food, sights and practicalities — they will not have this. "
                f"Say so rather than reporting a weak match as an answer."
            )

        by_id = {r[0]: r for r in rows}
        results = []
        for cid in ranked[:limit]:
            _cid, article, heading, text, _vec = by_id[cid]
            results.append({
                "article": article,
                "section": heading,
                "text": text,
                "similarity": round(cosines.get(cid, 0.0), 3),
                # Report the RANK in each retriever, not a "both" label.
                # Every top-3 result said "both" — true, but uninformative,
                # because RRF structurally favours chunks found twice. Ranks
                # show which retriever actually surfaced a result first.
                "dense_rank": dense_rank.get(cid),
                "keyword_rank": sparse_rank.get(cid),
            })
        return json.dumps(results)
    finally:
        conn.close()


def compare_retrievers(query: str, limit: int = 3) -> dict:
    """Run dense-only, keyword-only and hybrid side by side.

    Not a tool the agent calls — a measurement harness for you. "Hybrid is
    better" is the received wisdom, and received wisdom is exactly the thing
    an eval suite exists to check. If the three columns agree on every query,
    the second retriever is costing complexity for nothing.
    """
    import embeddings

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        dims = int(_meta(conn)["dims"])
        qvec = embeddings.embed_one(query)
        rows = conn.execute(
            "SELECT id, article, heading, vector FROM chunks"
        ).fetchall()

        scored = sorted(
            ((cid, _cosine(qvec, _unpack(vec, dims))) for cid, _a, _h, vec in rows),
            key=lambda r: -r[1],
        )
        dense = [c for c, _ in scored[:20]]

        try:
            sparse = [r[0] for r in conn.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY rank LIMIT 20",
                (" OR ".join(_fts_terms(query)),),
            )]
        except sqlite3.OperationalError:
            sparse = []

        fused: dict[int, float] = {}
        for ranks in (dense, sparse):
            for i, cid in enumerate(ranks):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + i)
        hybrid = sorted(fused, key=lambda c: -fused[c])

        label = {cid: f"{a.replace('Toronto/', '')} > {h}"
                 for cid, a, h, _v in rows}
        return {
            "query": query,
            "dense": [label[c] for c in dense[:limit]],
            "keyword": [label[c] for c in sparse[:limit]],
            "hybrid": [label[c] for c in hybrid[:limit]],
            "top_score": round(scored[0][1], 3) if scored else 0.0,
        }
    finally:
        conn.close()


def _fts_terms(query: str) -> list[str]:
    """FTS5 chokes on punctuation, so pass it bare words only."""
    words = [w for w in "".join(
        c if c.isalnum() or c.isspace() else " " for c in query
    ).split() if len(w) > 2]
    return [f'"{w}"' for w in words] or ['""']
