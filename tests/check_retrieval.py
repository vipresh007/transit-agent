"""Probe the guide index and report what comes back.

    python tests/check_retrieval.py

Writes traces/retrieval_check.json as well as printing, so the results can be
read by someone who isn't at your terminal.

This is not a pass/fail test — retrieval quality is a judgement call, and
asserting on it would either be trivially loose or brittle. What it does is
make the judgement *possible*: for each probe you see which passages came
back, how they were matched (semantic, keyword, or both), and the cosine
score, so you can tell whether the index is doing its job.

The probes are chosen to stress different failure modes:

  exact names       embeddings are famously fuzzy on proper nouns; keyword
                    search should carry these
  paraphrase        no shared vocabulary with the source text; only the
                    vector half can find these
  wrong tool        schedule questions must NOT surface confident-sounding
                    prose, because the guides have no timetables
  out of corpus     something Toronto guides can't answer; low scores here
                    are the correct outcome, not a failure
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import tools                       # noqa: E402
from tools.guides import guides_status  # noqa: E402

PROBES = [
    ("exact name", "Distillery District"),
    ("exact name", "Toronto Islands ferry"),
    ("paraphrase", "somewhere to eat late at night"),
    ("paraphrase", "a good spot to watch the sunset over the water"),
    ("character", "what is Kensington Market like?"),
    ("character", "is downtown Toronto walkable?"),
    ("practical", "how do I pay for public transit"),
    ("wrong tool", "when is the last 501 streetcar on a weekday"),
    ("out of corpus", "best ramen in Osaka"),
]


def main() -> None:
    if not os.path.exists("guides.db"):
        sys.exit("guides.db not found — run python load_guides.py first")

    print(guides_status())
    print()

    report = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "probes": []}

    for kind, query in PROBES:
        t0 = time.time()
        raw = tools.search_guides(query, limit=3)
        elapsed = time.time() - t0

        try:
            hits = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[{kind}] {query!r}\n  -> {raw[:160]}\n")
            report["probes"].append({"kind": kind, "query": query, "error": raw[:300]})
            continue

        print(f"[{kind}] {query!r}   ({elapsed:.2f}s)")
        for h in hits:
            d = h.get("dense_rank")
            k = h.get("keyword_rank")
            src = f"d{d if d is not None else '-'}/k{k if k is not None else '-'}"
            print(f"   {h['similarity']:.3f} {src:<9} "
                  f"{h['article']} > {h['section']}")
            print(f"          {h['text'][:110].replace(chr(10), ' ')}...")
        print()

        report["probes"].append({
            "kind": kind,
            "query": query,
            "seconds": round(elapsed, 2),
            "hits": [
                {k: (v[:400] if k == "text" else v) for k, v in h.items()}
                for h in hits
            ],
        })

    # --- ablation: is hybrid actually better than either half alone? --------
    from tools.guides import compare_retrievers

    print("\n" + "=" * 70)
    print("ABLATION — dense vs keyword vs hybrid (top result for each)")
    print("=" * 70)
    ablation = []
    for kind, query in PROBES:
        c = compare_retrievers(query, limit=1)
        ablation.append(c)
        d = c["dense"][0] if c["dense"] else "-"
        k = c["keyword"][0] if c["keyword"] else "-"
        hy = c["hybrid"][0] if c["hybrid"] else "-"
        agree = "same" if d == k == hy else ("hybrid=dense" if hy == d
                                             else "hybrid=keyword" if hy == k
                                             else "hybrid differs")
        print(f"\n{query!r}  [{agree}]")
        print(f"   dense   {d}")
        print(f"   keyword {k}")
        print(f"   hybrid  {hy}")

    report["ablation"] = ablation
    differ = sum(1 for c in ablation
                 if c["dense"][:1] != c["keyword"][:1])
    print(f"\nThe two retrievers disagreed on {differ}/{len(ablation)} probes.")
    print("If that number is 0, the second retriever is pure overhead.")

    Path("traces").mkdir(exist_ok=True)
    out = Path("traces/retrieval_check.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
