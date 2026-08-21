# Toronto transit & trip planning agent

A learning project. Build an agent that plans a day in Toronto using real
transit schedules, real opening hours, and real weather — on entirely free
data, with no paid services anywhere in the stack.

The point isn't the trip planner. The point is that each stage forces you to
learn one concept properly, and every stage produces something that runs.

## Setup

```bash
git clone <your-repo-url>
cd transit-agent

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then paste your Gemini key into .env
python agent.py "what should I do in Toronto tomorrow?"
```

You should see tool calls printed to stderr, then an answer. If the agent
answers without calling any tools, that's the first thing worth debugging —
usually the tool descriptions in `tools.py` aren't specific enough.

## Getting a Gemini API key (free, no card)

1. Go to https://aistudio.google.com/apikey and sign in with a Google account.
2. Click **Create API key**. Pick an existing Google Cloud project or let it
   make one. You do not need to enable billing.
3. Copy the key into `.env`.

Free tier as of mid-2026: roughly 250 requests/day on `gemini-2.5-flash`,
~1,000/day on `gemini-2.5-flash-lite`. One agent run burns several requests,
so switch to flash-lite when you're iterating hard.

Because we use Google's OpenAI-compatible endpoint, you can swap providers by
changing `base_url` and `MODEL` — Groq, OpenRouter, and Ollama all work with
this same code.

## Files

| File | What's in it |
|---|---|
| `agent.py` | The loop and its guardrails. Start here. |
| `providers.py` | Which model, failover order, pacing. Accessors, not globals. |
| `llm.py` | Retries, rate limits, quota, token accounting. |
| `cache.py` | Replaying identical requests during iteration. |
| `trace.py` | The JSON record written to `traces/` on every run. |
| `tools/geo.py` | Geocoding, weather, POIs (external APIs). |
| `tools/transit.py` | GTFS schema access and stop/trip lookups. |
| `tools/journey.py` | End-to-end journey planning with transfers. |
| `tools/registry.py` | Tool schemas + dispatch table. |
| `schemas.py` | The typed `Itinerary` and its validators. |
| `plan.py` | Two-phase research → structured itinerary. |
| `evals.py` | Test suite, plus `--selftest` for the checkers. |

Scripts: `load_gtfs.py` (build the DB), `optimize_db.py` (add indexes),
`list_models.py` (what your keys can actually use).

## Tests

```bash
python tests/run_all.py     # everything offline: no API key, no quota
```

Three suites, all free:

| Suite | Covers |
|---|---|
| `tests/test_tools.py` | tool logic, GTFS time arithmetic, SQL guardrails |
| `tests/test_agent.py` | retries, failover, throttling, the loop's guardrails |
| `evals.py --selftest` | whether the eval checkers themselves work |

`tests/_harness.py` stubs the OpenAI SDK before any project module loads, so
the loop runs against a scripted fake model. That's the only way to test
things like "the model emitted malformed tool-call JSON" or "the provider ran
out of daily quota mid-conversation" on demand.

Every check in `test_agent.py` corresponds to a bug that actually happened.
The comments say which.

Not included, because they cost something:

```bash
python tests/smoke_test.py   # hits Nominatim/Open-Meteo/Overpass
python evals.py              # spends model quota (~35 requests)
```

**Why this shape.** `agent.py` and `tools.py` both hit ~950 lines doing five
jobs each. The split isn't cosmetic: `providers.py` exposes `current()` and
`model()` as *functions* because failover reassigns the active provider at
runtime, and the old module-level `provider` variable produced a real bug —
`from agent import provider` bound the stale value and the footer reported
the wrong model for a whole run. Functions can't go stale.

## Free data sources

| Source | Use | Key needed |
|---|---|---|
| [Nominatim](https://nominatim.openstreetmap.org) | Geocoding | No (1 req/sec, set a User-Agent) |
| [Open-Meteo](https://open-meteo.com) | Weather forecast | No |
| [Overpass](https://overpass-api.de) | POIs, opening hours | No (few hundred queries/day) |
| [TTC GTFS](https://open.toronto.ca/dataset/ttc-routes-and-schedules/) | Toronto schedules | No — a zip of CSVs |
| [TTC GTFS-RT](https://open.toronto.ca/dataset/ttc-gtfs-realtime-gtfs-rt/) | Live alerts & delays | No |
| [Transitland](https://www.transit.land) | Feeds for other cities | Free key |
| [Wikivoyage dumps](https://dumps.wikimedia.org/enwikivoyage/) | Travel guide text (RAG corpus) | No |

## Roadmap

Each stage is roughly a sitting, and each one teaches exactly one thing.

**Stage 1 — the agent loop** ✅ *(this code)*
Tool calling, dispatch, feeding errors back to the model, a step cap.
Read `agent.py` top to bottom before moving on.

**Stage 2 — GTFS in SQLite** ✅
`python load_gtfs.py` pulls the current TTC feed and loads it into
`transit.db`. The agent gets two new tools: `describe_transit_schema` and
`query_transit` (read-only SELECT). It writes its own SQL, gets errors back as
text, and retries — self-correction with a real feedback signal.

**Stage 4 — evaluation** ✅ *(done early, out of order — and it should have been)*
`python evals.py`. Six cases with ground truth computed from the database at
run time, not hardcoded, so the suite survives the TTC's six-week republish.
Includes a hallucination check (asks for fare data the feed doesn't contain)
and a known-failing case (`calendar_dates` exceptions).

**Stage 3 — structured output** ✅
`python plan.py "how do I get from A to B?"`. Two phases: research with tools,
then a toolless structuring pass that emits JSON validated against the
`Itinerary` model in `schemas.py`. Validation errors are fed back with the
offending field named — self-correction against a perfect grader.

**Stage 4 — evaluation**
Write 15–20 questions with known-correct answers, plus constraint assertions:
no leg before 09:00, every transfer ≥10 minutes, no stop visited on a day it's
closed. This is the stage everyone skips. Don't. Without it, stages 5–8 are
guesswork.

**Stage 5 — RAG over Wikivoyage**
Download the Toronto article (and a few neighbourhood pages), chunk it, embed
it, store in Chroma. Add a `search_guides` tool. Watch your eval scores.

**Stage 6 — agentic RAG**
The interesting one. The agent must decide *whether* to retrieve: "what's
worth seeing in Kensington Market?" needs the guide; "when's the last
streetcar?" needs GTFS. Then let it rewrite its own query and retry when
retrieval comes back thin. Compare against stage 5 on the same evals.

**Stage 7 — self-correction & replanning**
Feed it constraint violations from stage 4 and make it fix its own itinerary.
Closed on Mondays, 4-minute transfer that needs 12, no service after 01:30.

**Stage 8 — memory**
Persist preferences across sessions. "No early mornings", "vegetarian",
"I'd rather walk than take a bus."

**Stage 9 — multi-agent**
One researcher per day of the trip running in parallel, a synthesizer merging
them into one itinerary. Now compare hand-rolled vs LangGraph and you'll have
earned an opinion.

## Gotchas hit so far

**`400 Function call is missing a thought_signature`**
Gemini's thinking models sign their reasoning and attach it to tool calls in
`tool_calls[N].extra_content.google.thought_signature` — a field the OpenAI
schema knows nothing about. Rebuilding the assistant message by hand drops it,
and the next turn is rejected. Fix: append `message.model_dump()` instead, so
provider extras round-trip. Escape hatch: `THINKING_BUDGET=0` in `.env`.

This is the general shape of the tradeoff with compatibility layers — portable
code, but provider-specific features leak. Worth remembering before you assume
"OpenAI-compatible" means drop-in.

## Notes

- Nominatim and Overpass are volunteer-run. Cache aggressively, don't hammer them.
- GTFS times can exceed 24:00:00 (a 25:30:00 departure is 1:30am the next
  service day). This will bite you in stage 2.
- Keep `.env` out of git. The `.gitignore` handles it, but check `git status`
  before your first push anyway.
