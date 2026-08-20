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
| `agent.py` | The loop. ~60 lines of real logic, no framework. |
| `tools.py` | Tool functions plus the JSON schemas the model sees. |
| `.env` | Your key. Gitignored. |

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

**Stage 2 — GTFS in SQLite**
Download the TTC feed, load `stops`, `routes`, `trips`, `stop_times`,
`calendar` into SQLite. Give the agent a `query_transit` tool. Expect to lose
an hour to `calendar_dates.txt` — that's the exception table for holidays, and
it's why your Sunday itinerary will be wrong until you handle it.

**Stage 3 — structured output**
Stop returning prose. Define an `Itinerary` Pydantic model (days → legs →
mode/start/end/cost) and have the agent fill it. Now the output is checkable.

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

## Notes

- Nominatim and Overpass are volunteer-run. Cache aggressively, don't hammer them.
- GTFS times can exceed 24:00:00 (a 25:30:00 departure is 1:30am the next
  service day). This will bite you in stage 2.
- Keep `.env` out of git. The `.gitignore` handles it, but check `git status`
  before your first push anyway.
