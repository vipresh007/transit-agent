"""A real web app: FastAPI + hand-written HTML. No framework chrome.

    pip install fastapi uvicorn
    python serve.py            ->  http://127.0.0.1:8000

WHY NOT STREAMLIT. Streamlit owns the page. Its header, hamburger and "Deploy"
button are part of the framework, and hiding them with CSS is fighting the
tool rather than using it. It's an excellent way to put a form in front of a
Python function and a poor way to build something that looks designed.

WHAT THIS IS. Four endpoints and three static files:

    GET  /                  the page
    GET  /api/traces        saved runs, for replay
    GET  /api/replay/{n}    a saved run as JSON — zero model requests
    POST /api/plan          start a run, returns an id
    GET  /api/stream/{id}   server-sent events: tool calls as they happen
    GET  /api/vehicles      live TTC vehicle positions, surface routes only

The agent still runs on a worker thread and publishes to a queue, exactly as
before. What changed is who drains the queue: an SSE generator instead of a
Streamlit loop. `trace.subscribe()` didn't need touching — that's the payoff
for having made the pipeline publish events rather than print them.

STILL ONE PIPELINE. plan.plan() returns a PlanResult; view.result_to_dict()
turns it into JSON; the browser draws it. Every judgement about whether an
answer is trustworthy is made in Python, so no front end can disagree with
another about it.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from transit import paths
from transit.core import llm, realtime, trace
from transit.pipeline import view
from transit.pipeline.plan import plan, replay, replayable
from transit.tools import memory
from transit.verify import constraints

ASSETS = paths.ROOT / "assets" / "web"

app = FastAPI(title="Toronto Transit Agent", docs_url=None, redoc_url=None)

# One queue per in-flight run. A dict rather than a single global because two
# browser tabs are two runs, and stage 9 already taught this project what
# happens when concurrent work shares one container.
RUNS: dict[str, queue.Queue] = {}


class Ask(BaseModel):
    question: str
    geocode: bool = False


def _worker(run_id: str, question: str, geocode: bool) -> None:
    """Run the pipeline, pushing every event onto this run's queue."""
    out = RUNS[run_id]

    def observer(event: dict) -> None:
        out.put(event)

    trace.subscribe(observer)
    try:
        prefs = constraints.Preferences.from_env()
        merged, remembered = memory.apply_to(prefs)
        out.put({"kind": "prefs", "describe": merged.describe(),
                 "remembered": remembered})

        result = plan(question, prefs=prefs, verbose=False)
        out.put({"kind": "done",
                 "result": view.result_to_dict(result, allow_network=geocode),
                 "usage": llm.usage_line()})
    except Exception as exc:                                # noqa: BLE001
        # Reaches the browser as an error card. A worker that dies quietly
        # leaves a spinner turning forever, which is the worst outcome.
        out.put({"kind": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        trace.unsubscribe(observer)
        out.put({"kind": "finished"})


@app.post("/api/plan")
def start(ask: Ask) -> dict:
    if not ask.question.strip():
        raise HTTPException(400, "empty question")
    run_id = uuid.uuid4().hex[:12]
    RUNS[run_id] = queue.Queue()
    threading.Thread(target=_worker, daemon=True,
                     args=(run_id, ask.question.strip(), ask.geocode)).start()
    return {"id": run_id}


@app.get("/api/stream/{run_id}")
async def stream(run_id: str) -> StreamingResponse:
    if run_id not in RUNS:
        raise HTTPException(404, "no such run")

    async def events():
        out = RUNS[run_id]
        loop = asyncio.get_running_loop()
        try:
            while True:
                # get() blocks, and blocking the event loop would freeze every
                # other request. Hand the wait to a thread.
                event = await loop.run_in_executor(None, out.get)
                yield f"data: {json.dumps(event, default=str)}\n\n"
                if event.get("kind") == "finished":
                    break
        finally:
            RUNS.pop(run_id, None)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/traces")
def traces() -> list[dict]:
    """Saved runs worth replaying, newest first."""
    out = []
    for path in replayable()[:40]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                                   # noqa: BLE001
            continue
        if not data.get("itinerary"):
            continue
        out.append({
            "name": path.stem,
            "question": data.get("question", ""),
            "when": data.get("when", ""),
            "model": data.get("model", ""),
        })
    return out


@app.get("/api/replay/{name}")
def replay_one(name: str, geocode: bool = False) -> dict:
    """A saved run, rendered from its trace. Zero model requests."""
    # Resolve through the known list rather than joining user input onto a
    # path. "../../etc/passwd" is a valid-looking name.
    match = next((p for p in replayable() if p.stem == name), None)
    if match is None:
        raise HTTPException(404, f"no trace named {name!r}")
    try:
        return view.result_to_dict(replay(match), allow_network=geocode)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/vehicles")
def vehicles(routes: str = "") -> dict:
    """Live vehicle positions for the given routes. Surface routes only.

    `available` is not decoration. None from the fetcher means "we couldn't
    reach the feed"; an empty dict means "nothing is running". A map that
    renders those two the same way tells you it's a quiet night during an
    outage, which is the failure this whole project keeps finding.

    Never fails the request. Real-time is a garnish on a page that is already
    correct without it.
    """
    wanted = [r for r in (routes or "").split("|") if r.strip()]
    if not wanted:
        return {"available": True, "routes": {}, "count": 0}

    grouped = realtime.for_routes(wanted)
    if grouped is None:
        return {"available": False, "routes": {}, "count": 0}
    return {"available": True, "routes": grouped,
            "count": sum(len(v) for v in grouped.values())}


@app.get("/api/status")
def status() -> dict:
    stored, notes = memory.load()
    return {
        "transit_db": paths.TRANSIT_DB.exists(),
        "guides_db": paths.GUIDES_DB.exists(),
        "preferences": stored,
        "notes": notes,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ASSETS / "index.html")


if ASSETS.exists():
    app.mount("/static", StaticFiles(directory=ASSETS), name="static")


def main() -> None:
    """Entry point. Config errors surface here, not as a stack trace."""
    import sys

    try:
        import uvicorn
    except ImportError:
        sys.exit("The web app needs FastAPI and uvicorn:\n"
                 "  pip install fastapi uvicorn")

    if not ASSETS.exists():
        sys.exit(f"Missing {ASSETS} — the front end files aren't there.")

    # Loopback by default: running a dev server on 0.0.0.0 exposes it to
    # everything on the coffee-shop wifi, and nothing here expects a stranger.
    #
    # In a container that default is wrong in the opposite direction —
    # 127.0.0.1 inside a network namespace is reachable only from inside the
    # container, so `docker run -p 8000:8000` maps a port that never answers
    # and the app looks hung rather than misconfigured. Hence the env var,
    # which the Dockerfile sets and a laptop doesn't.
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    print(f"  Toronto Transit Agent  ->  http://{shown}:{port}")
    print("  Saved runs replay for free; asking costs model requests.\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
