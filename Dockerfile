# Toronto Transit Agent — a container that builds its own database.
#
#     docker build -t transit-agent .
#     docker run --rm -p 8000:8000 --env-file .env transit-agent
#
# WHY TWO STAGES.
#
# The schedule database is 460MB built from a 34MB zip. Downloading and
# unpacking it leaves the zip, the extracted .txt files and a SQLite journal
# lying around, and in a single-stage build every one of those is permanently
# baked into a layer even if a later RUN deletes them. Layers are append-only:
# `rm` writes a deletion marker, it does not reclaim the bytes.
#
# So stage one builds the database in a container we throw away, and stage two
# copies out the single finished file. The intermediate junk never exists in
# the shipped image.
#
# WHY BUILD IT AT ALL RATHER THAN COPY THE LOCAL ONE.
#
# Three reasons, in increasing order of importance:
#   1. 460MB does not belong in a repo or a build context.
#   2. Copying it would freeze whatever schedule happened to be on one laptop
#      into every container, silently, until someone noticed stale times.
#   3. It proves the loaders run on a clean machine — which, until now, they
#      never had to. Every "works on my machine" bug in this project has been
#      a path or an assumption that only held locally.
#
# NO SECRETS HERE. The API keys arrive at RUN time via --env-file or -e, never
# at build time. A key passed as a build ARG is recoverable from the image
# with `docker history`, which is a mistake you cannot take back once pushed.

# --------------------------------------------------------------------------
# Stage 1: build the database
# --------------------------------------------------------------------------
FROM python:3.11-slim AS data

WORKDIR /build

# requests is all the loaders need. Installing the full requirements here
# would drag fastapi and pydantic into a stage that only downloads a zip.
RUN pip install --no-cache-dir requests

# Only what the loaders import. Copying the whole repo would invalidate this
# very expensive layer every time a CSS file changed.
COPY transit/__init__.py transit/paths.py transit/
COPY scripts/load_gtfs.py scripts/load_shapes.py scripts/optimize_db.py \
     scripts/__init__.py scripts/

# Fetch the zip, build tables and indexes, load 415k shape points, then
# ANALYZE and VACUUM. Roughly 4-7 minutes.
#
# The ANALYZE is not optional polish. It writes sqlite_stat1, which is how
# the query planner knows one index is selective and another isn't. Without
# it the planner guesses, and plan_journey's self-join over 4.2M rows is the
# query where a wrong guess costs seconds. The first version of this
# Dockerfile skipped it, so the container was quietly slower than the laptop
# with no way to see that from a build log.
#
# One RUN, not three: each RUN is a layer, and the intermediate database
# states would each be frozen into one.
RUN python scripts/load_gtfs.py \
 && python scripts/load_shapes.py \
 && python scripts/optimize_db.py

# --------------------------------------------------------------------------
# Stage 2: the actual image
# --------------------------------------------------------------------------
FROM python:3.11-slim

# PYTHONUNBUFFERED so print() reaches `docker logs` immediately instead of
# sitting in a buffer — the difference between watching a run and staring at
# nothing wondering if it hung.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    DATA_DIR=/app/data \
    TRACE_DIR=/app/traces \
    CACHE_DIR=/app/.cache

WORKDIR /app

# Only the four runtime dependencies plus the web server. langgraph and
# streamlit are in requirements.txt and deliberately excluded: nothing the
# server imports touches either, and langgraph alone is a large tree.
RUN pip install --no-cache-dir \
        "openai>=1.60.0" "python-dotenv>=1.0.1" "requests>=2.32.0" \
        "pydantic>=2.9.0" fastapi uvicorn

# THE USER IS CREATED BEFORE ANYTHING IS COPIED, and every COPY sets its own
# ownership. This is not style.
#
# The first version did the obvious thing — copy everything, then
# `RUN chown -R transit:transit /app` at the end — and it added 490MB to a
# 1.45GB image. `chown` modifies file metadata, and in an overlay filesystem
# modifying ANY attribute of a file from a lower layer copies the whole file
# up into the new one. The 488MB database was therefore in the image twice:
# once as copied, once as re-owned. Layers are append-only, so the first copy
# can never be reclaimed.
#
# `docker history` showed it as a 490MB `useradd` line, which is a strange
# enough thing to see that it's worth knowing why: the size belongs to the
# chown, not the user.
#
# Run as non-root because if this app can ever be talked into writing a file
# it shouldn't, doing that as uid 0 turns a small problem into a large one.
RUN useradd --create-home --uid 10001 transit \
 && mkdir -p /app/data /app/traces /app/.cache \
 && chown -R transit:transit /app

COPY --chown=transit:transit transit/ transit/
COPY --chown=transit:transit assets/ assets/
COPY --chown=transit:transit scripts/ scripts/
COPY --chown=transit:transit serve.py agent.py plan.py crew.py ./

# The one artefact worth carrying out of stage one. --chown here rather than
# afterwards is the whole point: written once, owned correctly on arrival.
COPY --from=data --chown=transit:transit \
     /build/data/transit.db /app/data/transit.db

# /app/traces and /app/.cache are writable but empty. memory.db and graph.db
# land there and do NOT survive the container — there is no volume. The app
# should say so rather than quietly forgetting a preference.
USER transit

EXPOSE 8000

# Answers only when the app can actually serve. /api/status touches the
# database, so a container reporting healthy has proved more than "python
# started" — which is all a TCP check would tell you.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8000') + '/api/status', timeout=4).status == 200 else 1)"]

CMD ["python", "serve.py"]
