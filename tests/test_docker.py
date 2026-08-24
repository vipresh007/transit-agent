"""Static checks on the container build. No docker required.

    python tests/test_docker.py

A docker build takes five or six minutes, most of it downloading a zip and
grinding 4.2 million rows into SQLite. Discovering a typo'd COPY path at
minute five is a bad way to spend an afternoon, and discovering a leaked key
after pushing the image is worse than bad — image layers are append-only, so
a secret copied in once is in there permanently no matter what a later RUN
deletes.

Everything here is text analysis of the Dockerfile and .dockerignore. It
cannot tell you the image works. It can tell you it won't waste your time or
publish your keys, which is most of the value for none of the runtime.
"""

import json
import re
import sys
from pathlib import Path

from _harness import check, section

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"


def _ignored() -> list:
    return [line.strip() for line in
            DOCKERIGNORE.read_text("utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def _copies(text: str) -> list:
    """(sources, destination) for each COPY that reads the build context.

    `COPY --from=stage` is excluded on purpose: those paths exist inside an
    earlier build stage, not on this filesystem, so checking them against the
    local tree would fail on a correct Dockerfile.
    """
    out = []
    for line in re.findall(r"^COPY\s+(.+)$", text, re.M):
        if line.startswith("--from"):
            continue
        parts = [p for p in line.split() if not p.startswith("--")]
        out.append((parts[:-1], parts[-1]))
    return out


def test_files_exist():
    section("the build files are there")
    check("Dockerfile exists", DOCKERFILE.exists())
    check(".dockerignore exists", DOCKERIGNORE.exists())


def test_nothing_secret_can_enter_a_layer():
    section("secrets stay out")

    text = DOCKERFILE.read_text("utf-8")
    ignored = _ignored()

    for pattern in (".env", "data/", "traces/", ".cache/", ".git/"):
        check(f".dockerignore excludes {pattern}", pattern in ignored)

    for sources, _ in _copies(text):
        for src in sources:
            check(f"COPY {src} is not a secret",
                  ".env" not in src and not src.endswith((".key", ".pem")))

    # A key given as --build-arg is readable forever via `docker history`.
    # This is the mistake that cannot be undone once an image is pushed.
    args = re.findall(r"^ARG\s+(\w+)", text, re.M)
    for name in args:
        check(f"build ARG {name} is not a credential",
              not re.search(r"KEY|TOKEN|SECRET|PASSWORD", name, re.I))

    # The database must be built, never copied from the developer's machine:
    # 589MB, and it would freeze one laptop's schedule into every container.
    for sources, _ in _copies(text):
        for src in sources:
            check(f"COPY {src} is not a database",
                  not src.endswith((".db", ".sqlite", ".sqlite3")))


def test_every_copy_source_exists():
    section("COPY paths resolve")

    for sources, _ in _copies(DOCKERFILE.read_text("utf-8")):
        for src in sources:
            check(f"{src} exists in the repo",
                  (ROOT / src.rstrip("/")).exists())


def test_the_port_is_reachable():
    section("the container can actually be talked to")

    text = DOCKERFILE.read_text("utf-8")
    # 127.0.0.1 inside a network namespace is reachable only from inside it.
    # `docker run -p 8000:8000` would then map a port that never answers, and
    # the app looks hung rather than misconfigured.
    check("HOST is set to 0.0.0.0", "HOST=0.0.0.0" in text)
    check("PORT is set", re.search(r"PORT=\d+", text) is not None)

    app = (ROOT / "transit" / "web" / "app.py").read_text("utf-8")
    check("the server reads HOST from the environment",
          'getenv("HOST"' in app)
    check("the server reads PORT from the environment",
          'getenv("PORT"' in app)
    check("but still defaults to loopback for local runs",
          '"HOST", "127.0.0.1"' in app)


def test_healthcheck_is_wellformed():
    section("healthcheck")

    text = DOCKERFILE.read_text("utf-8")
    match = re.search(r"HEALTHCHECK[^\n]*\\?\n?\s*CMD\s+(\[.*?\])\s*\n",
                      text, re.S)
    check("HEALTHCHECK uses exec form", match is not None)
    if match:
        try:
            argv = json.loads(match.group(1))
            check("it is valid JSON", isinstance(argv, list) and len(argv) >= 2)
        except json.JSONDecodeError as exc:
            # Shell form here would silently never run: dockerd doesn't
            # validate it, the check just fails forever and the container
            # sits marked unhealthy.
            check("it is valid JSON", False, f"JSONDecodeError: {exc}")
        # Probing a port only proves python started. /api/status opens the
        # database, so a healthy container has proved something worth knowing.
        check("it probes an endpoint, not just the port",
              "/api/" in match.group(1))


def test_the_database_is_built_not_shipped():
    section("multi-stage")

    text = DOCKERFILE.read_text("utf-8")
    check("there is a builder stage", re.search(r"FROM .+ AS \w+", text) is not None)
    check("the final stage copies from it", "--from=" in text)
    check("the loaders run during the build",
          "load_gtfs.py" in text and "load_shapes.py" in text)
    # Layers are append-only. If the zip and extracted CSVs were unpacked in
    # the final stage, deleting them later would not reclaim the space.
    final = text.split("FROM")[-1]
    check("the final stage does not download the feed",
          "load_gtfs.py" not in final)


def test_nothing_is_chowned_after_it_is_copied():
    section("no accidental second copy")

    # `chown -R` over a path holding files from an earlier layer copies every
    # one of them into the new layer. Overlay filesystems cannot change a
    # lower layer's metadata in place, so the file is duplicated — and layers
    # are append-only, so the original is unreclaimable.
    #
    # This cost 490MB: the 488MB database ended up in the image twice,
    # reported by `docker history` as a suspiciously large `useradd` line.
    # Nothing fails, nothing warns, and the build log looks perfect.
    #
    # TWO THINGS THIS CHECK GOT WRONG FIRST TIME, both worth keeping in view:
    #
    #   1. It read comments. The paragraph you are reading contains the
    #      offending string, so the detector flagged its own explanation.
    #      Second time that has happened in this test suite.
    #   2. It ignored build stages. Stage one COPYs before stage two chowns,
    #      so the correct, deliberate chown-before-any-copy in the final stage
    #      looked like the bug it exists to prevent.
    #
    # Both are the same failure the project keeps meeting from the other side:
    # a checker that demands one shape fires on a correct answer written in
    # another. Compare per stage, and read instructions rather than prose.
    text = DOCKERFILE.read_text("utf-8")
    code = [l.strip() for l in text.splitlines()
            if l.strip() and not l.strip().startswith("#")]

    stages: list = []
    for line in code:
        if line.upper().startswith("FROM"):
            stages.append([])
        if stages:
            stages[-1].append(line)

    for number, stage in enumerate(stages, 1):
        first_copy = next((i for i, l in enumerate(stage)
                           if l.startswith("COPY")), len(stage))
        late = [l for l in stage[first_copy:] if re.search(r"\bchown\s+-R\b", l)]
        check(f"stage {number}: no recursive chown after a COPY", not late,
              True if not late else f"found: {late}")

    # The safe form: ownership set as the file is written, one copy only.
    # Stage one is exempt — nothing survives it but the one file stage two
    # copies out, so ownership there is irrelevant.
    for line in stages[-1] if stages else []:
        if line.startswith("COPY"):
            check(f"{line.split('#')[0].strip()[:50]} sets --chown",
                  "--chown=" in line)


def test_it_does_not_run_as_root():
    section("privileges")

    text = DOCKERFILE.read_text("utf-8")
    check("a USER is set", re.search(r"^USER\s+\w+", text, re.M) is not None)
    users = re.findall(r"^USER\s+(\S+)", text, re.M)
    check("and it isn't root", users and users[-1] not in ("root", "0"))


if __name__ == "__main__":
    for fn in (test_files_exist,
               test_nothing_secret_can_enter_a_layer,
               test_every_copy_source_exists,
               test_the_port_is_reachable,
               test_healthcheck_is_wellformed,
               test_nothing_is_chowned_after_it_is_copied,
               test_the_database_is_built_not_shipped,
               test_it_does_not_run_as_root):
        fn()
    from _harness import PASSED
    print(f"\n{PASSED['n']} checks passed")
