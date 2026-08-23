"""Static checks over the whole codebase. No imports, no API key, ~0.2s.

    python tests/test_imports.py

WHY THIS EXISTS. The package reorganisation rewrote imports across 46 files.
Every suite passed afterwards, and `plan_journey` was still broken: the moved
code called `paths.readonly_uri(...)` while `journey.py` imported no `paths`.

Nothing caught it because:
  - it's a NameError inside a function body, so merely importing the module
    is clean;
  - the only test touching journey.py exercises leg labelling, not the SQL
    path, which needs a 4.2M-row database and half a minute.

It surfaced as a live run: the agent got an error string back from its main
schedule tool, retried, got nothing, and invented a midnight departure.
Thirty-six requests and 135K tokens to discover a missing import.

The lesson is about test SHAPE. Behavioural tests only cover code they
actually execute, and the expensive paths are precisely the ones they skip.
A static check covers every line for free — not as a substitute for the
behavioural suites, but because it fails in the class of ways refactors do.

Both checks below are deliberately built to have NO false positives, at the
cost of missing things a real linter would find. A guard the team learns to
ignore is worse than no guard — the same reason grounding stopped flagging
markdown headings.
"""

import ast
import builtins
import sys
from pathlib import Path

from _harness import check, section

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PACKAGES = ("transit", "scripts", "tests")
ENTRY_POINTS = ("agent.py", "plan.py", "crew.py", "graph.py")


def sources() -> list[Path]:
    found = [p for pkg in PACKAGES for p in (ROOT / pkg).rglob("*.py")
             if "__pycache__" not in p.parts]
    return sorted(found + [ROOT / f for f in ENTRY_POINTS])


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


# ---------------------------------------------------------------------------

def module_file(dotted: str) -> Path | None:
    base = ROOT / Path(dotted.replace(".", "/"))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def exported(path: Path) -> set[str]:
    """Top-level names a module provides. Conservative: over-collects."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


def test_project_imports_resolve():
    section("every project import points at something real")

    broken, checked = [], 0
    for path in sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                continue
            if node.module.split(".")[0] not in PACKAGES:
                continue
            target = module_file(node.module)
            if target is None:
                broken.append(f"{rel(path)}:{node.lineno} no module {node.module}")
                continue
            provides = exported(target)
            for alias in node.names:
                checked += 1
                if alias.name == "*":
                    continue
                # `from transit.tools import memory` imports a SUBMODULE, which
                # won't appear among the parent's top-level assignments.
                if (alias.name not in provides
                        and module_file(f"{node.module}.{alias.name}") is None):
                    broken.append(
                        f"{rel(path)}:{node.lineno} "
                        f"{node.module} has no {alias.name!r}")

    for b in broken:
        print(f"    {b}")
    check(f"all {checked} imported symbols resolve", broken, [])


# ---------------------------------------------------------------------------

def bound_names(tree: ast.AST) -> set[str]:
    """Every name bound ANYWHERE in the module, ignoring scope.

    Ignoring scope is the point. A real linter tracks scopes and reports a
    name used in one function but defined in another; that's a true finding
    and also, in a codebase this size, a stream of noise about deferred
    imports and conditional definitions. Flattening every binding into one
    set means this check only fires when a name is bound NOWHERE — which is
    exactly the missing-import case, and cannot be a false positive.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names |= set(node.names)
    return names


def test_no_dangling_module_references():
    section("no module calls something it never imported")

    # The exact bug: journey.py said `paths.readonly_uri(DB_PATH)` with no
    # `from transit import paths` anywhere in the file.
    safe = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__",
                                 "__package__", "__builtins__", "self", "cls"}

    dangling = []
    for path in sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        known = bound_names(tree) | safe
        seen = set()
        for node in ast.walk(tree):
            # Only NAME.attr — a bare undefined name is usually a typo the
            # behavioural tests catch, while an undefined MODULE is silent
            # until the one code path that uses it finally runs.
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and isinstance(node.value.ctx, ast.Load)
                    and node.value.id not in known
                    and (path, node.value.id) not in seen):
                seen.add((path, node.value.id))
                dangling.append(
                    f"{rel(path)}:{node.lineno} uses "
                    f"`{node.value.id}.{node.attr}` but never binds "
                    f"`{node.value.id}`")

    for d in dangling:
        print(f"    {d}")
    check("nothing references an unimported module", dangling, [])


# ---------------------------------------------------------------------------

def test_entry_points_are_thin():
    section("the root launchers stay launchers")

    for name in ENTRY_POINTS:
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        # Anything beyond docstring + import + `if __name__` belongs in the
        # package, where it can be imported and tested.
        body = [n for n in tree.body
                if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        logic = [n for n in body if not isinstance(n, (ast.Import, ast.ImportFrom, ast.If))]
        check(f"{name} contains no logic", logic, [])

        imported = {a.name for n in ast.walk(tree)
                    if isinstance(n, ast.ImportFrom) for a in n.names}
        check(f"{name} imports main()", "main" in imported)


def test_nothing_imports_a_root_launcher():
    section("no module is shadowed by its own launcher")

    # `import graph` resolves to the root LAUNCHER, not transit/pipeline/graph.
    # It imports cleanly and exports only main(), so every attribute lookup
    # fails at runtime — test_graph.py died on `G.build`. The rewrite that
    # moved these imports only matched at column zero, and this one sat inside
    # a `try:` block.
    #
    # Any bare `import agent|plan|crew|graph` is now ambiguous by construction.
    # The launchers are the price of keeping the commands short; this is the
    # guard that makes the price safe.
    shadowed = {f[:-3] for f in ENTRY_POINTS}

    offenders = []
    for path in sources():
        if path.name in ENTRY_POINTS:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            for n in names:
                if n.split(".")[0] in shadowed:
                    offenders.append(
                        f"{rel(path)}:{node.lineno} `{n}` is a root launcher — "
                        f"use transit.<subpackage>.{n.split('.')[0]}")

    for o in offenders:
        print(f"    {o}")
    check("nothing imports a bare launcher name", offenders, [])


def test_no_logic_hides_in_a_main_guard():
    section("nothing important lives under `if __name__`")

    # plan.py kept its crash-trace handler in the `__main__` guard. Adding the
    # root launchers moved the entry point to
    # `from transit.pipeline.plan import main`, so the guard stopped running
    # and the crash trace vanished — silently, since nothing crashed in a test.
    # A `__main__` block is dead code the moment anything imports the module,
    # so the only safe thing to put there is a call.
    offenders = []
    for path in sources():
        if path.parent.name == "tests" or path.name in ENTRY_POINTS:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.If)
                    and isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == "__name__"):
                continue
            for stmt in node.body:
                # A bare call, `sys.exit(main())`, or an import: all fine.
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    continue
                if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass)):
                    continue
                offenders.append(
                    f"{rel(path)}:{stmt.lineno} {type(stmt).__name__} in "
                    f"`if __name__` — move it into a function")

    for o in offenders:
        print(f"    {o}")
    check("__main__ guards only call something", offenders, [])


def test_no_library_module_calls_sys_exit():
    section("libraries raise, entry points exit")

    # providers.py called sys.exit() at import when no API key was set. That
    # is a decision only a command-line program may make, and providers.py is
    # imported by everything. The Streamlit UI forgot load_dotenv(), so the
    # import exited — and SystemExit is a BaseException, so the UI's
    # `except Exception` guard sailed past it and Streamlit swallowed it.
    # Result: a page reading "loading the agent…" forever, clean terminal.
    #
    # scripts/ is exempt: those ARE command-line programs.
    offenders = []
    for path in sources():
        if path.parts[-2:][0] in ("scripts", "tests") or path.name in ENTRY_POINTS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "exit"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "sys"):
                # A main() is an entry point even when it lives in the package.
                enclosing = [n.name for n in ast.walk(tree)
                             if isinstance(n, ast.FunctionDef)
                             and node.lineno >= n.lineno
                             and node.lineno <= (n.end_lineno or n.lineno)]
                if "main" in enclosing or "_plan" in enclosing:
                    continue
                offenders.append(
                    f"{rel(path)}:{node.lineno} sys.exit() in a library module "
                    f"— raise instead, and let the caller decide")

    for o in offenders:
        print(f"    {o}")
    check("no library module ends the process", offenders, [])


def test_env_is_loaded_once_by_the_package():
    section(".env loads before anything reads it")

    # This was per-entry-point, and it lived in agent.py. plan.py imports
    # agent, so the CLI worked; ui.py imports llm, which reaches providers
    # WITHOUT touching agent — so no key was set and providers exited during
    # import. Whether your config loaded depended on import order.
    #
    # A package __init__ is the only place guaranteed to run before any
    # submodule. Configuration everything depends on cannot be loaded by one
    # of the things that depends on it.
    init = (ROOT / "transit" / "__init__.py").read_text(encoding="utf-8")
    check("the package loads .env itself", "load_dotenv()" in init)

    # And exactly once: scattered copies re-create the ordering question they
    # were meant to answer, and hide which one is actually load-bearing.
    stragglers = []
    for path in sources() + [ROOT / "ui.py"]:
        if path.name == "__init__.py" and path.parent.name == "transit":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("load_dotenv()"):
                stragglers.append(rel(path))
    for straggler in set(stragglers):
        print(f"    {straggler}")
    check("nothing else calls it", sorted(set(stragglers)), [])


def test_only_one_openai_stub():
    section("one test double, not several")

    # test_crew.py hand-rolled a second openai stub. When _harness gained
    # APIStatusError to mirror the SDK hierarchy, the copy didn't, and the
    # whole suite died on an import. Two doubles for one dependency is two
    # things to keep in step, and the unofficial copy is the one that rots.
    offenders = []
    for path in (ROOT / "tests").glob("*.py"):
        if path.name == "_harness.py":
            continue
        source = path.read_text(encoding="utf-8")
        # Assembled at runtime so this file doesn't match itself. A detector
        # written as a literal is part of its own haystack — the first run
        # reported test_imports.py as the offender.
        needle = "sys.modules[" + chr(34) + "openai"
        if needle in source or needle.replace(chr(34), chr(39)) in source:
            offenders.append(f"{rel(path)} builds its own openai stub — "
                             f"use _harness.install_fake_openai()")
    for o in offenders:
        print(f"    {o}")
    check("only _harness stubs the SDK", offenders, [])


def test_every_package_is_importable():
    section("no package is missing its __init__.py")

    for pkg in (ROOT / "transit", ROOT / "scripts"):
        for d in [pkg, *(p for p in pkg.rglob("*") if p.is_dir())]:
            if "__pycache__" in d.parts:
                continue
            if any(f.suffix == ".py" for f in d.iterdir()):
                check(f"{rel(d)}/__init__.py exists", (d / "__init__.py").exists())


if __name__ == "__main__":
    for fn in (test_project_imports_resolve, test_no_dangling_module_references,
               test_entry_points_are_thin, test_nothing_imports_a_root_launcher,
               test_no_logic_hides_in_a_main_guard,
               test_no_library_module_calls_sys_exit,
               test_only_one_openai_stub,
               test_env_is_loaded_once_by_the_package,
               test_every_package_is_importable):
        fn()
    from _harness import PASSED
    print(f"\n{PASSED['n']} checks passed")
