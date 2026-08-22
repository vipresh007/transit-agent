"""Toronto transit & trip-planning agent.

    core/      the loop, model access, tracing, per-thread state
    tools/     what the agent can actually do
    verify/    is the answer possible, and does it trace to a source
    pipeline/  the runnable programs: plan, crew, graph, evals

Dependencies point one direction: pipeline -> verify -> tools -> core.
"""

# .env is loaded HERE, once, before any submodule can read os.getenv().
#
# It used to live in agent.py, so whether your key was loaded depended on
# whether you happened to import agent before providers. plan.py did, so the
# CLI worked. ui.py imported llm first, which reaches providers without ever
# touching agent — so no key was set, providers called sys.exit() at import,
# and because SystemExit is a BaseException the UI's guard never saw it. The
# page hung on "loading the agent…" with a clean terminal.
#
# Configuration that every module depends on cannot be loaded by one of them.
# A package __init__ is the only place guaranteed to run first.
try:
    from dotenv import load_dotenv
except ImportError:                                        # pragma: no cover
    def load_dotenv(*_args, **_kwargs) -> bool:
        """dotenv is optional: real environment variables still work."""
        return False

load_dotenv()
