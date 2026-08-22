"""Toronto transit & trip-planning agent.

    core/      the loop, model access, tracing, per-thread state
    tools/     what the agent can actually do
    verify/    is the answer possible, and does it trace to a source
    pipeline/  the runnable programs: plan, crew, graph, memory, evals

Nothing in core/ imports from pipeline/. Dependencies point one direction:
pipeline -> verify -> tools -> core.
"""
