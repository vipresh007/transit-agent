"""Loop mechanics: retries, failover, throttling, and the anti-nonsense guards.

Every test here reproduces a failure that actually happened while building
this. The comment above each one is the bug it prevents coming back.

    python tests/test_agent.py
"""

from unittest.mock import MagicMock, patch

from _harness import calls, check, clean_env, install_fake_openai, says, section

om = install_fake_openai()
clean_env()

import agent           # noqa: E402
import cache           # noqa: E402
import llm             # noqa: E402
import providers       # noqa: E402

DAILY = "429 quotaId: 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'"
MINUTE_WITH_DELAY = "429 Please retry in 3.5s."
BARE_429 = "429 {'message': 'Rate limit exceeded', 'code': '1300'}"
TOOL_JSON_BAD = "400 {'code': 'tool_use_failed', 'failed_generation': '...'}"

OK = MagicMock(choices=[MagicMock(message=MagicMock(content="done", tool_calls=None))])
OK.usage = None


def fresh(script, tools=None):
    """Run the loop against a scripted sequence of model responses."""
    providers._active = 0
    fake = MagicMock()
    fake.chat.completions.create.side_effect = script
    providers._client = fake
    llm.USAGE.update(n=0, prompt_tokens=0, completion_tokens=0)
    return fake, patch.dict(agent.TOOL_FUNCTIONS, tools or {})


def test_retry_and_quota():
    section("retries and rate limits")

    # A transient 503 used to discard every tool call made so far.
    fake, td = fresh([om.InternalServerError("503"), OK])
    with td, patch("time.sleep"):
        llm.call_model([{"role": "user", "content": "x"}], verbose=False)
    check("transient 503 retries and succeeds", fake.chat.completions.create.call_count, 2)

    # A stated delay beats our guess: Groq's "tokens per day" clears in minutes.
    check("parses Groq's compound delay",
          round(llm.server_retry_delay(Exception("try again in 5m4.128s")), 1), 304.1)
    check("parses Google's delay",
          llm.server_retry_delay(Exception("Please retry in 28s")), 28.0)
    check("no delay stated returns None",
          llm.server_retry_delay(Exception("429 quota exceeded")), None)

    # Retrying a genuine daily cap spends quota to learn the same thing.
    fake, td = fresh([om.RateLimitError(DAILY)])
    providers._active = len(providers.AVAILABLE) - 1   # no backup left to switch to
    with td, patch("time.sleep") as slept:
        try:
            llm.call_model([{"role": "user", "content": "x"}], verbose=False)
            raised = False
        except llm.DailyQuotaExhausted:
            raised = True
    check("daily quota raises instead of retrying", raised)
    check("daily quota never sleeps", slept.call_count, 0)

    # A bare 429 with no stated delay is usually per-minute: be patient.
    fake, td = fresh([om.RateLimitError(BARE_429)] * 5)
    providers._active = len(providers.AVAILABLE) - 1
    with td, patch("time.sleep") as slept:
        try:
            llm.call_model([{"role": "user", "content": "x"}], verbose=False)
        except om.RateLimitError:
            pass
    waited = sum(c.args[0] for c in slept.call_args_list)
    check("bare 429 waits over a minute in total", waited > 60)

    # The loop must never fall off the bottom returning None -- that surfaced
    # as AttributeError on `.choices`, naming entirely the wrong problem.
    fake, td = fresh([om.RateLimitError(MINUTE_WITH_DELAY)] * 6)
    providers._active = len(providers.AVAILABLE) - 1
    with td, patch("time.sleep"):
        try:
            result = llm.call_model([{"role": "user", "content": "x"}], verbose=False)
            check("exhausted retries raise, not return None", result, "should have raised")
        except om.RateLimitError:
            check("exhausted retries raise, not return None", True)

    # Malformed tool JSON is a sampling accident; resample. Other 400s aren't.
    providers._active = 0
    fake, td = fresh([om.BadRequestError(TOOL_JSON_BAD), OK])
    with td, patch("time.sleep"):
        llm.call_model([{"role": "user", "content": "x"}], verbose=False)
    check("tool_use_failed resamples", fake.chat.completions.create.call_count, 2)

    fake, td = fresh([om.BadRequestError("400 invalid schema")])
    with td, patch("time.sleep") as slept:
        try:
            llm.call_model([{"role": "user", "content": "x"}], verbose=False)
        except om.BadRequestError:
            pass
    check("a genuine 400 is not retried", slept.call_count, 0)


def test_failover():
    section("provider failover")

    providers._active = 0
    first = providers.describe()
    gem = MagicMock()
    gem.chat.completions.create.side_effect = om.RateLimitError(DAILY)
    providers._client = gem
    groq = MagicMock()
    groq.chat.completions.create.return_value = OK
    with patch.object(providers, "OpenAI", lambda **kw: groq), patch("time.sleep"):
        llm.call_model([{"role": "user", "content": "x"}], verbose=False)
    check("switched provider on daily quota", providers.describe() != first)

    # Gemini attaches thought_signature to tool calls; Groq rejects it.
    history = [{
        "role": "assistant",
        "tool_calls": [{
            "id": "1", "type": "function",
            "function": {"name": "geocode", "arguments": "{}"},
            "extra_content": {"google": {"thought_signature": "SIG"}},
        }],
    }]
    cleaned = providers.sanitize(history)
    check("vendor extras stripped when not on gemini",
          "extra_content" in cleaned[0]["tool_calls"][0], False)
    providers._active = 0


def test_loop_guards():
    section("loop guardrails")

    # Five find_pois calls at different radii taught nothing and ate the budget.
    ran = []
    script = [calls(("find_pois", {"lat": 1, "lon": 2, "category": "museum"}))] * 2 + [says("done")]
    fake, td = fresh(script, {"find_pois": lambda **k: (ran.append(k), '[{"name":"AGO"}]')[1]})
    with td:
        agent.run("q", verbose=False)
    check("exact duplicate tool call is blocked", len(ran), 1)
    check("duplicate recorded in flags", agent.LAST_RUN["repeats"], 1)

    # Barren results should throttle; productive ones should never be throttled.
    ran = []
    script = [calls(("find_pois", {"lat": 1, "lon": 2, "category": "museum", "radius_m": r}))
              for r in (500, 800, 1000, 1500, 2000)] + [says("done")]
    fake, td = fresh(script, {"find_pois": lambda **k: (ran.append(k), "No museum found within 500m.")[1]})
    with td:
        agent.run("q", verbose=False)
    check("barren tool cut off after 3 tries", len(ran), 3)

    ran = []
    script = [calls(("query_transit", {"sql": f"SELECT {i}"})) for i in range(6)] + [says("done")]
    fake, td = fresh(script, {"query_transit": lambda **k: (ran.append(k), '[{"col":1}]')[1]})
    with td:
        agent.run("q", verbose=False)
    check("productive tool is never throttled", len(ran), 6)

    # Running out of steps used to return "I gave up" while holding real data.
    script = [calls(("geocode", {"place": f"p{i}"})) for i in range(providers.MAX_STEPS)]
    script += [says("Partial answer from what I have.")]
    fake, td = fresh(script, {"geocode": lambda **k: '{"lat":43.6,"lon":-79.4}'})
    with td:
        out = agent.run("q", verbose=False)
    check("step budget forces a real answer", "gave up" not in out and "Stopped after" not in out)
    check("truncation is flagged", agent.LAST_RUN["truncated"])
    final_call = fake.chat.completions.create.call_args.kwargs
    check("final call removes tools entirely", final_call["tools"], None)


def test_verification_guards():
    section("verification: did it actually establish anything?")

    # A clean-looking run that verified nothing is the most dangerous state.
    script = [calls(("query_transit", {"sql": "SELECT DISTINCT service_id FROM calendar"})),
              says("The 506 departs at 8:04.")]
    fake, td = fresh(script, {"query_transit": lambda **k: '[{"service_id":"1"}]'})
    with td:
        agent.run("q", verbose=False)
    check("a service_id lookup is not a schedule time", agent.LAST_RUN["times_retrieved"], 0)

    # Any schedule tool counts -- keying this to one tool name went stale twice.
    for tool in sorted(agent.SCHEDULE_TOOLS):
        script = [calls((tool, {})), says("done")]
        fake, td = fresh(script, {tool: lambda **k: '[{"depart":"08:04:17"}]'})
        with td:
            agent.run("q", verbose=False)
        check(f"{tool} counts as a verified time", agent.LAST_RUN["times_retrieved"], 1)

    # require_times: answering early with invented times gets pushed back once.
    script = [calls(("geocode", {"place": "K"})),
              says("Leaves about 8am."),
              calls(("plan_journey", {"origin_lat": 43.6})),
              says("Departs 08:03:31, arrives 08:37:00.")]
    fake, td = fresh(script, {
        "geocode": lambda **k: '{"lat":43.6,"lon":-79.4}',
        "plan_journey": lambda **k: '[{"legs":[{"route":"510","depart":"08:03:31"}]}]',
    })
    with td:
        out = agent.run("q", verbose=False, require_times=True)
    check("pushback forces the work", "08:03:31" in out)

    # ...but a model that refuses must still terminate.
    script = [says("still guessing")] * 8
    fake, td = fresh(script, {})
    with td:
        out = agent.run("q", verbose=False, require_times=True)
    check("pushback fires only once", out, "still guessing")

    # No pushback when times aren't required.
    fake, td = fresh([says("It'll be sunny.")], {})
    with td:
        out = agent.run("weather?", verbose=False)
    check("non-journey questions are unaffected", out, "It'll be sunny.")


def test_result_clipping():
    section("context management")

    small = '[{"a": 1}]'
    check("small results pass through", agent.clip(small), small)
    big = agent.clip("x" * 9000)
    check("oversized results are clipped", len(big) < 3000)
    check("clipping announces itself", "TRUNCATED" in big)

    for text, barren in [
        ('[{"stop_id":"809"}]', False),
        ("No museum found within 1500m.", True),
        ("SQL error: no such column", True),
        ("Query returned no rows.", True),
        ("", True),
    ]:
        check(f"barren({text[:28]!r})", agent._is_barren(text), barren)


if __name__ == "__main__":
    for fn in (test_retry_and_quota, test_failover, test_loop_guards,
               test_verification_guards, test_result_clipping):
        fn()
    from _harness import PASSED
    print(f"\n{PASSED['n']} checks passed")
