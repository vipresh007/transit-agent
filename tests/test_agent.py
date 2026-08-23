"""Loop mechanics: retries, failover, throttling, and the anti-nonsense guards.

Every test here reproduces a failure that actually happened while building
this. The comment above each one is the bug it prevents coming back.

    python tests/test_agent.py
"""

from unittest.mock import MagicMock, patch

from _harness import calls, check, clean_env, install_fake_openai, says, section

om = install_fake_openai()
clean_env()

from transit.core import agent           # noqa: E402
from transit.core import cache           # noqa: E402
from transit.core import llm             # noqa: E402
from transit.core import providers       # noqa: E402

DAILY = "429 quotaId: 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'"
MINUTE_WITH_DELAY = "429 Please retry in 3.5s."
BARE_429 = "429 {'message': 'Rate limit exceeded', 'code': '1300'}"
TOOL_JSON_BAD = "400 {'code': 'tool_use_failed', 'failed_generation': '...'}"
# Real text from a live run: mid-plan, the model wrote a paragraph reasoning
# about which stop IDs to look up and never emitted the tool call. Groq's
# parser rejected it server-side, so the message never reached us.
OUTPUT_PARSE_BAD = (
    "400 {'code': 'output_parse_failed', 'message': 'Parsing failed. The model "
    "generated output that could not be parsed.', 'failed_generation': "
    "'We need to produce concrete journey with real clock times...'}")

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

    # Same class of accident, different code. Killed a real plan.py run.
    providers._active = 0
    fake, td = fresh([om.BadRequestError(OUTPUT_PARSE_BAD), OK])
    with td, patch("time.sleep"):
        llm.call_model([{"role": "user", "content": "x"}], verbose=False)
    check("output_parse_failed resamples too",
          fake.chat.completions.create.call_count, 2)

    # Failover is disabled for both of these: switching to another provider
    # would answer a different question, and the harness's fake client doesn't
    # survive providers.switch() rebuilding the client anyway.
    no_failover = patch("transit.core.providers.switch", return_value=False)

    # A stated delay that STOPS SHRINKING is a hard cap, not a busy minute.
    # Gemini's free tier is 20 requests/day for this model and it answered
    # "retry in 57s" four times running — four minutes asleep to learn
    # nothing, then a raw traceback.
    providers._active = 0
    stalled = om.RateLimitError(
        "429 Quota exceeded for metric: generate_content_free_tier_requests, "
        "quotaId: 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'. "
        "Please retry in 57.35s")
    fake, td = fresh([stalled, stalled, stalled, OK])
    raised = None
    with td, no_failover, patch("time.sleep") as slept:
        try:
            llm.call_model([{"role": "user", "content": "x"}], verbose=False)
        except Exception as exc:                            # noqa: BLE001
            raised = exc
    check("a repeated delay gives up instead of sleeping through it",
          slept.call_count, 1)
    check("and says it's a daily quota, not a mystery",
          isinstance(raised, llm.DailyQuotaExhausted))

    # A delay that IS shrinking is a rolling window — Groq's tokens-per-day
    # clears in minutes, and giving up on it threw away a whole run once.
    providers._active = 0
    fake, td = fresh([om.RateLimitError("429 per day. Please retry in 60s"),
                      om.RateLimitError("429 per day. Please retry in 20s"),
                      OK])
    with td, no_failover, patch("time.sleep") as slept:
        llm.call_model([{"role": "user", "content": "x"}], verbose=False)
    check("a shrinking delay is waited out", slept.call_count, 2)
    check("and the run then succeeds",
          fake.chat.completions.create.call_count, 3)

    # 413: the conversation is bigger than the provider's per-minute token
    # budget. Groq free tier is 8,000 TPM and ours reached 9,489 — a single
    # request larger than the cap can never fit, so waiting is pointless and
    # failover is the only move. "Not here" vs "not yet", one layer down from
    # daily quota vs busy minute.
    providers._active = 0
    too_big = om.APIStatusError("413 Request too large ... TPM Limit 8000, "
                                "Requested 9489")
    too_big.status_code = 413
    fake, td = fresh([too_big, OK])
    with td, patch("time.sleep") as slept:
        llm.call_model([{"role": "user", "content": "x"}], verbose=False)
    check("an oversized request fails over instead of sleeping",
          slept.call_count, 0)
    check("and the provider actually changed", providers._active, 1)

    # ORDER IS THE LOGIC. Every one of these subclasses APIStatusError, so a
    # base-class handler placed too early swallows all of them. This checks
    # the specific behaviours survive.
    providers._active = 0
    fake, td = fresh([om.InternalServerError("503"), OK])
    with td, patch("time.sleep"):
        llm.call_model([{"role": "user", "content": "x"}], verbose=False)
    check("a 503 still retries rather than hitting the 413 handler",
          fake.chat.completions.create.call_count, 2)

    providers._active = 0
    fake, td = fresh([om.BadRequestError(TOOL_JSON_BAD), OK])
    with td, patch("time.sleep"):
        llm.call_model([{"role": "user", "content": "x"}], verbose=False)
    check("tool_use_failed still resamples rather than hitting it too",
          fake.chat.completions.create.call_count, 2)

    fake, td = fresh([om.BadRequestError("400 invalid schema")])
    with td, patch("time.sleep") as slept:
        try:
            llm.call_model([{"role": "user", "content": "x"}], verbose=False)
        except om.BadRequestError:
            pass
    check("a genuine 400 is not retried", slept.call_count, 0)


def test_tool_scoping():
    section("only offering the tools a task needs")

    from transit.tools import TOOL_SETS, schemas_for

    full = schemas_for(None)
    journey = schemas_for(TOOL_SETS["journey"])
    check("the default is still everything", len(full), 13)
    check("journey is narrower", len(journey) < len(full))

    # The saving is the whole point, and it's per-call: every schema rides
    # along on all 16-20 requests of a run.
    import json
    saved = (len(json.dumps(full)) - len(json.dumps(journey))) // 4
    check("journey saves ~700+ tokens per call", saved > 700)

    names = {s["function"]["name"] for s in journey}
    check("it keeps the schedule tools", "plan_journey" in names)
    check("and preferences", "recall_preferences" in names)
    check("but drops the guides", "search_guides" not in names)
    check("and the weather", "get_weather" not in names)

    # A typo'd tool name that silently sends fewer tools than intended is the
    # exact kind of absence this project keeps learning to surface.
    try:
        schemas_for(["plan_journey", "teleport"])
        check("an unknown tool name raises", False)
    except ValueError as exc:
        check("an unknown tool name raises", "teleport" in str(exc))

    # THE CACHE MUST SEE THE DIFFERENCE. Keying a narrowed request the same as
    # a full one would return the wrong cached answer confidently, which is
    # strictly worse than a miss.
    from transit.core import cache
    msgs = [{"role": "user", "content": "x"}]
    wide = cache.key_for("m", msgs, True, 0, full)
    narrow = cache.key_for("m", msgs, True, 0, journey)
    check("a narrowed run gets its own cache key", wide != narrow)
    check("and None still means the full set",
          cache.key_for("m", msgs, True, 0, None), wide)

    # What actually reaches the provider.
    providers._active = 0
    fake, td = fresh([says("done")])
    with td, patch("time.sleep"):
        llm.call_model([{"role": "user", "content": "x"}], verbose=False,
                       schemas=journey)
    sent = fake.chat.completions.create.call_args.kwargs["tools"]
    check("the narrowed set is what gets sent", len(sent), len(journey))


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

    # Groq attaches a top-level `reasoning` string to assistant messages.
    # Replaying that to Mistral on failover returned 422 extra_forbidden.
    # We had a blacklist, met a second vendor extension, and broke a run —
    # so sanitize now whitelists the spec instead of naming known offenders.
    messy = [{
        "role": "assistant", "content": None,
        "reasoning": "groq's chain of thought",
        "audio": None, "function_call": None, "refusal": None,
        "tool_calls": [{"id": "1", "type": "function",
                        "function": {"name": "geocode", "arguments": "{}"},
                        "extra_content": {"google": {"thought_signature": "S"}}}],
    }, {"role": "tool", "tool_call_id": "1", "content": "r", "junk": "x"}]

    out = providers.sanitize(messy)
    check("unknown assistant fields are dropped",
          set(out[0]), {"role", "content", "tool_calls"})
    check("unknown tool-message fields are dropped",
          set(out[1]), {"role", "content", "tool_call_id"})
    check("tool_calls keep only spec keys",
          set(out[0]["tool_calls"][0]), {"id", "type", "function"})

    # ...but Gemini REQUIRES its signature echoed back, so that one survives.
    providers._active = next(i for i, p in enumerate(providers.AVAILABLE)
                             if p["name"] == "gemini")
    out = providers.sanitize(messy)
    check("gemini still gets its thought_signature",
          "extra_content" in out[0]["tool_calls"][0])
    check("but not groq's reasoning", "reasoning" in out[0], False)
    providers._active = 0


def test_sanitize_follows_the_provider():
    section("payload is sanitized for whoever actually receives it")

    # sanitize() ran ONCE before the retry loop, so a mid-loop failover
    # resent a Gemini-shaped payload to Mistral, which 422'd on
    # thought_signature. The sanitizer was correct; it ran before the thing
    # it depended on changed.
    history = [{
        "role": "assistant", "content": None, "reasoning": "groq cot",
        "tool_calls": [{"id": "1", "type": "function",
                        "function": {"name": "geocode", "arguments": "{}"},
                        "extra_content": {"google": {"thought_signature": "S"}}}],
    }]
    seen = []

    providers._active = 0
    gem = MagicMock()
    def gem_create(**kw):
        seen.append(("gemini", kw["messages"]))
        raise om.RateLimitError(DAILY)
    gem.chat.completions.create = gem_create
    providers._client = gem

    groq = MagicMock()
    def groq_create(**kw):
        seen.append(("groq", kw["messages"]))
        return OK
    groq.chat.completions.create = groq_create

    with patch.object(providers, "OpenAI", lambda **kw: groq), patch("time.sleep"):
        llm.call_model(history, verbose=False)

    check("two providers were tried", len(seen), 2)
    check("gemini received its thought_signature",
          "extra_content" in seen[0][1][0]["tool_calls"][0])
    check("the fallback did NOT receive it",
          "extra_content" in seen[1][1][0]["tool_calls"][0], False)
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




def test_grounding_pushback():
    section("grounding pushback")

    # Reproduces a real run: the agent listed nine Toronto venues found
    # nowhere in the retrieved passages, then closed with "all of the venues
    # above are listed in the Wikivoyage guides, so you can trust the
    # opening-hour details". Inventing venues is bad; asserting that the
    # invention came from a source is worse.
    retrieved = ('[{"article":"Toronto","section":"See > Museums",'
                 '"text":"The Art Gallery of Ontario holds Canadian works. '
                 'The Royal Ontario Museum covers natural history."}]')
    bad = ("Visit the Art Gallery of Ontario, the Royal Ontario Museum, "
           "Eaton Centre, Yorkdale Shopping Centre and Casa Loma. "
           "All of these are listed in the Wikivoyage guides.")
    good = ("Visit the Art Gallery of Ontario or the Royal Ontario Museum. "
            "The guides don't cover other indoor options.")

    script = [calls(("search_guides", {"query": "indoor"})), says(bad), says(good)]
    fake, td = fresh(script, {"search_guides": lambda **k: retrieved})
    with td:
        out = agent.run("rainy day?", verbose=False, require_grounding=True)
    check("ungrounded answer is rejected", out, good)
    check("coverage was recorded", agent.LAST_RUN["grounding"] is not None)

    # A well-grounded answer must pass straight through.
    script = [calls(("search_guides", {"query": "museums"})), says(good)]
    fake, td = fresh(script, {"search_guides": lambda **k: retrieved})
    with td:
        out = agent.run("museums?", verbose=False, require_grounding=True)
    check("grounded answer passes first time", out, good)
    check("only two model calls needed", fake.chat.completions.create.call_count, 2)

    # Pushback fires once; a model that ignores it must still terminate.
    script = [calls(("search_guides", {"query": "x"}))] + [says(bad)] * 6
    fake, td = fresh(script, {"search_guides": lambda **k: retrieved})
    with td:
        out = agent.run("rainy day?", verbose=False, require_grounding=True)
    check("pushback does not loop", out, bad)

    # Off by default: journey answers cite the schedule tool, not the guides.
    script = [says(bad)]
    fake, td = fresh(script, {})
    with td:
        out = agent.run("anything?", verbose=False)
    check("no pushback when not requested", out, bad)


def test_retrieval_telemetry():
    section("retrieval telemetry")

    weak = '{"quality":"weak","best_score":0.57,"results":[],"suggested_terms":["See","Museums"]}'
    strong = '{"quality":"strong","best_score":0.81,"results":[]}'

    script = [calls(("search_guides", {"query": "rainy day"})),
              calls(("search_guides", {"query": "indoor museums"})),
              says("Try the museums.")]
    seq = iter([weak, strong])
    fake, td = fresh(script, {"search_guides": lambda **k: next(seq)})
    with td:
        agent.run("rainy day?", verbose=False)
    check("counts searches", agent.LAST_RUN["searches"], 2)
    check("records that it re-queried", agent.LAST_RUN["rewrote_query"])
    check("tracks the best score seen", agent.LAST_RUN["best_retrieval"], 0.81)

    script = [calls(("search_guides", {"query": "kensington"})), says("ok")]
    fake, td = fresh(script, {"search_guides": lambda **k: strong})
    with td:
        agent.run("kensington?", verbose=False)
    check("a single strong search is not a rewrite",
          agent.LAST_RUN["rewrote_query"], False)


if __name__ == "__main__":
    for fn in (test_retry_and_quota,
               test_tool_scoping, test_failover,
               test_sanitize_follows_the_provider, test_loop_guards,
               test_verification_guards, test_result_clipping,
               test_grounding_pushback, test_retrieval_telemetry):
        fn()
    from _harness import PASSED
    print(f"\n{PASSED['n']} checks passed")
