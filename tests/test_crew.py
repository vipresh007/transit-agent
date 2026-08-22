"""Multi-agent: decomposition, concurrent research, synthesis.

    python tests/test_crew.py

The concurrency here is what forced threadstate.py into existence. Before it,
LAST_RUN and trace.EVENTS were module globals, so two subagents wrote into the
same dict and list — no crash, just a trace mixing two conversations and flags
belonging to neither. Global mutable state is fine until you need two of
something.

Everything here runs against a scripted fake model: no API key, no quota.
"""

import json, sys, types, os, time
from unittest.mock import MagicMock, patch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
om=types.ModuleType("openai")
for n in ["InternalServerError","RateLimitError","APIConnectionError","APITimeoutError","NotFoundError","BadRequestError"]:
    setattr(om,n,type(n,(Exception,),{}))
om.OpenAI=lambda **kw: MagicMock()
tm=types.ModuleType("openai.types"); cm=types.ModuleType("openai.types.chat"); cm.ChatCompletion=MagicMock()
sys.modules["openai"]=om; sys.modules["openai.types"]=tm; sys.modules["openai.types.chat"]=cm
dm=types.ModuleType("dotenv"); dm.load_dotenv=lambda *a,**k: None; sys.modules["dotenv"]=dm
os.environ.update(GEMINI_API_KEY="g", CACHE="0", TRACE_DIR="/tmp/crewtrace")
from transit.pipeline import crew
from transit.core import agent
from transit.core import providers
from transit.core import llm
def says(t):
    m=MagicMock(); m.content=t; m.tool_calls=None
    m.model_dump=lambda exclude_none=False: {"role":"assistant","content":t}
    return MagicMock(choices=[MagicMock(message=m)])

# --- decomposition ---
fake=MagicMock(); providers._client=fake
fake.chat.completions.create.return_value = says('["Museums in Toronto?", "Lunch downtown?"]')
print("decompose ->", crew.decompose("plan my day", verbose=False))

fake.chat.completions.create.return_value = says("I cannot split this.")
print("unparseable plan falls back to one task:", crew.decompose("simple q", verbose=False))

fake.chat.completions.create.return_value = says("[]")
print("empty plan falls back too:", crew.decompose("simple q", verbose=False))

# --- concurrent research keeps state separate ---
calls=[]
def per_thread(**kw):
    msgs = kw["messages"]
    task = next((m["content"] for m in msgs if m["role"]=="user"), "")
    calls.append(task[:20])
    time.sleep(0.05)
    return says(f"answer for {task[:24]}")
fake.chat.completions.create.side_effect = per_thread
fake.chat.completions.create.return_value = None

t0=time.time()
results = crew.research(["Task A about museums", "Task B about lunch",
                         "Task C about parks"], verbose=False)
elapsed=time.time()-t0
print(f"\n3 subtasks in {elapsed:.2f}s (serial would be ~0.15s)")
for r in results:
    print(f"  [{r['index']}] {r['task'][:24]!r} -> {r['answer'][:34]!r}")
assert [r["index"] for r in results]==[0,1,2], "results must be ordered"
assert all(r["task"][:6] in r["answer"] for r in results), "answers crossed threads"
print("  each answer matches its own task: no cross-contamination")

# --- one failure must not lose the others ---
def flaky(**kw):
    msgs=kw["messages"]; task=next((m["content"] for m in msgs if m["role"]=="user"),"")
    if "Task B" in task: raise om.InternalServerError("503")
    return says(f"answer for {task[:20]}")
fake.chat.completions.create.side_effect = flaky
with patch("time.sleep"):
    results = crew.research(["Task A", "Task B", "Task C"], verbose=False)
ok=[r for r in results if not r["error"]]; bad=[r for r in results if r["error"]]
print(f"\npartial failure: {len(ok)} succeeded, {len(bad)} failed")
assert len(ok)==2 and len(bad)==1

fake.chat.completions.create.side_effect=None
fake.chat.completions.create.return_value = says("merged answer")
out = crew.synthesize("q", results, verbose=False)
print("synthesis still produced an answer:", out[:40])
print("\nALL CREW CHECKS PASS")

# --- two-level grounding -----------------------------------------------------
# The gap this closes, from a real run: the synthesiser wrote "walk north under
# the Gardiner to reach the Distillery District" (it is east) and "509 or 510
# from Union Station" (the 510 does not serve Union). No section said either.
# Research grounding passed both at 87% because the union of three subtasks'
# tool output contains all those words somewhere; it spent its complaints on
# 'Saturdays' and 'African' instead.
print("\ntwo-level grounding")

results = [
    {"index": 0, "task": "morning", "answer": "Visit the Distillery District.",
     "error": None, "seconds": 0.0, "flags": {}, "events": [
         {"kind": "tool_call",
          "result": "Distillery District: pedestrian-only. Harbourfront "
                    "Centre is on the lake. The 509 runs from Union Station. "
                    "The 510 Spadina runs north from Queens Quay."}]},
    {"index": 1, "task": "lunch", "answer": "Eat at St. Lawrence Market.",
     "error": None, "seconds": 0.0, "flags": {}, "events": [
         {"kind": "tool_call", "result": "St. Lawrence Market, 93 Front St E."}]},
]

faithful = "Visit the Distillery District, then eat at St. Lawrence Market."
a = crew.audit(faithful, results)
assert not a["synthesis"]["unsupported"], a["synthesis"]["unsupported"]
print("  a merge that only reuses section text is clean")

invented = (faithful + " Walk north under the Gardiner to reach the "
                       "Distillery District, or take the 510 from Union Station.")
a = crew.audit(invented, results)
assert a["synthesis"]["unsupported"], "invented joining material went unflagged"
# The point of the second level: research grounding is the weaker check here,
# because its haystack is every tool result from every subtask.
assert a["synthesis"]["coverage"] < a["research"]["coverage"], (
    a["synthesis"]["coverage"], a["research"]["coverage"])
print(f"  invented connective tissue caught: synthesis "
      f"{a['synthesis']['coverage']:.0%} vs research "
      f"{a['research']['coverage']:.0%}")

# Choosing is allowed; altering is not. Dropping a candidate must not be
# reported as an unsupported claim.
a = crew.audit("Eat at St. Lawrence Market.", results)
assert not a["synthesis"]["unsupported"], a["synthesis"]["unsupported"]
print("  dropping a candidate is not flagged as invention")

print("\nALL GROUNDING-AUDIT CHECKS PASS")
