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
import crew, agent, providers, llm

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
