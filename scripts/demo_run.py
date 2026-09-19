"""Drive one agent run from the CLI and print the trace. Useful for testing the
whole pipeline without the browser, and as a fallback demo surface.

    python scripts/demo_run.py "goal text" [file ...]
"""
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
API = "http://127.0.0.1:8080"

C = {"route": "\033[38;5;209m", "plan": "\033[38;5;80m", "swap": "\033[38;5;80m",
     "step": "\033[1m", "ok": "\033[38;5;114m", "bad": "\033[38;5;203m",
     "dim": "\033[38;5;244m", "r": "\033[0m"}


def main():
    goal = sys.argv[1] if len(sys.argv) > 1 else (
        "Read the attached inspection report, extract the thickness findings, "
        "and draft a note for approval for the deviation")
    files = [Path(a) for a in sys.argv[2:]]

    data = {"goal": goal}
    fs = [("files", (p.name, p.read_bytes())) for p in files if p.exists()]
    print(f"\n{C['dim']}goal   :{C['r']} {goal}")
    print(f"{C['dim']}files  :{C['r']} {[p.name for p in files] or 'none'}\n")

    t0 = time.time()
    artifacts = []
    with httpx.Client(timeout=httpx.Timeout(1800.0)) as cl:
        with cl.stream("POST", f"{API}/api/run", data=data, files=fs) as r:
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                t = f"{time.time() - t0:6.1f}s"
                k = ev["type"]
                if k == "route":
                    d = ev["decision"]
                    print(f"{t} {C['route']}ROUTE{C['r']}   {d['task_class']} "
                          f"conf={d['confidence']:.3f} stage={d['stage']} "
                          f"-> {d['model_id']} ({d['latency_ms']} ms)")
                    for s in d["signals"]:
                        print(f"        {C['dim']}· {s}{C['r']}")
                elif k == "plan":
                    print(f"{t} {C['plan']}PLAN{C['r']}    "
                          + " -> ".join(s["tool"] for s in ev["steps"]))
                elif k == "swap":
                    print(f"{t} {C['swap']}SWAP{C['r']}    {ev['model']} in "
                          f"{ev['seconds']}s  VRAM {ev['vram']['used_mib']}/"
                          f"{ev['vram']['total_mib']} MiB")
                elif k == "step_start":
                    print(f"{t} {C['step']}STEP{C['r']}    {ev['tool']}")
                elif k == "tool_log":
                    print(f"        {C['dim']}{ev['msg']}{C['r']}")
                elif k == "code":
                    print(f"        {C['dim']}code by {ev['model']} attempt "
                          f"{ev['attempt']} @ {ev['tps']} tok/s{C['r']}")
                elif k == "sandbox":
                    res = ev["result"]
                    col = C["ok"] if res["ok"] else C["bad"]
                    print(f"        {col}sandbox {'PASS' if res['ok'] else 'FAIL'}"
                          f"{C['r']} {res['duration_s']}s · {res['network']}")
                    for ln in (res["stdout"] or "").splitlines()[:12]:
                        print(f"        {C['dim']}| {ln}{C['r']}")
                    if not res["ok"]:
                        for ln in (res["stderr"] or "").splitlines()[-4:]:
                            print(f"        {C['bad']}| {ln}{C['r']}")
                elif k == "provenance":
                    g = ev["gate"]
                    col = C["ok"] if g["passed"] else C["bad"]
                    print(f"        {col}provenance {len(g['supported'])}/{g['total']}"
                          f" figures traced{C['r']}"
                          + ("" if g["passed"] else f"  unsupported: {g['unsupported']}"))
                elif k == "artifact":
                    a = ev["artifact"]
                    artifacts.append(a)
                    print(f"        {C['ok']}▣ {a['filename']} "
                          f"({a['size']/1024:.0f} KB){C['r']}")
                elif k == "step_result":
                    col = C["ok"] if ev.get("verify_ok") else C["bad"]
                    mark = "✓" if ev.get("verify_ok") else "✗"
                    print(f"        {col}{mark} verify · {ev.get('verify_note')}{C['r']}")
                elif k == "token":
                    print(ev["text"], end="", flush=True)
                elif k in ("step_error", "fatal"):
                    print(f"\n{t} {C['bad']}ERROR{C['r']}   {ev.get('error')}")
                elif k == "abort":
                    print(f"\n{t} {C['bad']}ABORT{C['r']}   {ev.get('reason')}")
                elif k == "done":
                    print(f"\n{t} {C['ok']}DONE{C['r']}    run {ev['run_id'][:8]} · "
                          f"{len(ev.get('citations', []))} citations · "
                          f"VRAM {ev['vram']['used_mib']} MiB")

    if artifacts:
        print(f"\n  deliverables in data/artifacts/:")
        for a in artifacts:
            print(f"    {a['filename']}  sha256 {a['sha256'][:16]}…")
    print()


if __name__ == "__main__":
    main()
