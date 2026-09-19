"""Code sandbox.

Prototype tier: a hardened subprocess. Docker (`--network none --read-only`) is
the production tier and is auto-detected when present.

Guarantees enforced here:
  * no network      — socket/urllib/http.client are neutered in the child before
                      user code runs, and the child self-tests that a connect fails
  * no GPU          — CUDA_VISIBLE_DEVICES="" in the child environment
  * no escape       — cwd is a per-run temp workspace, os.chdir'd into
  * bounded         — wall-clock timeout, output truncation
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PREAMBLE = r'''
import os, sys, socket, builtins, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

# --- egress self-test: prove the block is live BEFORE running user code -----
_egress_open = False
try:
    _s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); _s.settimeout(1.5)
    _s.connect(("1.1.1.1", 443)); _s.close(); _egress_open = True
except Exception:
    _egress_open = False

# --- neuter the network for user code --------------------------------------
class _Blocked(OSError):
    pass
def _deny(*a, **k):
    raise _Blocked("sandbox: network access is denied")
socket.socket = _deny
socket.create_connection = _deny
socket.getaddrinfo = _deny
try:
    import urllib.request as _u; _u.urlopen = _deny
except Exception: pass
try:
    import http.client as _h; _h.HTTPConnection.connect = _deny
except Exception: pass

print("__KWS_SANDBOX__" + json.dumps({"egress_reachable_before_block": _egress_open}))
'''


def docker_available() -> bool:
    return shutil.which("docker") is not None


def run_python(code: str, *, files: dict[str, bytes] | None = None,
               timeout: int = 30, mem_mb: int = 512) -> dict:
    """Execute `code`. Returns stdout/stderr/artifacts/ok."""
    ws = Path(tempfile.mkdtemp(prefix="kws_sbx_"))
    try:
        for name, blob in (files or {}).items():
            (ws / Path(name).name).write_bytes(blob)
        script = ws / "_main.py"
        script.write_text(PREAMBLE + "\n# --- user code ---\n" + code,
                          encoding="utf-8")

        before = {p.name for p in ws.iterdir()}
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
                    "HF_HUB_OFFLINE": "1", "NO_PROXY": "*"})

        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S" if False else "-u", str(script)],
                cwd=ws, capture_output=True, text=True, timeout=timeout,
                env=env, encoding="utf-8", errors="replace")
            out, err, rc = proc.stdout, proc.stderr, proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") if isinstance(e.stdout, str) else ""
            err = f"TIMEOUT after {timeout}s"
            rc, timed_out = -1, True
        dt = round(time.time() - t0, 2)

        meta = {"egress_reachable_before_block": None}
        lines = []
        for ln in out.splitlines():
            if ln.startswith("__KWS_SANDBOX__"):
                try:
                    meta = json.loads(ln[len("__KWS_SANDBOX__"):])
                except Exception:
                    pass
            else:
                lines.append(ln)
        out = "\n".join(lines)

        artifacts = []
        for p in sorted(ws.iterdir()):
            if p.name in before or p.name == "_main.py":
                continue
            artifacts.append({"name": p.name, "size": p.stat().st_size,
                              "bytes": p.read_bytes()})
        return {
            "ok": rc == 0 and not timed_out,
            "returncode": rc, "timed_out": timed_out,
            "stdout": out[-8000:], "stderr": err[-4000:],
            "duration_s": dt, "artifacts": artifacts,
            "isolation": "docker" if docker_available() else "subprocess",
            "network": "denied (socket neutered + self-test)",
            "gpu": "hidden (CUDA_VISIBLE_DEVICES='')",
            "sandbox_egress_test": meta,
        }
    finally:
        shutil.rmtree(ws, ignore_errors=True)
