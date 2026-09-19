"""Runtime adapter.

The rest of the codebase never talks to an inference server directly — it asks a
Runtime for a completion given a *role*. Tonight the only implementation is
llama.cpp-via-Ollama on loopback. The vLLM implementation for the 80 GB tier
implements the same three methods (chat / chat_stream / structured) and nothing
above this file changes. See §12 of the design dossier.
"""
from __future__ import annotations
import json
import re
import subprocess
import time
from typing import Iterator, Optional

import httpx

from .config import OLLAMA_HOST, ALLOWED_HOSTS
from urllib.parse import urlparse



REASONING_TAG = re.compile(r"<think>.*?</think>\s*", re.S | re.I)
REASONING_OPENERS = (
    "okay, let", "okay so", "okay,", "hmm,", "hmm ", "wait,", "wait ",
    "let me think", "let me start", "let's tackle", "let's see",
    "first, i need to", "first, let", "alright, let", "i need to figure",
    "so the user", "the user wants", "the user is asking",
)


def strip_reasoning(text: str) -> str:
    """Defence in depth against reasoning leaking into content.

    Ollama's `think: false` is silently ignored by some hybrid-reasoning tags
    (verified: qwen3:4b returns its scratchpad in `content` with `thinking`
    empty). We therefore serve a true instruct model AND scrub here, so a model
    swap can never put a monologue into a Word document.
    """
    if not text:
        return text
    text = REASONING_TAG.sub("", text)
    # a monologue often ends with a real answer after the last blank line
    low = text.lstrip().lower()
    if any(low.startswith(o) for o in REASONING_OPENERS):
        # keep everything after the last blank line that ends the monologue
        parts = text.split("\n\n")
        for i in range(len(parts) - 1, 0, -1):
            tail = "\n\n".join(parts[i:]).strip()
            if tail and not any(tail.lower().startswith(o)
                                for o in REASONING_OPENERS):
                return tail
    return text.strip()


class EgressBlocked(RuntimeError):
    pass


def _guard(url: str):
    """The single door. Every outbound request in the app passes through here."""
    host = urlparse(url).hostname or ""
    if host not in ALLOWED_HOSTS:
        raise EgressBlocked(f"refused non-local host: {host!r}")
    return url


# ---------------------------------------------------------------- VRAM telemetry
_VRAM_CACHE: dict = {"t": 0.0, "v": None}
_VRAM_TTL = 1.5   # nvidia-smi costs ~500 ms with a live CUDA context; the solver
                  # runs on every request and must not pay that.


def vram(force: bool = False) -> dict:
    """Live VRAM in MiB. Returns zeros if nvidia-smi is unavailable (CPU profile)."""
    now = time.time()
    if not force and _VRAM_CACHE["v"] and now - _VRAM_CACHE["t"] < _VRAM_TTL:
        return _VRAM_CACHE["v"]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
        )
        total, used, free = [int(x.strip()) for x in out.stdout.strip().split(",")]
        v = {"total_mib": total, "used_mib": used, "free_mib": free, "gpu": True}
    except Exception:
        v = {"total_mib": 0, "used_mib": 0, "free_mib": 0, "gpu": False}
    _VRAM_CACHE.update(t=now, v=v)
    return v


# ---------------------------------------------------------------- Ollama runtime
class OllamaRuntime:
    name = "llamacpp/ollama"

    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host
        self._client = httpx.Client(timeout=httpx.Timeout(600.0, connect=5.0))
        self.resident: Optional[str] = None
        self.last_load_s: float = 0.0

    # -- introspection ------------------------------------------------------
    def up(self) -> bool:
        try:
            self._client.get(_guard(f"{self.host}/api/tags"), timeout=3)
            return True
        except Exception:
            return False

    def installed(self) -> list[str]:
        try:
            r = self._client.get(_guard(f"{self.host}/api/tags"), timeout=5)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def loaded(self) -> list[dict]:
        try:
            r = self._client.get(_guard(f"{self.host}/api/ps"), timeout=5)
            return [
                {"model": m["name"], "vram_mib": int(m.get("size_vram", 0) / 1048576)}
                for m in r.json().get("models", [])
            ]
        except Exception:
            return []

    # -- lifecycle ----------------------------------------------------------
    def ensure(self, model: str, ctx: int = 4096) -> float:
        """Make `model` resident. Returns seconds spent loading (0.0 if warm)."""
        if any(m["model"].startswith(model.split(":")[0]) and m["model"] == model
               for m in self.loaded()):
            self.resident = model
            self.last_load_s = 0.0
            return 0.0
        t0 = time.time()
        # Single-resident policy on a 4 GiB box: evict others before loading.
        for m in self.loaded():
            if m["model"] != model:
                self.unload(m["model"])
        self._client.post(
            _guard(f"{self.host}/api/generate"),
            json={"model": model, "prompt": "", "keep_alive": "10m",
                  "options": {"num_ctx": ctx}},
            timeout=300,
        )
        self.resident = model
        self.last_load_s = time.time() - t0
        return self.last_load_s

    def unload(self, model: str):
        try:
            self._client.post(
                _guard(f"{self.host}/api/generate"),
                json={"model": model, "prompt": "", "keep_alive": 0}, timeout=60)
        except Exception:
            pass

    # -- generation ---------------------------------------------------------
    def chat(self, model: str, messages: list[dict], *, ctx: int = 4096,
             temperature: float = 0.2, fmt: Optional[dict | str] = None,
             images: Optional[list[str]] = None, think: bool = False,
             num_predict: int = 1400) -> dict:
        msgs = [dict(m) for m in messages]
        if images:
            msgs[-1]["images"] = images
        body = {
            "model": model, "messages": msgs, "stream": False,
            "keep_alive": "10m",
            "options": {"num_ctx": ctx, "temperature": temperature,
                        "num_predict": num_predict},
        }
        if fmt is not None:
            body["format"] = fmt              # constrained decoding (JSON schema)
        # Qwen3 is a hybrid-reasoning family and thinks BY DEFAULT. Left implicit,
        # a 90 ms router probe becomes a 90 s one. Always state it.
        body["think"] = bool(think)
        t0 = time.time()
        r = self._client.post(_guard(f"{self.host}/api/chat"), json=body)
        r.raise_for_status()
        d = r.json()
        dt = time.time() - t0
        out_tok = d.get("eval_count", 0) or 0
        return {
            "text": strip_reasoning(d.get("message", {}).get("content", "")),
            "thinking": d.get("message", {}).get("thinking", ""),
            "model": model,
            "latency_s": round(dt, 2),
            "tokens_out": out_tok,
            "tps": round(out_tok / dt, 1) if dt > 0 and out_tok else 0.0,
        }

    def chat_stream(self, model: str, messages: list[dict], *, ctx: int = 4096,
                    temperature: float = 0.3, images: Optional[list[str]] = None,
                    num_predict: int = 1600, think: bool = False) -> Iterator[str]:
        msgs = [dict(m) for m in messages]
        if images:
            msgs[-1]["images"] = images
        body = {"model": model, "messages": msgs, "stream": True,
                "keep_alive": "10m", "think": bool(think),
                "options": {"num_ctx": ctx, "temperature": temperature,
                            "num_predict": num_predict}}
        with self._client.stream("POST", _guard(f"{self.host}/api/chat"),
                                 json=body) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                piece = d.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if d.get("done"):
                    break

    def structured(self, model: str, messages: list[dict], schema: dict,
                   *, ctx: int = 4096, temperature: float = 0.0,
                   images: Optional[list[str]] = None, think: bool = False,
                   num_predict: int = 1200) -> dict:
        """Grammar-constrained JSON. A malformed tool call is unrepresentable."""
        res = self.chat(model, messages, ctx=ctx, temperature=temperature,
                        fmt=schema, images=images, think=think,
                        num_predict=num_predict)
        txt = (res["text"] or "").strip()
        try:
            res["json"] = json.loads(txt)
        except Exception:
            s, e = txt.find("{"), txt.rfind("}")
            res["json"] = json.loads(txt[s:e + 1]) if s >= 0 < e else {}
        return res


RUNTIME = OllamaRuntime()
