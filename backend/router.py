"""The four-stage capability router.

Stage 0  deterministic signals      ~0 ms    attachments, file types, explicit pin
Stage 1  embedding kNN              ~6 ms    bge-small centroids over exemplars
Stage 2  Qwen3-0.6B judge           ~90 ms   only when stage 1 margin is low
Stage 3  capability solver          ~1 ms    hard filter, score, hysteresis

Every decision is logged with its features, confidence, chosen model, runner-up
and margin. The UI renders this; see §4 of the design dossier.
"""
from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .config import ROOT, EMBED_MODEL
from .registry_loader import REGISTRY, ModelCard, TASK_CLASSES
from .runtime import RUNTIME, vram
from .db import db

EXEMPLARS = ROOT / "data" / "exemplars.jsonl"

# margin below which we escalate to the 0.6B judge
# Measured on the 31-query held-out set (scripts/eval_router.py --sweep):
#   judge=none      margin 0.000 -> 100.0%, stage 2 never fires
#   judge=qwen3-4b  margin 0.035 -> 100.0%, stage 2 fires on 9.7%
#   judge=qwen3-4b  margin 0.090 ->  80.6%, stage 2 fires on 71%
#   judge=qwen3-06b margin 0.090 ->  71.0%  <- a 0.6B judge is worse than the
#                                             embedding lane it is meant to help
# So: adjudicate rarely, and with the 4B (already resident, zero swap cost).
MARGIN_THRESHOLD = 0.035
# Which model adjudicates a low-margin call. Measured on the held-out set:
# the 0.6B is WORSE than the embedding lane it is meant to help, so the judge
# is the 4B (usually already resident, so the swap cost is zero).
JUDGE_MODEL_ID = "qwen3-4b"
# hysteresis: a challenger must beat the resident model by this many points
LAMBDA_HYSTERESIS = 6.0
LAMBDA_SWAP = 1.4          # points penalty per second of load cost
LAMBDA_LATENCY = 0.15      # points penalty per estimated second of generation

CODE_HINT = re.compile(
    r"```|\b(def |class |import |SELECT |traceback|stack trace|error \d|exception|"
    r"\.py\b|\.sql\b|\.ipynb\b|\.vba\b|npm |pip )", re.I)
DEBUG_HINT = re.compile(
    r"\b(fix|debug|error|crash|throws|traceback|exception|not working|wrong output|"
    r"fails|broken)\b", re.I)
CALC_HINT = re.compile(
    r"\b(calculate|compute|how much|what size|sizing|work out|determine)\b", re.I)
DRAW_HINT = re.compile(r"\b(p&id|p and id|pnid|isometric|drawing|layout|title block|gad)\b", re.I)
KB_HINT = re.compile(
    r"\b(our|company|as per (the )?(sop|standard|procedure|manual|policy)|"
    r"per our|internal (sop|standard))\b", re.I)


# ---------------------------------------------------------------- embedder
class Embedder:
    """bge-small-en-v1.5 via fastembed (ONNX, CPU). Falls back to a hashing
    vectoriser if the model files are not present, so the router never hard-fails."""

    def __init__(self):
        self.model = None
        self.mode = "hash"
        try:
            from fastembed import TextEmbedding
            from .config import DATA
            self.model = TextEmbedding(model_name=EMBED_MODEL,
                                       cache_dir=str(DATA / "models" / "embed"))
            self.mode = "bge-small-en-v1.5"
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[router] embedder fallback to hashing: {e}")

    def __call__(self, texts: list[str]) -> np.ndarray:
        if self.model is not None:
            v = np.array(list(self.model.embed(texts)), dtype=np.float32)
        else:
            v = np.zeros((len(texts), 512), dtype=np.float32)
            for i, t in enumerate(texts):
                for tok in re.findall(r"[a-z]{3,}", t.lower()):
                    v[i, hash(tok) % 512] += 1.0
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(n, 1e-9, None)


@dataclass
class Decision:
    task_class: str
    confidence: float
    margin: float
    stage: int
    role: str
    model_id: str
    serve_name: str
    runner_up: str | None
    score_gap: float
    resident_reused: bool
    needs_vision: bool
    needs_tools: bool
    think: bool
    ctx: int
    latency_ms: float
    signals: list[str]
    scores: dict
    class_scores: dict
    swap_s: float = 0.0
    vram_after: dict | None = None

    def dict(self):
        return asdict(self)


class Router:
    def __init__(self):
        self.embedder = Embedder()
        self.centroids: dict[str, np.ndarray] = {}
        self.n_exemplars = 0
        self.fit()

    # ------------------------------------------------------------ stage 1 fit
    def fit(self):
        rows = []
        if EXEMPLARS.exists():
            for line in EXEMPLARS.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            return
        texts = [r["t"] for r in rows]
        vecs = self.embedder(texts)
        by = {}
        for r, v in zip(rows, vecs):
            by.setdefault(r["c"], []).append(v)
        self.centroids = {}
        for c, vs in by.items():
            m = np.mean(np.stack(vs), axis=0)
            self.centroids[c] = m / max(float(np.linalg.norm(m)), 1e-9)
        self.n_exemplars = len(rows)

    # ------------------------------------------------------------ the cascade
    def route(self, text: str, *, has_image=False, has_pdf=False,
              filenames: list[str] | None = None, pin: str | None = None,
              run_id: str | None = None) -> Decision:
        t0 = time.perf_counter()
        filenames = filenames or []
        signals: list[str] = []
        stage = 0
        task_class = None
        conf = 1.0
        margin = 1.0
        class_scores: dict[str, float] = {}

        # ---- stage 0: deterministic signals -------------------------------
        if pin:
            signals.append(f"explicit pin @{pin}")
            task_class = "general_qa"
        elif has_image or has_pdf:
            drawing = bool(DRAW_HINT.search(text)) or any(
                DRAW_HINT.search(f) for f in filenames)
            task_class = "vision_drawing" if drawing else "vision_doc"
            signals.append("attachment: image/scanned pdf → vision lane")
            if drawing:
                signals.append("drawing keyword in prompt/filename")
        elif any(f.lower().endswith((".py", ".sql", ".ipynb", ".js", ".vba"))
                 for f in filenames):
            task_class = "code_debug" if DEBUG_HINT.search(text) else "code_write"
            signals.append("code file attached")
        if task_class:
            conf, margin = 1.0, 1.0

        # ---- stage 1: embedding kNN ---------------------------------------
        if task_class is None and self.centroids:
            stage = 1
            v = self.embedder([text])[0]
            sims = {c: float(v @ m) for c, m in self.centroids.items()}
            ordered = sorted(sims.items(), key=lambda kv: -kv[1])
            class_scores = {k: round(s, 4) for k, s in ordered}
            task_class, top = ordered[0]
            second = ordered[1][1] if len(ordered) > 1 else 0.0
            margin = top - second
            conf = top
            signals.append(
                f"kNN top={task_class} ({top:.3f}) runner-up={ordered[1][0]} "
                f"({second:.3f}) margin={margin:.3f}")

            # ---- stage 2: 0.6B judge on low margin -------------------------
            if margin < MARGIN_THRESHOLD:
                # The judge is a TIE-BREAKER, not a re-classifier: it may only
                # choose among the classes the embedding lane already shortlisted.
                # Unconstrained, a 0.6B model overrides correct kNN calls.
                shortlist = [c for c, _ in ordered[:3]]
                judged = self._judge(text, shortlist)
                if judged:
                    stage = 2
                    signals.append(
                        f"low margin → 0.6B judge chose '{judged}' "
                        f"from shortlist {shortlist}")
                    task_class = judged
                else:
                    signals.append("low margin → judge unavailable, kept kNN")

        if task_class is None:
            task_class = "general_qa"
            signals.append("no signal → default general_qa")

        # ---- requirements derived from the class --------------------------
        needs_vision = task_class.startswith("vision_")
        needs_tools = task_class in (
            "code_write", "code_debug", "calculation", "structured_extract",
            "doc_draft", "kb_qa")
        think = task_class == "calculation"
        if think:
            signals.append("calculation → thinking mode ON (free, same weights)")
        if task_class == "kb_qa" or KB_HINT.search(text):
            signals.append("internal-knowledge phrasing → KB grounding required")

        # ---- stage 3: capability solver -----------------------------------
        dec = self._solve(task_class, needs_vision, needs_tools, text)
        dec.update(stage=stage, task_class=task_class, confidence=round(conf, 4),
                   margin=round(margin, 4), signals=signals,
                   class_scores=class_scores, needs_vision=needs_vision,
                   needs_tools=needs_tools, think=think)
        dec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        d = Decision(**dec)
        self._log(text, d, run_id)
        return d

    # ------------------------------------------------------------ stage 2
    def _judge(self, text: str, shortlist: list[str] | None = None) -> str | None:
        # Adjudicate with a model that is ALREADY resident. Pulling the 4B in
        # just to break a tie costs a 7 s cold load and turns a 6 ms routing
        # decision into a 5.7 s one - worse than the mistake it prevents.
        loaded = {m["model"] for m in RUNTIME.loaded()}
        card = next((c for c in REGISTRY.available()
                     if c.serve_name in loaded and c.tool_calling != "none"), None)
        if card is None:
            # Nothing resident: adjudicating would cost a cold load. The kNN lane
            # measures 100% on the held-out set on its own, so a tie-break is
            # never worth a 7 s swap. Decline and keep the embedding verdict.
            return None
        if not card or not card.installed:
            return None
        allowed = shortlist or TASK_CLASSES
        schema = {
            "type": "object",
            "properties": {"task_class": {"type": "string", "enum": allowed}},
            "required": ["task_class"],
        }
        try:
            res = RUNTIME.structured(
                card.serve_name,
                [{"role": "system",
                  "content": "Classify the user's request into exactly one task class. "
                             "Reply with JSON only."},
                 {"role": "user", "content": text[:1200]}],
                schema, ctx=2048, think=False, num_predict=40)
            tc = (res.get("json") or {}).get("task_class")
            return tc if tc in allowed else None
        except Exception:
            return None

    # ------------------------------------------------------------ stage 3
    def _solve(self, task_class: str, needs_vision: bool, needs_tools: bool,
               text: str) -> dict:
        v = vram()
        free = v.get("free_mib", 0)
        live = RUNTIME.loaded()
        loaded = {m["model"] for m in live}
        # Capacity, not free space. Under the single-resident policy the
        # scheduler evicts the incumbent before loading a challenger, so VRAM
        # currently held by a resident model is available to that challenger.
        # Scoring against `free` alone charged the 3B coder a "will not fit"
        # penalty for memory that was about to be released, and handed coding
        # work to the general model.
        capacity = free + sum(m.get("vram_mib", 0) for m in live)
        candidates: list[ModelCard] = []
        for c in REGISTRY.available():
            if needs_vision and "image" not in c.modalities:
                continue
            if needs_tools and c.tool_calling == "none" and not needs_vision:
                continue
            candidates.append(c)
        if not candidates:                      # nothing installed yet
            candidates = [c for c in REGISTRY.cards.values()
                          if not needs_vision or "image" in c.modalities] \
                         or list(REGISTRY.cards.values())

        est_out_s = 1400 / 28.0                 # rough decode estimate at 28 tok/s
        scores: dict[str, float] = {}
        for c in candidates:
            resident = c.serve_name in loaded
            s = c.skill(task_class)
            s -= 0.0 if resident else LAMBDA_SWAP * c.load_cost_s
            s -= LAMBDA_LATENCY * est_out_s * (c.total_mib / 3000.0)
            # a model that cannot fit even after eviction is penalised, not banned
            if capacity > 0 and not resident and c.total_mib > capacity + 400:
                s -= 25
            scores[c.id] = round(s, 2)

        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        best_id = ordered[0][0]
        runner = ordered[1][0] if len(ordered) > 1 else None
        gap = round(ordered[0][1] - ordered[1][1], 2) if len(ordered) > 1 else 0.0

        # hysteresis — do not swap for a marginal win on a 4 GiB box
        reused = False
        resident_ids = [c.id for c in candidates if c.serve_name in loaded]
        if resident_ids and best_id not in resident_ids:
            rid = max(resident_ids, key=lambda i: scores[i])
            if scores[best_id] - scores[rid] < LAMBDA_HYSTERESIS:
                runner, best_id, reused = best_id, rid, True
        elif best_id in resident_ids:
            reused = True

        card = REGISTRY.get(best_id)
        role = card.roles[0] if card.roles else "general"
        return dict(role=role, model_id=card.id, serve_name=card.serve_name,
                    runner_up=runner, score_gap=gap, resident_reused=reused,
                    scores=scores, ctx=card.served_ctx, latency_ms=0.0)

    # ------------------------------------------------------------ logging
    def _log(self, text: str, d: Decision, run_id: str | None):
        try:
            con = db()
            con.execute(
                "INSERT INTO router_decisions(run_id,ts,request,stage,task_class,"
                "confidence,margin,chosen_model,runner_up,score_gap,resident_reused,"
                "latency_ms,features) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, time.time(), text[:600], d.stage, d.task_class,
                 d.confidence, d.margin, d.model_id, d.runner_up, d.score_gap,
                 int(d.resident_reused), d.latency_ms,
                 json.dumps({"signals": d.signals, "scores": d.scores})))
            con.commit()
        except Exception as e:
            print(f"[router] log failed: {e}")


ROUTER = Router()
