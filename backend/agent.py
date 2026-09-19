"""Plan → Act → Verify agent runtime.

The planner is an LLM call constrained to a JSON schema whose step names are an
enum, so a 3-4B model cannot emit an unrunnable plan. If it fails entirely we
fall back to the class-default plan — the system degrades, it never dies.

Every step is persisted before it runs, so a run survives a crash and can be
replayed for audit. The governor caps wall clock, tool calls and repeats.
"""
from __future__ import annotations
import base64
import json
import re
import time
import uuid
from pathlib import Path
from typing import Iterator

from .config import MAX_STEPS, MAX_TOOL_CALLS, MAX_WALL_CLOCK_S, UPLOADS
from .db import db, audit
from .registry_loader import REGISTRY
from .router import ROUTER
from .runtime import RUNTIME, vram
from . import kb, docgen, sandbox

STEP_TYPES = [
    "kb.search", "vision.read", "extract.fields", "units.calc",
    "python.exec", "doc.render_docx", "doc.render_xlsx", "summarise", "answer",
]

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {"type": "array", "items": {"type": "string"},
                  "minItems": 1, "maxItems": 6},
    },
    "required": ["lines"],
}

# The model never does arithmetic. It identifies the quantities and the single
# relation between them; the machine evaluates it. A 4B model asked to write a
# whole calculation script produced a sign error (it negated the corrosion rate
# to make a negative remaining life come out positive). Asked only to name the
# quantities and the formula, it is reliable - and the result is deterministic
# and re-checkable by a human.
CALC_SCHEMA = {
    "type": "object",
    "properties": {
        "quantity": {"type": "string"},
        "unit": {"type": "string"},
        "given": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                },
                "required": ["name", "value", "unit"],
            },
        },
        "expression": {"type": "string"},
        "basis": {"type": "string"},
    },
    "required": ["quantity", "unit", "given", "expression"],
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tool": {"type": "string", "enum": STEP_TYPES},
                    "why": {"type": "string"},
                },
                "required": ["name", "tool", "why"],
            },
        }
    },
    "required": ["steps"],
}

DEFAULT_PLANS = {
    "vision_doc": ["vision.read", "extract.fields", "kb.search", "doc.render_docx"],
    "vision_drawing": ["vision.read", "extract.fields", "answer"],
    "structured_extract": ["kb.search", "extract.fields", "doc.render_xlsx"],
    "doc_draft": ["kb.search", "doc.render_docx"],
    "doc_summarise": ["kb.search", "summarise"],
    "kb_qa": ["kb.search", "answer"],
    "calculation": ["units.calc", "answer"],
    "code_write": ["python.exec", "summarise"],
    "code_debug": ["python.exec", "summarise"],
    "general_qa": ["answer"],
}


CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def _extract_code(raw: str) -> str:
    """Small coder models wrap code in fences AND append prose explanations.
    Take the fenced block when there is one; otherwise cut the text at the first
    line that is clearly not Python. Getting this wrong surfaces as a
    SyntaxError on a line of English - exactly what we hit in testing."""
    raw = (raw or "").strip()
    blocks = CODE_FENCE.findall(raw)
    if blocks:
        return max(blocks, key=len).strip()
    out = []
    for ln in raw.splitlines():
        st = ln.strip()
        if not st or ln[:1] in " \t#":
            out.append(ln)
            continue
        looks_py = bool(re.match(
            r"^(import |from |def |class |if |for |while |try|except|else|elif|"
            r"with |return |print\(|assert |@|[\w\[\]\.]+\s*[-+*/]?=[^=]|"
            r"\w+\(|\)|\]|\}|\w+\.\w+)", st))
        if looks_py:
            out.append(ln)
        elif out:
            break
    return "\n".join(out).strip() or raw


def _syntax_repair(code: str, max_drops: int = 8) -> tuple[str, int]:
    """Deterministically delete lines that are not Python.

    Small models interleave explanation with code ("t_min is the minimum
    thickness allowed. So if..."), which the model then cannot fix by itself
    because it does not believe it wrote prose. Compiling and dropping the
    offending line is cheaper and more reliable than another generation round.
    """
    dropped = 0
    for _ in range(max_drops):
        try:
            compile(code, "<model>", "exec")
            return code, dropped
        except SyntaxError as e:
            ln = (e.lineno or 0) - 1
            lines = code.splitlines()
            if not (0 <= ln < len(lines)):
                return code, dropped
            lines.pop(ln)
            code = "\n".join(lines)
            dropped += 1
        except Exception:
            return code, dropped
    return code, dropped

def _ev(kind: str, **kw) -> dict:
    return {"type": kind, "t": round(time.time(), 3), **kw}


class Governor:
    def __init__(self):
        self.t0 = time.time()
        self.tool_calls = 0
        self.seen: dict[str, int] = {}

    def check(self, state_hash: str) -> str | None:
        if time.time() - self.t0 > MAX_WALL_CLOCK_S:
            return "wall-clock budget exhausted"
        if self.tool_calls > MAX_TOOL_CALLS:
            return "tool-call budget exhausted"
        self.seen[state_hash] = self.seen.get(state_hash, 0) + 1
        if self.seen[state_hash] >= 3:
            return "cycle detected (identical state 3×)"
        return None


class Agent:
    def __init__(self, goal: str, attachments: list[str] | None = None):
        self.goal = goal
        self.attachments = [Path(a) for a in (attachments or [])]
        self.run_id = str(uuid.uuid4())
        self.gov = Governor()
        self.evidence: list[str] = []      # every string a claim may draw on
        self.citations: list[dict] = []
        self.review_flags: list[dict] = []
        self.facts: dict = {}
        self.artifacts: list[dict] = []

    # ------------------------------------------------------------------ run
    def run(self) -> Iterator[dict]:
        con = db()
        con.execute("INSERT INTO runs(id,goal,status,created_at) VALUES(?,?,?,?)",
                    (self.run_id, self.goal, "running", time.time()))
        con.commit()
        audit("user", "run.start", {"run_id": self.run_id, "goal": self.goal[:400]})

        has_image = any(a.suffix.lower() in
                        (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
                        for a in self.attachments)
        has_pdf = any(a.suffix.lower() == ".pdf" for a in self.attachments)

        d = ROUTER.route(self.goal, has_image=has_image, has_pdf=has_pdf,
                         filenames=[a.name for a in self.attachments],
                         run_id=self.run_id)
        yield _ev("route", decision=d.dict())

        # No upfront load: on a 4 GiB box only one model is resident, and each
        # step ensures the model it actually needs. Loading the routed model here
        # too pays an 18 s swap that the first step immediately discards.
        plan = self._plan(d.task_class)
        yield _ev("plan", steps=plan, run_id=self.run_id)
        con.execute("UPDATE runs SET plan=? WHERE id=?",
                    (json.dumps(plan), self.run_id))
        con.commit()

        for idx, step in enumerate(plan):
            stop = self.gov.check(f"{idx}:{step['tool']}")
            if stop:
                yield _ev("abort", reason=stop)
                break
            yield _ev("step_start", idx=idx, name=step["name"], tool=step["tool"],
                      why=step.get("why", ""))
            sid = self._persist_step(idx, step)
            t0 = time.time()
            try:
                for out in self._execute(step, d):
                    yield out
                    if out.get("type") == "step_result":
                        self._finish_step(sid, "done", out.get("data"),
                                          out.get("verify_ok"), out.get("verify_note"))
            except Exception as e:
                self._finish_step(sid, "error", {"error": str(e)}, 0, str(e))
                yield _ev("step_error", idx=idx, error=f"{type(e).__name__}: {e}")
            yield _ev("step_end", idx=idx, seconds=round(time.time() - t0, 2))

        con.execute("UPDATE runs SET status=?, finished_at=? WHERE id=?",
                    ("done", time.time(), self.run_id))
        con.commit()
        audit("system", "run.end",
              {"run_id": self.run_id, "artifacts": [a["filename"] for a in self.artifacts]})
        yield _ev("done", run_id=self.run_id, artifacts=self.artifacts,
                  citations=self.citations, vram=vram())

    # ----------------------------------------------------------------- plan
    def _plan(self, task_class: str) -> list[dict]:
        default = DEFAULT_PLANS.get(task_class, ["answer"])
        if not self.attachments:
            default = [s for s in default if s != "vision.read"]
        fallback = [{"name": t.split(".")[-1], "tool": t,
                     "why": f"class default for {task_class}"} for t in default]
        card = REGISTRY.get("qwen3-4b") or REGISTRY.get("qwen3-06b")
        if not card or not card.installed:
            return fallback
        try:
            res = RUNTIME.structured(
                card.serve_name,
                [{"role": "system",
                  "content": "You plan short tool pipelines for an industrial "
                             "document assistant. Choose 2-5 steps from the allowed "
                             "tools, in execution order. JSON only."},
                 {"role": "user",
                  "content": f"Task class: {task_class}\nAttachments: "
                             f"{[a.name for a in self.attachments]}\n"
                             f"Goal: {self.goal}\n"
                             f"Allowed tools: {STEP_TYPES}\n"
                             f"A sensible default is: {default}"}],
                PLAN_SCHEMA, ctx=2048, think=False, num_predict=300)
            steps = (res.get("json") or {}).get("steps") or []
            steps = [s for s in steps if s.get("tool") in STEP_TYPES][:MAX_STEPS]
            if not self.attachments:
                steps = [s for s in steps if s["tool"] != "vision.read"]
            return steps or fallback
        except Exception:
            return fallback

    def _persist_step(self, idx: int, step: dict) -> int:
        con = db()
        cur = con.execute(
            "INSERT INTO steps(run_id,idx,name,tool,status,started_at) "
            "VALUES(?,?,?,?,?,?)",
            (self.run_id, idx, step["name"], step["tool"], "running", time.time()))
        con.commit()
        return cur.lastrowid

    def _finish_step(self, sid: int, status: str, output, vok, vnote):
        con = db()
        con.execute(
            "UPDATE steps SET status=?,output=?,verify_ok=?,verify_note=?,"
            "finished_at=? WHERE id=?",
            (status, json.dumps(output, default=str)[:20000],
             1 if vok else 0, vnote, time.time(), sid))
        con.commit()

    # -------------------------------------------------------------- execute
    def _execute(self, step: dict, d) -> Iterator[dict]:
        tool = step["tool"]
        self.gov.tool_calls += 1
        fn = getattr(self, "_t_" + tool.replace(".", "_"), None)
        if fn is None:
            yield _ev("step_result", data={"skipped": tool}, verify_ok=True,
                      verify_note="no handler")
            return
        yield from fn(d)

    # ------------------------------------------------------------ tools
    def _t_vision_read(self, d) -> Iterator[dict]:
        """OCR first (deterministic, CPU), VLM only where OCR is unsure."""
        if not self.attachments:
            yield _ev("step_result", data={"note": "no attachment"}, verify_ok=True,
                      verify_note="skipped")
            return
        results = []
        for path in self.attachments:
            info = kb.ingest_file(path, title=path.stem, doc_type="scanned")
            pages = kb.extract_pages(path)
            # a re-uploaded file is deduped, so the ingest summary carries no
            # OCR stats; take them from the extraction we just did either way
            source = pages[0]["source"] if pages else "none"
            mean_conf = (round(sum(p["ocr_conf"] for p in pages) / len(pages), 3)
                         if pages else None)
            low = [w for p in pages for w in p["words"] if w["conf"] < 0.70]
            text = "\n".join(p["text"] for p in pages)
            self.evidence.append(text)
            results.append({
                "file": path.name, "pages": len(pages),
                "source": source, "chunks": info.get("chunks"),
                "mean_ocr_conf": mean_conf, "deduped": info.get("deduped", False),
                "low_conf_cells": len(low),
                "preview": text[:600],
            })
            for w in low[:6]:
                self.review_flags.append(
                    {"field": w["text"][:40], "reason": f"OCR confidence {w['conf']}"})
            yield _ev("tool_log", tool="ocr",
                      msg=f"{path.name}: {len(pages)}p via {source}, "
                          f"mean conf {mean_conf}, "
                          f"{len(low)} low-confidence cells flagged")

            # VLM pass — the multimodal lane. Deliberately NOT unconditional:
            # OCR is deterministic, CPU-only and far cheaper, so the VLM is
            # reserved for pages OCR could not read confidently, for drawings,
            # and for images that are not text documents at all.
            conf = mean_conf or 0.0
            sparse = len(text.strip()) < 200
            need_vlm = (conf < 0.85 or bool(low) or sparse
                        or d.task_class == "vision_drawing")
            vcard = next((c for c in REGISTRY.available()
                          if "image" in c.modalities), None)
            if not need_vlm:
                yield _ev("tool_log", tool="vision",
                          msg=f"VLM skipped - OCR confident ({conf}), "
                              f"{len(low)} low-confidence cells; saved a model swap")
            if need_vlm and vcard and path.suffix.lower() in (
                    ".png", ".jpg", ".jpeg", ".webp"):
                swap = RUNTIME.ensure(vcard.serve_name, ctx=vcard.served_ctx)
                if swap:
                    yield _ev("swap", model=vcard.serve_name,
                              seconds=round(swap, 2), vram=vram())
                b64 = base64.b64encode(path.read_bytes()).decode()
                vr = RUNTIME.chat(
                    vcard.serve_name,
                    [{"role": "user",
                      "content": "Describe this document image: its type, and any "
                                 "readings, tags or observations visible. Be terse "
                                 "and factual."}],
                    images=[b64], ctx=vcard.served_ctx, num_predict=400)
                self.evidence.append(vr["text"])
                results[-1]["vlm"] = vr["text"][:800]
                results[-1]["vlm_model"] = vcard.id
                yield _ev("tool_log", tool="vision",
                          msg=f"{vcard.id}: {vr['tokens_out']} tok @ {vr['tps']} tok/s")
        self.facts["ingest"] = results
        yield _ev("step_result", data=results, verify_ok=bool(results),
                  verify_note="pages extracted")

    def _t_kb_search(self, d) -> Iterator[dict]:
        hits = kb.search(self.goal, k=6)
        for h in hits:
            self.evidence.append(h["content"])
            self.citations.append({
                "title": h["title"], "page": h["page"], "bbox": h["bbox"],
                "ocr_conf": h["ocr_conf"], "document_id": h["document_id"],
                "chunk_id": h["chunk_id"], "score": h["score"],
                "lanes": ("bm25+dense" if h["in_bm25"] and h["in_dense"]
                          else "bm25" if h["in_bm25"] else "dense"),
            })
        yield _ev("tool_log", tool="kb.search",
                  msg=f"{len(hits)} chunks after RRF fusion "
                      f"({sum(1 for h in hits if h['in_bm25'])} bm25, "
                      f"{sum(1 for h in hits if h['in_dense'])} dense)")
        self.facts["kb"] = hits
        yield _ev("step_result", data=hits, verify_ok=True,
                  verify_note=f"{len(hits)} grounded chunks")

    def _t_extract_fields(self, d) -> Iterator[dict]:
        card = REGISTRY.get("qwen3-4b") or REGISTRY.get("qwen3-06b")
        swap = RUNTIME.ensure(card.serve_name, ctx=card.served_ctx)
        if swap:
            yield _ev("swap", model=card.serve_name, seconds=round(swap, 2),
                      vram=vram())
        schema = {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item": {"type": "string"},
                            "value": {"type": "string"},
                            "unit": {"type": "string"},
                            "status": {"type": "string"},
                        },
                        "required": ["item", "value"],
                    },
                }
            },
            "required": ["findings"],
        }
        ctx_text = "\n\n".join(self.evidence)[:4500]
        res = RUNTIME.structured(
            card.serve_name,
            [{"role": "system",
              "content": "Extract factual findings from the source text. Copy "
                         "values EXACTLY as they appear. Never invent a number. "
                         "JSON only."},
             {"role": "user", "content": f"SOURCE:\n{ctx_text}\n\nTASK: {self.goal}"}],
            schema, ctx=card.served_ctx, think=False, num_predict=900)
        findings = (res.get("json") or {}).get("findings", [])
        self.facts["findings"] = findings
        ok = len(findings) > 0
        yield _ev("tool_log", tool="extract",
                  msg=f"{len(findings)} findings, schema-constrained "
                      f"({res['tokens_out']} tok @ {res['tps']} tok/s)")
        yield _ev("step_result", data=findings, verify_ok=ok,
                  verify_note="schema valid" if ok else "no findings extracted")

    def _t_python_exec(self, d) -> Iterator[dict]:
        card = (REGISTRY.get("qwen25-coder-3b") if d.task_class.startswith("code")
                else REGISTRY.get("qwen3-4b")) or REGISTRY.get("qwen3-4b")
        if not card.installed:
            card = next(iter(REGISTRY.available()), card)
        swap = RUNTIME.ensure(card.serve_name, ctx=card.served_ctx)
        if swap:
            yield _ev("swap", model=card.serve_name, seconds=round(swap, 2),
                      vram=vram())

        # Escalation ladder (design §4): two attempts on the specialist, then a
        # genuine model escalation to the strongest general model available.
        # A 3B coder writes plausible code with wrong self-checks; the 4B fixes
        # its own asserts more often than the 3B does.
        ladder = [card, card, REGISTRY.get("qwen3-4b") or card]
        note = ""
        for attempt, rung in enumerate(ladder):
            card = rung if (rung and rung.installed) else card
            swap = RUNTIME.ensure(card.serve_name, ctx=card.served_ctx)
            if swap:
                yield _ev("swap", model=card.serve_name, seconds=round(swap, 2),
                          vram=vram())
            prompt = (
                "Write ONE self-contained Python script.\n"
                "RULES:\n"
                "- Output ONLY code. No explanation before or after it.\n"
                "- Use plain float arithmetic. Put units in comments, NOT in a "
                "units library.\n"
                "- Allowed imports: csv, math, json, statistics, pathlib, numpy, "
                "pandas, matplotlib (Agg backend).\n"
                "- No network, no input(), no argparse.\n"
                "- print() the results as a readable table. The script MUST "
                "produce visible output; a silent script is a failed script.\n"
                "- END with 2-3 assert statements. Compare floats with a "
                "tolerance, e.g. assert abs(x - 7.14) < 0.01 - never with ==.\n"
                "- Only assert things your own code actually computes.\n"
                "- Save any CSV or chart into the current directory.\n"
                "- Do NOT hardcode expected answers in asserts. Assert "
                "RELATIONSHIPS you can be sure of.\n\n"
                "EXAMPLE of the expected shape and style:\n"
                "import csv\n"
                "t_min = 6.4          # mm\n"
                "rate = 0.21          # mm/year\n"
                "rows = []\n"
                "for tag, meas in [(\"A-01\", 8.0), (\"A-02\", 6.2)]:\n"
                "    life = (meas - t_min) / rate      # years\n"
                "    flag = \"BELOW t-min\" if meas <= t_min else \"OK\"\n"
                "    rows.append([tag, meas, round(life, 2), flag])\n"
                "    print(f\"{tag:8} {meas:6.2f} {life:8.2f} {flag}\")\n"
                "with open(\"results.csv\", \"w\", newline=\"\") as f:\n"
                "    w = csv.writer(f)\n"
                "    w.writerow([\"tag\", \"measured_mm\", \"life_years\", \"status\"])\n"
                "    w.writerows(rows)\n"
                "assert len(rows) == 2\n"
                "assert all(r[2] <= 0 for r in rows if r[3] != \"OK\")\n\n"
                f"TASK: {self.goal}")
            if note:
                prompt += f"\n\nYour previous attempt failed with:\n{note}\nFix it."
            res = RUNTIME.chat(card.serve_name,
                               [{"role": "user", "content": prompt}],
                               ctx=card.served_ctx, temperature=0.1,
                               num_predict=1200)
            code = _extract_code(res["text"])
            code, dropped = _syntax_repair(code)
            if dropped:
                yield _ev("tool_log", tool="repair",
                          msg=f"dropped {dropped} non-Python line(s) the model "
                              f"interleaved with its code, then re-compiled")
            yield _ev("code", code=code, model=card.id, attempt=attempt + 1,
                      tps=res["tps"])
            out = sandbox.run_python(code, timeout=40)
            self.gov.tool_calls += 1
            yield _ev("sandbox", result={k: v for k, v in out.items()
                                         if k != "artifacts"},
                      artifacts=[{"name": a["name"], "size": a["size"]}
                                 for a in out["artifacts"]])
            if out["ok"]:
                stdout = (out["stdout"] or "").strip()
                self.evidence.append(
                    f"[verified sandbox output]\n{stdout}" if stdout else
                    "[the script ran and passed its own assertions but printed "
                    "no output]")
                self.evidence.append(f"[the script that produced it]\n{code}")
                self.facts["exec"] = {"code": code, "stdout": out["stdout"]}
                for a in out["artifacts"]:
                    p = Path(docgen.ARTIFACTS) / f"{self.run_id[:8]}-{a['name']}"
                    p.write_bytes(a["bytes"])
                    self.artifacts.append({"filename": p.name, "kind": "file",
                                           "path": str(p), "size": a["size"]})
                yield _ev("step_result", data={"stdout": out["stdout"][:2000]},
                          verify_ok=True,
                          verify_note=f"executed + self-asserted in sandbox "
                                      f"({out['duration_s']}s, network denied)")
                return
            note = (out["stderr"] or "")[-800:]
            if out.get("stdout"):
                self.evidence.append(
                    f"[partial output, attempt {attempt+1}]\n{out['stdout']}")
            nxt = ladder[attempt + 1].id if attempt + 1 < len(ladder) else None
            yield _ev("tool_log", tool="verify",
                      msg=f"attempt {attempt+1} failed -> rung {attempt+1}"
                          + (f", escalating to {nxt}" if nxt else ", no rungs left"))
        self.facts["exec"] = {"code": code, "error": note}
        self.evidence.append(
            "The script could not be made to pass its own checks. Last error:\n"
            + note)
        yield _ev("step_result", data={"error": note}, verify_ok=False,
                  verify_note=f"failed after {len(ladder)} rungs - "
                              "reported as a failure, not as a result")

    def _t_units_calc(self, d) -> Iterator[dict]:
        """Deterministic calculation: model supplies the spec, we evaluate it."""
        card = REGISTRY.get("qwen3-4b") or next(iter(REGISTRY.available()), None)
        swap = RUNTIME.ensure(card.serve_name, ctx=card.served_ctx)
        if swap:
            yield _ev("swap", model=card.serve_name, seconds=round(swap, 2),
                      vram=vram())
        ev = "\n\n".join(self.evidence)[:3000]
        res = RUNTIME.structured(
            card.serve_name,
            [{"role": "system",
              "content": "You set up engineering calculations. Name each given "
                         "quantity with its value and unit, then give ONE Python "
                         "expression using those names. Do NOT compute the answer "
                         "yourself and do NOT change the sign of any given value. "
                         "Example for remaining life: given t_measured, t_min, "
                         "corrosion_rate; expression '(t_measured - t_min) / "
                         "corrosion_rate'. JSON only."},
             {"role": "user",
              "content": (f"SOURCE:\n{ev}\n\n" if ev else "") + f"TASK: {self.goal}"}],
            CALC_SCHEMA, ctx=card.served_ctx, num_predict=500)
        spec = res.get("json") or {}
        given = spec.get("given") or []
        expr = str(spec.get("expression") or "").strip()
        env = {g["name"]: float(g["value"]) for g in given
               if isinstance(g.get("value"), (int, float))}
        yield _ev("tool_log", tool="calc",
                  msg="given: " + ", ".join(f"{k}={v}" for k, v in env.items())
                      + f"  |  expression: {expr}")

        if not expr or not env:
            yield _ev("step_result", data=spec, verify_ok=False,
                      verify_note="could not form a calculation spec")
            return
        if not re.fullmatch(r"[A-Za-z0-9_+\-*/(). ]+", expr):
            yield _ev("step_result", data=spec, verify_ok=False,
                      verify_note=f"expression rejected by the guard: {expr!r}")
            return
        try:
            value = eval(expr, {"__builtins__": {}}, dict(env))   # guarded above
        except Exception as e:
            yield _ev("step_result", data={"error": str(e)}, verify_ok=False,
                      verify_note=f"evaluation failed: {e}")
            return

        subst = expr
        for k, v in sorted(env.items(), key=lambda kv: -len(kv[0])):
            subst = subst.replace(k, f"{v:g}")
        steps = [
            f"Quantity : {spec.get('quantity','result')}",
            *[f"Given    : {g['name']} = {g['value']} {g['unit']}" for g in given],
            f"Formula  : {expr}",
            f"Substitute: {subst}",
            f"Result   : {value:.4g} {spec.get('unit','')}",
        ]
        if spec.get("basis"):
            steps.append(f"Basis    : {spec['basis']}")
        block = "\n".join(steps)
        self.evidence.append("[deterministic calculation]\n" + block)
        self.facts["calc"] = {"expression": expr, "given": env, "value": value}
        for ln in steps:
            yield _ev("tool_log", tool="calc-step", msg=ln)
        yield _ev("step_result", data=self.facts["calc"], verify_ok=True,
                  verify_note=f"evaluated deterministically: {value:.4g} "
                              f"{spec.get('unit','')}")

    SUMMARY_SYS = (
        "You are reporting work that has ALREADY been completed on-premises. "
        "The SOURCE below is real output from a script that has already run in a "
        "sandbox. Do NOT redo the reasoning, do NOT rewrite the code, do NOT "
        "narrate your thinking. In at most 6 short lines, state what was computed "
        "and what the numbers show, quoting the actual figures. Plant "
        "engineering register.")

    def _t_summarise(self, d) -> Iterator[dict]:
        yield from self._t_answer(d, mode="summary")

    def _t_answer(self, d, mode: str = "answer") -> Iterator[dict]:
        card = REGISTRY.get(d.model_id) or REGISTRY.get("qwen3-4b")
        if not card.installed:
            card = next(iter(REGISTRY.available()), card)
        swap = RUNTIME.ensure(card.serve_name, ctx=card.served_ctx)
        if swap:
            yield _ev("swap", model=card.serve_name, seconds=round(swap, 2),
                      vram=vram())
        ev = "\n\n---\n".join(self.evidence)[:5000]
        exec_failed = bool(self.facts.get("exec", {}).get("error"))
        if mode == "summary" and (exec_failed or not ev.strip()):
            msg = ("The task could not be completed: the generated script did "
                   "not pass verification in the sandbox, so there is no "
                   "result to report. Nothing has been fabricated."
                   if exec_failed else
                   "The step completed but produced no output to summarise.")
            yield _ev("token", text=msg)
            yield _ev("step_result", data={"text": msg}, verify_ok=False,
                      verify_note="no verified output to summarise")
            return
        sys_msg = (self.SUMMARY_SYS if mode == "summary" else
                   "You are an assistant inside an air-gapped industrial plant. "
                   "Answer ONLY from the SOURCE material when it is present, and "
                   "say plainly when it does not cover something. Do not narrate "
                   "your reasoning; give the answer directly. Be concise and use "
                   "plant engineering register.")
        user = (f"SOURCE:\n{ev}\n\n" if ev else "") + f"REQUEST: {self.goal}"
        # Summaries are schema-constrained. The hybrid-reasoning tag leaks its
        # scratchpad into `content` (ollama's think:false is ignored for it), and
        # a monologue must never reach a deliverable. Under a grammar it is not
        # merely discouraged - it is unrepresentable.
        if mode == "summary":
            res = RUNTIME.structured(
                card.serve_name,
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": user}],
                SUMMARY_SCHEMA, ctx=card.served_ctx, num_predict=400)
            lines = (res.get("json") or {}).get("lines") or []
            clean = "\n".join(str(x).strip() for x in lines if str(x).strip())
            clean = clean or "No verified result to report."
            self.facts["answer"] = clean
            for i in range(0, len(clean), 48):
                yield _ev("token", text=clean[i:i + 48])
            yield _ev("step_result", data={"text": clean}, verify_ok=True,
                      verify_note="schema-constrained summary of executed output")
            return

        # Free-text answers: generate, scrub, then emit in chunks so the UI
        # still animates.
        res = RUNTIME.chat(card.serve_name,
                           [{"role": "system", "content": sys_msg},
                            {"role": "user", "content": user}],
                           ctx=card.served_ctx, temperature=0.3,
                           think=bool(getattr(d, "think", False)),
                           num_predict=900)
        clean = res["text"].strip()
        buf = [clean]
        for i in range(0, len(clean), 48):
            yield _ev("token", text=clean[i:i + 48])
        text = "".join(buf)
        self.facts["answer"] = text
        yield _ev("step_result", data={"text": text[:4000]}, verify_ok=bool(text),
                  verify_note=f"grounded in {len(self.citations)} citations"
                              if self.citations else "no KB grounding available")

    def _t_doc_render_docx(self, d) -> Iterator[dict]:
        card = REGISTRY.get("qwen3-4b") or next(iter(REGISTRY.available()), None)
        swap = RUNTIME.ensure(card.serve_name, ctx=card.served_ctx)
        if swap:
            yield _ev("swap", model=card.serve_name, seconds=round(swap, 2),
                      vram=vram())
        ev = "\n\n---\n".join(self.evidence)[:4500]
        findings = json.dumps(self.facts.get("findings", []))[:2000]
        res = RUNTIME.structured(
            card.serve_name,
            [{"role": "system",
              "content": "You draft Indian PSU office notes. Produce a DocumentSpec. "
                         "Every figure you write MUST already appear in the SOURCE. "
                         "Never invent numbers, dates or references. JSON only."},
             {"role": "user",
              "content": f"SOURCE:\n{ev}\n\nEXTRACTED FINDINGS:\n{findings}\n\n"
                         f"TASK: {self.goal}"}],
            docgen.DOCUMENT_SPEC_SCHEMA, ctx=card.served_ctx,
            think=False, num_predict=1600)
        spec = res.get("json") or {}
        spec.setdefault("doc_type", "Note for Approval")
        yield _ev("tool_log", tool="draft",
                  msg=f"DocumentSpec: {len(spec.get('sections',[]))} sections "
                      f"({res['tokens_out']} tok @ {res['tps']} tok/s)")

        gate = docgen.provenance_check(spec, self.evidence)
        yield _ev("provenance", gate=gate)
        if not gate["passed"]:
            for n in gate["unsupported"]:
                self.review_flags.append(
                    {"field": f"figure {n}", "reason": "not traceable to a source — "
                                                       "model-generated"})
        art = docgen.render_docx(spec, citations=self.citations,
                                 review_flags=self.review_flags,
                                 run_id=self.run_id)
        self.artifacts.append(art)
        yield _ev("artifact", artifact=art)
        yield _ev("step_result", data=art, verify_ok=gate["passed"],
                  verify_note=(f"provenance {len(gate['supported'])}/{gate['total']} "
                               f"figures traced"
                               + ("" if gate["passed"] else " — flagged in document")))

    def _t_doc_render_xlsx(self, d) -> Iterator[dict]:
        findings = self.facts.get("findings") or []
        if not findings:
            yield _ev("step_result", data={"note": "nothing to tabulate"},
                      verify_ok=True, verify_note="skipped")
            return
        cols = ["Item", "Value", "Unit", "Status"]
        rows = [[f.get("item", ""), f.get("value", ""), f.get("unit", ""),
                 f.get("status", "")] for f in findings]
        art = docgen.render_xlsx("Findings", cols, rows,
                                 notes=[f"Generated on-premises · run {self.run_id[:8]}"],
                                 run_id=self.run_id)
        self.artifacts.append(art)
        yield _ev("artifact", artifact=art)
        yield _ev("step_result", data=art, verify_ok=True,
                  verify_note=f"{len(rows)} rows written")
