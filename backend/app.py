"""SynaptiX API — FastAPI, loopback only."""
from __future__ import annotations
import json
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import (StreamingResponse, FileResponse, HTMLResponse,
                               JSONResponse)
from fastapi.staticfiles import StaticFiles

from .config import UPLOADS, ARTIFACTS, WEB_DIR, DEMO_DIR
from .db import db, audit, verify_audit_chain
from .registry_loader import REGISTRY
from .router import ROUTER
from .runtime import RUNTIME, vram
from .egress import MONITOR
from .agent import Agent
from . import kb, sandbox

app = FastAPI(title="SynaptiX", docs_url="/api/docs")


@app.on_event("startup")
def _startup():
    REGISTRY.load()
    REGISTRY.mark_installed(RUNTIME.installed())
    MONITOR.start()
    MONITOR.sample()
    audit("system", "service.start",
          {"models": [c.id for c in REGISTRY.available()],
           "embedder": ROUTER.embedder.mode})


# ---------------------------------------------------------------- system
@app.get("/api/health")
def health():
    REGISTRY.mark_installed(RUNTIME.installed())
    ok, broken, n = verify_audit_chain()
    return {
        "ok": True,
        "ollama_up": RUNTIME.up(),
        "vram": vram(),
        "loaded": RUNTIME.loaded(),
        "models": [c.to_dict() for c in REGISTRY.cards.values()],
        "installed": [c.id for c in REGISTRY.available()],
        "embedder": ROUTER.embedder.mode,
        "exemplars": ROUTER.n_exemplars,
        "kb": kb.stats(),
        "egress": MONITOR.state,
        "sandbox_isolation": "docker" if sandbox.docker_available() else "subprocess",
        "audit": {"chain_ok": ok, "records": n, "broken_at": broken},
    }


@app.get("/api/egress")
def egress():
    return MONITOR.sample()


@app.get("/api/models")
def models():
    REGISTRY.mark_installed(RUNTIME.installed())
    return {"models": [c.to_dict() for c in REGISTRY.cards.values()],
            "loaded": RUNTIME.loaded(), "vram": vram()}


@app.post("/api/registry/reload")
def reload_registry():
    """Hot-add a model: drop a YAML in backend/registry/ and call this."""
    REGISTRY.load()
    REGISTRY.mark_installed(RUNTIME.installed())
    audit("user", "registry.reload", {"count": len(REGISTRY.cards)})
    return {"models": [c.to_dict() for c in REGISTRY.cards.values()]}


@app.post("/api/router/refit")
def refit():
    ROUTER.fit()
    return {"exemplars": ROUTER.n_exemplars, "classes": list(ROUTER.centroids)}


# ---------------------------------------------------------------- router probe
@app.post("/api/route")
def route(payload: dict):
    d = ROUTER.route(payload.get("text", ""),
                     has_image=payload.get("has_image", False),
                     has_pdf=payload.get("has_pdf", False),
                     filenames=payload.get("filenames", []))
    return d.dict()


@app.get("/api/router/decisions")
def decisions(limit: int = 30):
    rows = db().execute(
        "SELECT * FROM router_decisions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return {"decisions": [dict(r) for r in rows]}


# ---------------------------------------------------------------- knowledge base
@app.post("/api/kb/ingest")
async def kb_ingest(file: UploadFile = File(...),
                    classification: int = Form(1)):
    dest = UPLOADS / file.filename
    dest.write_bytes(await file.read())
    info = kb.ingest_file(dest, title=dest.stem, classification=classification)
    return info


@app.get("/api/kb/search")
def kb_search(q: str, k: int = 6):
    return {"query": q, "hits": kb.search(q, k=k)}


@app.get("/api/kb/stats")
def kb_stats():
    return kb.stats()


# ---------------------------------------------------------------- agent
@app.post("/api/run")
async def run_agent(goal: str = Form(...), files: list[UploadFile] = File(None)):
    paths = []
    for f in (files or []):
        if not f or not f.filename:
            continue
        p = UPLOADS / f.filename
        p.write_bytes(await f.read())
        paths.append(str(p))

    agent = Agent(goal, paths)

    def gen():
        try:
            for ev in agent.run():
                yield f"data: {json.dumps(ev, default=str)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'fatal','error':f'{type(e).__name__}: {e}'})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/runs")
def runs(limit: int = 20):
    rows = db().execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                        (limit,)).fetchall()
    return {"runs": [dict(r) for r in rows]}


@app.get("/api/run/{run_id}")
def run_detail(run_id: str):
    con = db()
    r = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not r:
        raise HTTPException(404)
    steps = con.execute("SELECT * FROM steps WHERE run_id=? ORDER BY idx",
                        (run_id,)).fetchall()
    arts = con.execute("SELECT * FROM artifacts WHERE run_id=?", (run_id,)).fetchall()
    return {"run": dict(r), "steps": [dict(s) for s in steps],
            "artifacts": [dict(a) for a in arts]}


# ---------------------------------------------------------------- artifacts
@app.get("/api/artifacts")
def artifacts(limit: int = 40):
    rows = db().execute("SELECT * FROM artifacts ORDER BY created_at DESC LIMIT ?",
                        (limit,)).fetchall()
    return {"artifacts": [dict(r) for r in rows]}


@app.get("/api/artifact/{name}")
def artifact(name: str):
    p = ARTIFACTS / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, filename=p.name)


# ---------------------------------------------------------------- audit
@app.get("/api/audit")
def audit_export(limit: int = 500):
    ok, broken, n = verify_audit_chain()
    rows = db().execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
                        (limit,)).fetchall()
    return JSONResponse({
        "chain_verified": ok, "broken_at": broken, "records": n,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "entries": [dict(r) for r in rows],
    }, headers={"Content-Disposition": "attachment; filename=kws-audit.json"})


# ---------------------------------------------------------------- demo helper
@app.get("/api/demo/files")
def demo_files():
    if not DEMO_DIR.exists():
        return {"files": []}
    return {"files": [{"name": p.name, "size": p.stat().st_size}
                      for p in sorted(DEMO_DIR.iterdir()) if p.is_file()]}


# ---------------------------------------------------------------- web
@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
