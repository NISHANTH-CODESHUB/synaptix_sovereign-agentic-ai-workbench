# SynaptiX — Sovereign On-Premise Agentic AI Workbench

Smart India Hackathon 2026 · Problem Statement 26117 (MRPL / Ministry of Petroleum & Natural Gas) · Team SynaptiX

A self-hosted, air-gapped AI workbench for refineries, PSUs and defence-linked
manufacturing. Multi-model, agentic, multimodal, and provably offline.

Running now on an **RTX 3050 Laptop, 4094 MiB VRAM**. Everything below is what
that box actually does, measured — not what the architecture could do.

---

## Set up (once)

Windows 11, Python 3.11, an NVIDIA GPU with 4 GB or more, and about 12 GB of
disk for models. Internet is needed **only** for this step; the workbench itself
never uses it.

```powershell
winget install Python.Python.3.11 Ollama.Ollama
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3:4b; ollama pull qwen2.5-coder:3b; ollama pull qwen2.5vl:3b; ollama pull qwen3:0.6b
```

The first start also downloads the small BGE embedding model (~130 MB) into
`data/models/embed/`; after that everything is cached and offline.

## Run it

```powershell
.\run.ps1 -Rebuild
```

`-Rebuild` generates the synthetic demo corpus (a scanned inspection report,
an SOP, a delegation-of-authority schedule, a shutdown report) and ingests it.
Use plain `.\run.ps1` afterwards. Then open <http://127.0.0.1:8080>.
Two supporting commands:

```powershell
.\.venv\Scripts\python.exe scripts\eval_router.py --sweep
```

```powershell
.\scripts\airgap.ps1 -On
```

The first measures router accuracy against a held-out set. The second (needs
Administrator) blocks outbound traffic for python and ollama, which turns the
egress badge green while the machine still has a working internet connection —
a stronger demonstration than simply unplugging the cable.

---

## What is actually built

| Component | State | Where |
|---|---|---|
| Four-stage capability router, with logged, explainable decisions | working | `backend/router.py` |
| Model registry — add a model by dropping in a YAML | working | `backend/registry/*.yaml` |
| Plan → act → verify agent over a persisted step DAG | working | `backend/agent.py` |
| Code sandbox: no network, no GPU, self-tested | working | `backend/sandbox.py` |
| OCR + layout intake, VLM used only where OCR is unsure | working | `backend/kb.py` |
| Hybrid retrieval, BM25 + dense, RRF fused | working | `backend/kb.py` |
| Word / Excel factory behind a numeric-provenance gate | working | `backend/docgen.py` |
| Deterministic calculation engine | working | `backend/agent.py` |
| Egress canary + socket monitor | working | `backend/egress.py` |
| Hash-chained audit log, verified on every health check | working | `backend/db.py` |

Deferred to the pilot, by choice: PostgreSQL + pgvector (SQLite in the prototype, same
schema shape), the vLLM adapter, Keycloak RBAC, the P&ID tiler, Grafana.

---

## Measured on this machine

Router, against 31 held-out queries that appear nowhere in the exemplar set
(`scripts/eval_router.py`):

| Configuration | Accuracy | Stage 2 fires on |
|---|---|---|
| Embedding lane alone | **100 %** | 0 % |
| + Qwen3-4B judge, margin 0.035 | 100 % | 9.7 % |
| + Qwen3-4B judge, margin 0.090 | 80.6 % | 71 % |
| + Qwen3-0.6B judge, margin 0.090 | 71.0 % | 71 % |

Two things that table settles, and they are worth saying out loud:

1. **A 0.6B judge is worse than the embedding lane it was meant to help.** The
   cascade is configured to adjudicate rarely (margin 0.035) and only with the
   4B, and only when that model is already resident.
2. **31 queries is a small set and 100 % should not be read as "solved."** The
   held-out items are close in distribution to the exemplars. The number to
   quote is the *shape* — the embedding lane carries the work — not the ceiling.

Other measurements:

| Metric | Value |
|---|---|
| Routing overhead | 5–7 ms (p50), stage 1 |
| Decode, Qwen3-4B Q4_K_M @ ctx 6144 | 14 tok/s |
| Decode, Qwen2.5-Coder-3B | 55 tok/s |
| Warm model swap | 3.1–4.7 s |
| Cold VL swap | 18 s |
| OCR, 2-page scan, CPU | ~19 s, mean confidence 0.983 |
| Scan → verified Word note, end to end | ~150 s |
| Calculation, end to end | ~35 s |
| Peak VRAM | 3787 / 4094 MiB |

### The VRAM fix worth showing

`qwen3:4b` at ctx 8192 with an fp16 KV cache needs 4.1 GB and does not fit, so
Ollama pushes 45 % of it onto the CPU and decode drops to 7.8 tok/s. Enabling
`OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`, and serving 6144
instead of 8192, takes it to 3.4 GB and 14 tok/s. Same card, same model,
1.8× faster — the VRAM budget from the design, done rather than described.

---

## Things that were fixed because testing found them

These are in the code with comments explaining why. They are worth knowing
about, because each one is a question a judge could ask.

- **Qwen3's hybrid reasoning ignores `think: false`.** Verified: `qwen3:4b`
  returns its scratchpad in `content` with the `thinking` field empty. Left
  alone it wrote "Okay, let's tackle this problem…" into a draft approval note.
  Fixed three ways: schema-constrained generation wherever output feeds a
  deliverable (a grammar makes a monologue unrepresentable), `strip_reasoning()`
  on free text, and no raw token streaming into documents.
- **The provenance gate produced false positives on OCR text.** OCR collapses
  whitespace, so `9.5   6.1` arrives as `9.56.1` and tokenising alone reported
  a figure that *is* in the source as invented. Now matched against the
  whitespace-stripped stream as well.
- **Small coder models interleave prose with code.** A deterministic repair loop
  compiles, drops the offending line, and recompiles — cheaper and more reliable
  than another generation round. It recovered a run that had failed twice.
- **A 4B model asked to write a whole calculation script made a sign error**,
  negating the corrosion rate so a negative remaining life came out positive.
  So the model no longer does arithmetic: it names the quantities and the single
  relation, and the machine evaluates it. Now exact, and roughly 4× faster.
- **The VLM ran on every scan.** With OCR confidence at 0.98 that was 55 s of a
  180 s run for no gain. The VLM is now reserved for low-confidence pages,
  drawings, and images that are not text documents.
- **The eval harness was measuring the wrong thing.** It never marked models
  installed, so stage 2 silently never fired and it reported the embedding
  lane's accuracy as the cascade's. Found and fixed before it flattered us.

---

## Known limits, stated plainly

- Small coder models still fail their own assertions on the first try. The
  escalation ladder (3B → 3B → 4B) plus the repair loop usually recovers it, but
  a coding task can take three rungs and about 90 seconds.
- The stage-2 judge declines to run when no model is resident, rather than pay a
  7 s cold load to break a tie. Good behaviour, but it means the very first
  query of a session never uses stage 2.
- SQLite, not PostgreSQL. Retrieval is a linear scan over chunk vectors, which is
  fine at this corpus size and would not be at a million chunks.
- The demo corpus is synthetic and marked as such in every document.
- The egress badge reports the truth. On a laptop with Wi-Fi on it says OPEN,
  because it is. That is the point: green means the block was *tested*.

---

## Layout

```
backend/          app, router, agent, runtime adapter, kb, docgen, sandbox, egress
backend/registry/ one YAML per model — the whole model fabric
web/index.html    the workbench UI
scripts/          demo corpus builder, router eval, CLI runner, air-gap script
demo_corpus/      generated by scripts/make_demo_corpus.py (not committed)
data/             runtime state, not committed - except data/exemplars.jsonl,
                  the router's training set
DEMO.md           the demo run sheet
```
