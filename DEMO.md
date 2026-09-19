# Run sheet — prototype demo

Total 6–7 minutes. Ordered so the riskiest thing runs after two wins are banked.

## Before you start

```powershell
.\run.ps1
```

Wait for `Application startup complete`, open <http://127.0.0.1:8080>, and check
the header reads four models, 3 docs / 7 chunks, audit chain verified.

Warm the 4B once so the first demo is not paying a cold load — ask it anything
trivial and discard the answer. **Do this before the judges sit down.**

If you have Administrator, run `.\scripts\airgap.ps1 -On` in a second window.
The egress badge turns green within five seconds. If you do not, just turn off
Wi-Fi — same effect, and more visceral.

---

## 1 · Open on the constraint (30 s)

Say the number first, because it is the whole story:

> "This is a 4 GB laptop GPU. 3.9 GB usable. Everything you are about to see —
> reasoning, coding, vision, retrieval, document generation — is running inside
> that, with no network."

Point at the model fabric panel on the left: four models, their VRAM cost, and
which one is resident right now. Point at the egress badge.

---

## 2 · Routing, twice (90 s) — *the headline claim*

Click **Routing A · document**, run it. Then click **Routing B · code**, run it.

Expand the router card on each. Two different classes, two different models,
different confidences, and the swap visible in the trace.

The line that matters:

> "Same box, same prompt field. It picked a coder model for one and a general
> model for the other, and it will tell you exactly why."

If asked how the router works: four stages — deterministic signals, an embedding
classifier over 82 labelled exemplars, a tie-break by a resident LLM, then a
capability solver with hysteresis so it does not thrash models on 4 GB.

---

## 3 · Agentic run, scan → Word (2 min) — *the money shot*

Attach `demo_corpus\UT-Thickness-Report-U2-0417_SCAN.pdf`. It is an image-only
PDF: no text layer, rotated, with scan noise. Prompt:

> Read the attached inspection report, extract the thickness findings, and draft
> a note for approval for the deviation

Narrate while it runs:

- OCR on CPU, mean confidence ~0.98, and **the VLM is deliberately skipped** —
  the trace says so. Deterministic OCR is 40× cheaper; the vision model is for
  pages OCR cannot read.
- Extraction is schema-constrained. The model cannot emit malformed output.
- KB search pulls the SOP clause — that is a *different document* from the scan.
- The provenance gate: every figure traced to a source before anything renders.

Open the `.docx`. It is on a letterhead, classification-marked, with the approval
chain and a SOURCES list. Show that it cites SOP-MEC-014 clause 7.2 and the
180-day validity — **which is not in the scanned report at all**. That is
cross-document synthesis, not summarisation.

---

## 4 · Sandboxed code (90 s)

Click **Routing B · code** again if you skipped it, or use the CML script prompt.

What to point at:

- The generated code, the sandbox verdict, and the script's own assertions.
- `network: denied (socket neutered + self-test)` — the sandbox tries to open a
  connection at startup and refuses to run if it succeeds.
- If it fails a rung, **do not apologise** — say: "that is the escalation ladder;
  it just moved to a stronger model and repaired the code." It is a feature, and
  it usually lands on rung 2 or 3.

---

## 5 · Deterministic calculation (45 s)

Click **Calculation**. Watch the trace print Given / Formula / Substitute /
Result.

> "The model did not do this arithmetic. It identified the quantities and the
> relation; the machine evaluated it. When we let a 4B model write the whole
> calculation it flipped a sign and turned a negative remaining life positive.
> This cannot."

Answer: −1.429 years. Already past retirement thickness.

---

## 6 · Extensibility (30 s)

Open `backend/registry/` in the editor. Show a manifest. Say:

> "Adding a model is this file plus the weights. Nothing else in the system
> changes, because application code asks for a *role*, never a model name. That
> is also how this moves to an H100 — one config file."

Click **reload** in the model fabric panel to show it re-reads at runtime.

---

## 7 · Close on the proof (30 s)

- Egress badge, still green, timestamped seconds ago.
- Click **export audit** — hash-chained log of every prompt, model choice,
  retrieved chunk and file written.
- Header shows the chain verified.

> "The canary deliberately tries to reach the internet every five seconds and
> asserts that it fails. A zero packet count proves nothing. A failing canary
> proves the gap."

---

## If something breaks

| Symptom | Do this |
|---|---|
| First query is slow | It is a cold model load. Say so. Always pre-warm. |
| A coding run fails all rungs | Move on to the calculation demo — it is deterministic and always correct. |
| Ollama not responding | `ollama serve` in a second window, wait 8 s, reload the page. |
| UI blank | Check the server window; `data\server.err` has the traceback. |
| Judge asks for a model you lack | `ollama pull <model>`, write a 12-line YAML in `backend/registry/`, click reload. Offer to do it live — it is the strongest possible answer. |

## Two questions you will be asked

**"Isn't this just Ollama with a UI?"**
Ollama serves one model per request. It has no notion of task class, no VRAM
budgeting, no verification, and produces chat, not files. Show the router card
and the provenance gate.

**"A 4B model can't be trusted with an approval note."**
Correct, unsupervised — and the design assumes that. It is grounded in retrieved
documents, extraction is schema-constrained, calculations are deterministic, and
no figure reaches the document without being traced to a source. The output is
explicitly a draft for human approval, which is how these notes already move.
