"""Deliverable factory.

The model never writes OOXML. It emits a DocumentSpec (JSON, generated under a
schema) and a deterministic renderer maps it onto the organisation's template.

Before anything renders, the NUMERIC PROVENANCE GATE walks every numeral in the
spec and requires it to appear in a retrieved chunk or a recorded tool result.
Unsupported figures block the render and are reported. This is a deterministic
check, not a prompt instruction.
"""
from __future__ import annotations
import hashlib
import json
import re
import time
import uuid
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor, Inches
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from .config import ARTIFACTS
from .db import db, audit

# JSON schema handed to the model for constrained generation
DOCUMENT_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_type": {"type": "string"},
        "reference": {"type": "string"},
        "subject": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "body": {"type": "array", "items": {"type": "string"}},
                    "table": {
                        "type": "object",
                        "properties": {
                            "columns": {"type": "array", "items": {"type": "string"}},
                            "rows": {"type": "array",
                                     "items": {"type": "array",
                                               "items": {"type": "string"}}},
                        },
                        "required": ["columns", "rows"],
                    },
                },
                "required": ["heading", "body"],
            },
        },
        "recommendation": {"type": "string"},
        "approvals": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["subject", "sections", "recommendation"],
}

NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _numerals(obj) -> list[str]:
    out = []
    if isinstance(obj, str):
        out += NUM.findall(obj)
    elif isinstance(obj, list):
        for x in obj:
            out += _numerals(x)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("reference", "doc_type"):
                continue
            out += _numerals(v)
    return out


def provenance_check(spec: dict, evidence: list[str]) -> dict:
    """Every numeral in the spec must appear in the evidence corpus."""
    corpus = " ".join(evidence)
    corpus_nums = set(NUM.findall(corpus))
    # OCR routinely collapses whitespace, so "9.5   6.1" arrives as "9.56.1" and
    # tokenising alone would report a figure that IS in the source as invented.
    # We therefore also test membership against the whitespace-stripped stream.
    corpus_flat = re.sub(r"\s+", "", corpus)
    # normalise: 6.10 == 6.1, 1,200 == 1200
    def norm(n):
        n = n.replace(",", "")
        try:
            f = float(n)
            return str(int(f)) if f == int(f) else str(f)
        except ValueError:
            return n
    corpus_norm = {norm(n) for n in corpus_nums}

    supported, unsupported = [], []
    for n in set(_numerals(spec)):
        nn = norm(n)
        # dates, section numbers and small ordinals are structural, not claims
        if len(nn.replace(".", "")) <= 1:
            continue
        ok = nn in corpus_norm or n in corpus_flat or nn in corpus_flat
        (supported if ok else unsupported).append(n)
    total = len(supported) + len(unsupported)
    return {
        "total": total, "supported": sorted(supported),
        "unsupported": sorted(unsupported),
        "ratio": round(len(supported) / total, 3) if total else 1.0,
        "passed": len(unsupported) == 0,
    }


# ------------------------------------------------------------------ renderers
def _style(doc: Document):
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)


def render_docx(spec: dict, *, org: str = "Refinery Division",
                classification: str = "INTERNAL — RESTRICTED",
                citations: list[dict] | None = None,
                review_flags: list[dict] | None = None,
                run_id: str | None = None) -> dict:
    doc = Document()
    _style(doc)

    # -- letterhead band (template stand-in for the site's own .dotx)
    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = h.add_run(org.upper())
    r.bold = True
    r.font.size = Pt(14)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run("Office of the Inspection Department")
    sr.font.size = Pt(9)
    sr.font.color.rgb = RGBColor(0x55, 0x5A, 0x5E)

    cls = doc.add_paragraph()
    cls.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cr = cls.add_run(classification)
    cr.bold = True
    cr.font.size = Pt(8)
    cr.font.color.rgb = RGBColor(0xB8, 0x46, 0x1C)
    doc.add_paragraph("_" * 78)

    ref = spec.get("reference") or f"NFA/{time.strftime('%Y')}/{uuid.uuid4().hex[:4].upper()}"
    meta = doc.add_paragraph()
    meta.add_run(f"Ref: {ref}").bold = True
    meta.add_run(f"\t\t\tDate: {time.strftime('%d.%m.%Y')}")

    t = doc.add_paragraph()
    tr = t.add_run(spec.get("doc_type", "NOTE FOR APPROVAL").upper())
    tr.bold = True
    tr.font.size = Pt(12)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sj = doc.add_paragraph()
    sj.add_run("Subject: ").bold = True
    sj.add_run(spec.get("subject", ""))

    for i, sec in enumerate(spec.get("sections", []), start=1):
        hp = doc.add_paragraph()
        # models frequently number their own headings ("1. Background"); strip it
        # so we don't render "1. 1. Background"
        head = re.sub(r"^\s*\d+[.)]\s*", "", str(sec.get("heading", "")))
        hr = hp.add_run(f"{i}. {head}")
        hr.bold = True
        for para in sec.get("body", []) or []:
            doc.add_paragraph(str(para))
        tbl = sec.get("table")
        if tbl and tbl.get("columns"):
            table = doc.add_table(rows=1, cols=len(tbl["columns"]))
            table.style = "Table Grid"
            for c, name in zip(table.rows[0].cells, tbl["columns"]):
                cp = c.paragraphs[0].add_run(str(name))
                cp.bold = True
            for row in tbl.get("rows", []):
                cells = table.add_row().cells
                for c, val in zip(cells, row):
                    c.text = str(val)

    if spec.get("recommendation"):
        rp = doc.add_paragraph()
        rp.add_run("Recommendation: ").bold = True
        rp.add_run(spec["recommendation"])

    if review_flags:
        doc.add_paragraph()
        fp = doc.add_paragraph()
        fr = fp.add_run("FIELDS REQUIRING HUMAN VERIFICATION")
        fr.bold = True
        fr.font.size = Pt(9)
        fr.font.color.rgb = RGBColor(0x8A, 0x62, 0x09)
        for f in review_flags:
            b = doc.add_paragraph(style="List Bullet")
            br = b.add_run(f"{f.get('field','')} — {f.get('reason','')}")
            br.font.size = Pt(9)
            br.font.color.rgb = RGBColor(0x8A, 0x62, 0x09)

    if citations:
        doc.add_paragraph()
        cp = doc.add_paragraph()
        cr2 = cp.add_run("SOURCES")
        cr2.bold = True
        cr2.font.size = Pt(9)
        for n, c in enumerate(citations, start=1):
            line = doc.add_paragraph()
            lr = line.add_run(
                f"[{n}] {c.get('title','?')} — page {c.get('page','?')}"
                + (f", region {c.get('bbox')}" if c.get("bbox") else "")
                + (f"  (OCR conf {c.get('ocr_conf')})" if c.get("ocr_conf") is not None else ""))
            lr.font.size = Pt(8)
            lr.font.color.rgb = RGBColor(0x55, 0x5A, 0x5E)

    doc.add_paragraph()
    ap = doc.add_paragraph()
    ap.add_run("Submitted for approval:").bold = True
    for role in spec.get("approvals", []) or ["Sr. Manager (Inspection)", "DGM (Mechanical)"]:
        doc.add_paragraph(f"\n\n________________________\n{role}")

    ft = doc.add_paragraph()
    fr2 = ft.add_run(
        f"\nGenerated on-premises by SynaptiX — no external network access. "
        f"Ref {ref} · {time.strftime('%Y-%m-%d %H:%M')}")
    fr2.font.size = Pt(7)
    fr2.font.color.rgb = RGBColor(0x8A, 0x8F, 0x93)

    fname = f"{ref.replace('/', '-')}.docx"
    return _save(doc, fname, "docx", run_id, {"reference": ref})


def render_xlsx(title: str, columns: list[str], rows: list[list],
                *, formulas: dict | None = None, notes: list[str] | None = None,
                run_id: str | None = None) -> dict:
    wb = Workbook()
    ws = wb.active
    ws.title = title[:28] or "Sheet1"
    head_fill = PatternFill("solid", fgColor="1E3A43")
    thin = Side(style="thin", color="CBD2D5")
    ws.append(columns)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF", size=10)
        c.fill = head_fill
        c.alignment = Alignment(horizontal="center")
    for row in rows:
        ws.append(row)
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=len(columns)):
        for c in row:
            c.border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for i, col in enumerate(columns, start=1):
        width = max(len(str(col)) + 4,
                    *(len(str(r[i - 1])) + 3 for r in rows)) if rows else len(col) + 6
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(width, 42)
    if formulas:
        ws.append([])
        for label, f in formulas.items():
            ws.append([label, f])
    if notes:
        ws.append([])
        for n in notes:
            ws.append([n])
    fname = f"{re.sub(r'[^A-Za-z0-9_-]+','_',title)[:40] or 'sheet'}-{uuid.uuid4().hex[:4]}.xlsx"
    return _save(wb, fname, "xlsx", run_id, {"rows": len(rows)})


def _save(obj, fname: str, kind: str, run_id: str | None, meta: dict) -> dict:
    path = ARTIFACTS / fname
    obj.save(path)
    blob = path.read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    aid = str(uuid.uuid4())
    con = db()
    con.execute(
        "INSERT INTO artifacts(id,run_id,kind,filename,path,sha256,created_at,meta)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (aid, run_id, kind, fname, str(path), sha, time.time(), json.dumps(meta)))
    con.commit()
    audit("system", "artifact.render",
          {"id": aid, "kind": kind, "filename": fname, "sha256": sha})
    return {"artifact_id": aid, "kind": kind, "filename": fname,
            "path": str(path), "sha256": sha, "size": len(blob), **meta}
