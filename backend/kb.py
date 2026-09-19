"""Local knowledge base: ingest + hybrid retrieval.

Ingest : PDF/image → text layer or OCR → page/bbox-anchored chunks → embeddings
Search : BM25 (FTS5) ∥ dense (cosine) → Reciprocal Rank Fusion → top-k

Nothing here touches the network. The Postgres + pgvector schema in the dossier
is a port of these same tables.
"""
from __future__ import annotations
import hashlib
import json
import re
import time
import uuid
from pathlib import Path

import numpy as np

from .config import EMBED_DIM
from .db import db, audit

_OCR = None


def ocr_engine():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


def _embed(texts: list[str]) -> np.ndarray:
    from .router import ROUTER
    return ROUTER.embedder(texts)


# ------------------------------------------------------------------ chunking
def chunk_text(text: str, page: int, max_chars: int = 650, overlap: int = 100):
    """Section-aware-ish chunking: split on blank lines, pack to max_chars."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                out.append(buf)
            buf = (buf[-overlap:] + "\n\n" + p) if buf else p
            if len(buf) > max_chars:
                for i in range(0, len(buf), max_chars):
                    out.append(buf[i:i + max_chars])
                buf = ""
    if buf:
        out.append(buf)
    return [(c, page) for c in out if len(c.strip()) > 40]


# ------------------------------------------------------------------ extraction
def extract_pages(path: Path) -> list[dict]:
    """Returns [{page, text, source, ocr_conf, words:[{text,bbox,conf}]}]."""
    suffix = path.suffix.lower()
    pages = []
    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
        pages.append(_ocr_image(str(path), 1))
        return pages

    if suffix == ".txt" or suffix == ".md":
        return [{"page": 1, "text": path.read_text(encoding="utf-8", errors="replace"),
                 "source": "text", "ocr_conf": 1.0, "words": []}]

    import fitz  # PyMuPDF
    doc = fitz.open(path)
    for i, page in enumerate(doc, start=1):
        txt = page.get_text("text").strip()
        if len(txt) >= 60:
            words = []
            for w in page.get_text("words")[:4000]:
                words.append({"text": w[4], "bbox": [round(w[0]), round(w[1]),
                                                    round(w[2]), round(w[3])],
                              "conf": 1.0})
            pages.append({"page": i, "text": txt, "source": "text-layer",
                          "ocr_conf": 1.0, "words": words})
        else:
            pix = page.get_pixmap(dpi=200)
            tmp = path.parent / f".{path.stem}_p{i}.png"
            pix.save(tmp)
            try:
                pages.append(_ocr_image(str(tmp), i))
            finally:
                tmp.unlink(missing_ok=True)
    doc.close()
    return pages


def _ocr_image(img_path: str, page: int) -> dict:
    res, _ = ocr_engine()(img_path)
    words, confs, lines = [], [], []
    for item in (res or []):
        box, txt, conf = item[0], item[1], float(item[2])
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        words.append({"text": txt, "conf": round(conf, 3),
                      "bbox": [round(min(xs)), round(min(ys)),
                               round(max(xs)), round(max(ys))]})
        confs.append(conf)
        lines.append(txt)
    return {"page": page, "text": "\n".join(lines), "source": "ocr",
            "ocr_conf": round(float(np.mean(confs)), 3) if confs else 0.0,
            "words": words}


# ------------------------------------------------------------------ ingest
def ingest_file(path: Path, *, title: str | None = None, doc_type: str = "document",
                classification: int = 1, plant: str = "", unit: str = "") -> dict:
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    con = db()
    existing = con.execute("SELECT id,title FROM documents WHERE sha256=?",
                           (sha,)).fetchone()
    if existing:
        return {"document_id": existing["id"], "title": existing["title"],
                "deduped": True, "chunks": 0}

    pages = extract_pages(path)
    doc_id = str(uuid.uuid4())
    con.execute(
        "INSERT INTO documents(id,sha256,title,doc_type,plant,unit,revision,"
        "classification,source_path,page_count,ingested_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (doc_id, sha, title or path.stem, doc_type, plant, unit, "R0",
         classification, str(path), len(pages), time.time()))

    rows = []
    for pg in pages:
        for content, page in chunk_text(pg["text"], pg["page"]):
            bbox = _bbox_for(content, pg["words"])
            rows.append((doc_id, f"p{page}", page, json.dumps(bbox),
                         pg["ocr_conf"], content))
    if rows:
        vecs = _embed([r[5] for r in rows])
        for r, v in zip(rows, vecs):
            con.execute(
                "INSERT INTO chunks(document_id,section_path,page,bbox,ocr_conf,"
                "content,embedding) VALUES(?,?,?,?,?,?,?)",
                (*r, v.astype(np.float32).tobytes()))
    con.commit()
    audit("system", "kb.ingest",
          {"document_id": doc_id, "sha256": sha, "pages": len(pages),
           "chunks": len(rows), "source": pages[0]["source"] if pages else "none"})
    return {"document_id": doc_id, "title": title or path.stem,
            "pages": len(pages), "chunks": len(rows), "deduped": False,
            "ocr_conf": pages[0]["ocr_conf"] if pages else None,
            "source": pages[0]["source"] if pages else "none"}


def _bbox_for(content: str, words: list[dict]) -> list[int] | None:
    """Union box of the first few tokens of the chunk — enough to anchor a click."""
    if not words:
        return None
    head = [w for w in re.findall(r"\w+", content[:80])][:6]
    hits = [w for w in words if any(h.lower() in w["text"].lower() for h in head)]
    if not hits:
        return None
    xs0 = min(w["bbox"][0] for w in hits[:12])
    ys0 = min(w["bbox"][1] for w in hits[:12])
    xs1 = max(w["bbox"][2] for w in hits[:12])
    ys1 = max(w["bbox"][3] for w in hits[:12])
    return [xs0, ys0, xs1, ys1]


# ------------------------------------------------------------------ search
def search(query: str, k: int = 6, doc_ids: list[str] | None = None,
           clearance: int = 3) -> list[dict]:
    con = db()
    # -- BM25 lane
    bm, dense = [], []
    fts_q = " OR ".join(re.findall(r"[A-Za-z0-9]{3,}", query)[:12]) or query
    try:
        for r in con.execute(
                "SELECT c.id FROM chunks_fts f JOIN chunks c ON c.id=f.rowid "
                "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT 50",
                (fts_q,)):
            bm.append(r["id"])
    except Exception:
        pass

    # -- dense lane
    rows = con.execute(
        "SELECT c.id, c.embedding FROM chunks c WHERE c.embedding IS NOT NULL"
    ).fetchall()
    if rows:
        mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32)
                        for r in rows])
        if mat.shape[1] == EMBED_DIM or True:
            qv = _embed([query])[0]
            if qv.shape[0] == mat.shape[1]:
                sims = mat @ qv
                order = np.argsort(-sims)[:50]
                dense = [rows[i]["id"] for i in order]

    # -- Reciprocal Rank Fusion
    K = 60
    fused: dict[int, float] = {}
    for lane in (bm, dense):
        for rank, cid in enumerate(lane):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (K + rank + 1)
    if not fused:
        return []
    top = sorted(fused.items(), key=lambda kv: -kv[1])[:k * 3]

    out = []
    for cid, score in top:
        r = con.execute(
            "SELECT c.*, d.title, d.doc_type, d.classification, d.source_path "
            "FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=?",
            (cid,)).fetchone()
        if not r or r["classification"] > clearance:
            continue                                  # clearance filter in SQL
        if doc_ids and r["document_id"] not in doc_ids:
            continue
        out.append({
            "chunk_id": cid, "document_id": r["document_id"], "title": r["title"],
            "page": r["page"], "bbox": json.loads(r["bbox"]) if r["bbox"] else None,
            "ocr_conf": r["ocr_conf"], "content": r["content"],
            "score": round(score, 5),
            "in_bm25": cid in bm, "in_dense": cid in dense,
            "source_path": r["source_path"],
        })
        if len(out) >= k:
            break
    return out


def stats() -> dict:
    con = db()
    d = con.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"]
    c = con.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
    return {"documents": d, "chunks": c}
