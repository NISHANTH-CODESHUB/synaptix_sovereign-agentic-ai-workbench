"""SQLite state layer.

Prototype substitute for the PostgreSQL 17 + pgvector schema in the design dossier.
The table shapes deliberately mirror the Postgres DDL so the pilot migration is a
port, not a redesign: documents / chunks / runs / steps / router_decisions /
artifacts / audit_log.
"""
import sqlite3
import json
import time
import hashlib
from contextlib import contextmanager
from .config import DB_PATH

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, sha256 TEXT UNIQUE, title TEXT, doc_type TEXT,
  plant TEXT, unit TEXT, revision TEXT, classification INTEGER DEFAULT 1,
  source_path TEXT, page_count INTEGER, ingested_at REAL
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id TEXT REFERENCES documents(id),
  section_path TEXT, page INTEGER, bbox TEXT, ocr_conf REAL,
  content TEXT, embedding BLOB
);

-- BM25 lane (Postgres tsvector/GIN equivalent)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  content, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
END;

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, goal TEXT, status TEXT, created_at REAL,
  finished_at REAL, plan TEXT, error TEXT
);

CREATE TABLE IF NOT EXISTS steps (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT REFERENCES runs(id),
  idx INTEGER, name TEXT, tool TEXT, status TEXT, model_id TEXT,
  attempts INTEGER DEFAULT 0, input TEXT, output TEXT,
  verify_ok INTEGER, verify_note TEXT, started_at REAL, finished_at REAL
);

CREATE TABLE IF NOT EXISTS router_decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL,
  request TEXT, stage INTEGER, task_class TEXT, confidence REAL,
  margin REAL, chosen_model TEXT, runner_up TEXT, score_gap REAL,
  resident_reused INTEGER, latency_ms REAL, features TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY, run_id TEXT, kind TEXT, filename TEXT,
  path TEXT, sha256 TEXT, created_at REAL, meta TEXT
);

-- append-only, hash-chained (tamper-evident)
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, actor TEXT, action TEXT,
  detail TEXT, prev_hash TEXT, hash TEXT
);
"""


def connect():
    con = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


_con = None


def db():
    global _con
    if _con is None:
        _con = connect()
        _con.executescript(SCHEMA)
        _con.commit()
    return _con


@contextmanager
def tx():
    con = db()
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise


def audit(actor: str, action: str, detail) -> str:
    """Append one hash-chained audit record. Returns the new head hash."""
    con = db()
    row = con.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
    prev = row["hash"] if row else "GENESIS"
    ts = time.time()
    payload = json.dumps(detail, default=str, sort_keys=True)
    h = hashlib.sha256(f"{prev}|{ts}|{actor}|{action}|{payload}".encode()).hexdigest()
    con.execute(
        "INSERT INTO audit_log(ts,actor,action,detail,prev_hash,hash) VALUES(?,?,?,?,?,?)",
        (ts, actor, action, payload, prev, h),
    )
    con.commit()
    return h


def verify_audit_chain():
    """Recompute the chain. Returns (ok, broken_at_id | None, count)."""
    con = db()
    prev = "GENESIS"
    n = 0
    for r in con.execute("SELECT * FROM audit_log ORDER BY id"):
        h = hashlib.sha256(
            f"{prev}|{r['ts']}|{r['actor']}|{r['action']}|{r['detail']}".encode()
        ).hexdigest()
        if h != r["hash"] or r["prev_hash"] != prev:
            return False, r["id"], n
        prev = h
        n += 1
    return True, None, n
