"""SynaptiX — central config. Everything here is local-only by construction."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
ARTIFACTS = DATA / "artifacts"
KB_DIR = DATA / "kb"
REGISTRY_DIR = ROOT / "backend" / "registry"
WEB_DIR = ROOT / "web"
DEMO_DIR = ROOT / "demo_corpus"
DB_PATH = DATA / "kws.db"

for d in (DATA, UPLOADS, ARTIFACTS, KB_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- inference backend (loopback only; never a remote host) ---
OLLAMA_HOST = os.environ.get("KWS_OLLAMA", "http://127.0.0.1:11434")

# --- offline enforcement: set before any library import that might phone home ---
for k, v in {
    "HF_HUB_OFFLINE": "0",          # flipped to 1 by run.ps1 after first warm-up
    "TRANSFORMERS_OFFLINE": "0",
    "DO_NOT_TRACK": "1",
    "ANONYMIZED_TELEMETRY": "False",
    "GRADIO_ANALYTICS_ENABLED": "False",
}.items():
    os.environ.setdefault(k, v)

# Hosts the app is permitted to talk to. Anything else raises EgressBlocked.
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}

EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-d, ONNX, CPU, ~130 MB
EMBED_DIM = 384

# Agent governor budgets
MAX_STEPS = 12
MAX_TOOL_CALLS = 40
MAX_WALL_CLOCK_S = 900
