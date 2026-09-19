"""Model registry — a directory of capability manifests.

Adding a model to the system is: drop a YAML here, restart (or POST
/api/registry/reload). No code path anywhere else names a model.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml

from .config import REGISTRY_DIR

TASK_CLASSES = [
    "general_qa", "doc_summarise", "structured_extract", "doc_draft",
    "code_write", "code_debug", "calculation", "kb_qa",
    "vision_doc", "vision_drawing",
]


@dataclass
class ModelCard:
    id: str
    serve_name: str
    family: str
    params: str
    license: str
    runtime: str
    tier: str
    modalities: list[str]
    tool_calling: str
    context: dict
    footprint: dict
    skills: dict
    roles: list[str]
    notes: str = ""
    installed: bool = False
    path: str = ""

    @property
    def total_mib(self) -> int:
        return int(self.footprint.get("total_mib", 0))

    @property
    def load_cost_s(self) -> float:
        return float(self.footprint.get("load_cost_s", 5.0))

    @property
    def served_ctx(self) -> int:
        return int(self.context.get("served", 4096))

    def skill(self, task_class: str) -> float:
        return float(self.skills.get(task_class, 0))

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["total_mib"] = self.total_mib
        d["served_ctx"] = self.served_ctx
        return d


@dataclass
class Registry:
    cards: dict[str, ModelCard] = field(default_factory=dict)

    def load(self) -> "Registry":
        self.cards = {}
        for p in sorted(Path(REGISTRY_DIR).glob("*.yaml")):
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
            raw.setdefault("notes", "")
            card = ModelCard(path=str(p), **raw)
            self.cards[card.id] = card
        return self

    def mark_installed(self, serve_names: list[str]):
        have = set(serve_names)
        # ollama reports "qwen3:4b"; tolerate a ":latest" suffix mismatch
        bare = {n.split(":")[0] for n in have}
        for c in self.cards.values():
            c.installed = c.serve_name in have or (
                ":" not in c.serve_name and c.serve_name in bare)

    def by_role(self, role: str) -> list[ModelCard]:
        return [c for c in self.cards.values() if role in c.roles]

    def available(self) -> list[ModelCard]:
        return [c for c in self.cards.values() if c.installed]

    def get(self, mid: str) -> ModelCard | None:
        return self.cards.get(mid)


REGISTRY = Registry().load()
