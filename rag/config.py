"""RAG 配置读取。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RAGSettings:
    project_root: Path
    knowledge_dir: Path
    include_files: tuple[Path, ...]
    persist_dir: Path
    collection_name: str
    embedding_model: str
    embedding_device: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_history_messages: int

    @property
    def metadata_path(self) -> Path:
        return self.persist_dir / "index_metadata.json"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def load_rag_settings(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> RAGSettings:
    root = Path(project_root).expanduser().resolve()
    path = _resolve(root, config_path or "configs/rag.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"RAG配置不存在: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    chunk_size = int(data.get("chunk_size", 900))
    overlap = int(data.get("chunk_overlap", 150))
    top_k = int(data.get("top_k", 4))
    if chunk_size < 200:
        raise ValueError("RAG chunk_size 不能小于200")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("RAG chunk_overlap 必须大于等于0且小于chunk_size")
    if not 1 <= top_k <= 10:
        raise ValueError("RAG top_k 必须在1到10之间")

    collection_name = str(data.get("collection_name", "pcb_knowledge")).strip()
    if not collection_name:
        raise ValueError("RAG collection_name 不能为空")
    return RAGSettings(
        project_root=root,
        knowledge_dir=_resolve(root, data.get("knowledge_dir", "knowledge")),
        include_files=tuple(_resolve(root, item) for item in data.get("include_files", ["README.md"])),
        persist_dir=_resolve(root, data.get("persist_dir", "data/rag_index")),
        collection_name=collection_name,
        embedding_model=str(data.get("embedding_model", "BAAI/bge-small-zh-v1.5")).strip(),
        embedding_device=str(data.get("embedding_device", "cpu")).strip(),
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        top_k=top_k,
        max_history_messages=max(0, int(data.get("max_history_messages", 6))),
    )
