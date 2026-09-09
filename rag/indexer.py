"""构建持久化 Chroma 知识索引。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import RAGSettings, load_rag_settings
from .loader import chunk_documents, discover_knowledge_files, load_knowledge_file


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("未安装 sentence-transformers，无法使用RAG") from error
    return SentenceTransformer(model_name, device=device)


def _fingerprint(files: list[Path], settings: RAGSettings) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"{settings.embedding_model}|{settings.chunk_size}|{settings.chunk_overlap}|pcb-rag-v1\n".encode("utf-8")
    )
    for path in files:
        stat = path.stat()
        digest.update(f"{path}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def _read_metadata(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def build_rag_index(
    project_root: str | Path,
    *,
    config_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    settings = load_rag_settings(project_root, config_path)
    files = discover_knowledge_files(settings.knowledge_dir, settings.include_files)
    if not files:
        raise FileNotFoundError(
            f"没有找到知识文档；请把 md/txt/pdf/docx/pptx 放入 {settings.knowledge_dir}"
        )
    fingerprint = _fingerprint(files, settings)
    existing = _read_metadata(settings.metadata_path)
    if not force and existing and existing.get("fingerprint") == fingerprint:
        existing["reused"] = True
        return existing

    documents = [
        document
        for path in files
        for document in load_knowledge_file(path, settings.project_root)
    ]
    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise ValueError("知识文档中没有可索引文字")

    try:
        import chromadb
    except ImportError as error:
        raise RuntimeError("未安装 chromadb，无法构建RAG索引") from error
    settings.persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.persist_dir))
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    model = get_embedding_model(settings.embedding_model, settings.embedding_device)
    for start in range(0, len(chunks), 64):
        batch = chunks[start : start + 64]
        embeddings = model.encode(
            [chunk.content for chunk in batch],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        collection.upsert(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.content for chunk in batch],
            metadatas=[chunk.metadata for chunk in batch],
            embeddings=embeddings,
        )

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection_name": settings.collection_name,
        "embedding_model": settings.embedding_model,
        "embedding_device": settings.embedding_device,
        "document_count": len(files),
        "page_count": len(documents),
        "chunk_count": len(chunks),
        "sources": [str(path.relative_to(settings.project_root)) if path.is_relative_to(settings.project_root) else path.name for path in files],
        "fingerprint": fingerprint,
        "reused": False,
    }
    temporary = settings.metadata_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    temporary.replace(settings.metadata_path)
    return metadata


def load_index_metadata(settings: RAGSettings) -> dict[str, Any] | None:
    return _read_metadata(settings.metadata_path)
