"""查询 PCB 知识库并返回带来源的证据。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_rag_settings
from .indexer import get_embedding_model, load_index_metadata


class RAGNotReadyError(RuntimeError):
    """知识索引尚未构建。"""


def search_knowledge(
    query: str,
    *,
    project_root: str | Path,
    top_k: int | None = None,
) -> dict[str, Any]:
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("问题不能为空")
    settings = load_rag_settings(project_root)
    metadata = load_index_metadata(settings)
    if not metadata:
        raise RAGNotReadyError("知识库尚未建立，请先运行 python build_rag_index.py")
    count = int(top_k or settings.top_k)
    if not 1 <= count <= 10:
        raise ValueError("top_k必须在1到10之间")

    try:
        import chromadb
    except ImportError as error:
        raise RuntimeError("未安装 chromadb，无法查询RAG知识库") from error
    model = get_embedding_model(settings.embedding_model, settings.embedding_device)
    query_embedding = model.encode(
        [clean_query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    client = chromadb.PersistentClient(path=str(settings.persist_dir))
    try:
        collection = client.get_collection(settings.collection_name)
    except Exception as error:
        raise RAGNotReadyError("知识索引不存在或已损坏，请重新运行 python build_rag_index.py --force") from error
    raw = collection.query(
        query_embeddings=query_embedding,
        n_results=min(count, int(metadata.get("chunk_count", count))),
        include=["documents", "metadatas", "distances"],
    )
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    results = []
    for rank, (content, item_metadata, distance) in enumerate(
        zip(documents, metadatas, distances, strict=False),
        start=1,
    ):
        item_metadata = item_metadata or {}
        results.append(
            {
                "rank": rank,
                "content": content,
                "source": item_metadata.get("source", "未知来源"),
                "page": int(item_metadata.get("page") or 0) or None,
                "citation": item_metadata.get("citation", "[未知来源]"),
                "distance": round(float(distance), 6),
            }
        )
    return {"query": clean_query, "found": bool(results), "count": len(results), "results": results}
