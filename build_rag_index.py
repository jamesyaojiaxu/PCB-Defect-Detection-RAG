#!/usr/bin/env python
"""构建 PCB 问答系统的本地 Chroma 知识索引。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag import build_rag_index


BASE_DIR = Path(__file__).parent.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="索引 knowledge/ 和 README.md，供PCB知识助手检索")
    parser.add_argument("--project-root", type=Path, default=BASE_DIR)
    parser.add_argument("--config", type=Path, help="RAG配置文件；默认 configs/rag.yaml")
    parser.add_argument("--force", action="store_true", help="忽略指纹缓存并重新生成索引")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metadata = build_rag_index(args.project_root, config_path=args.config, force=args.force)
        print(f"RAG索引已{'复用' if metadata.get('reused') else '生成'}")
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"RAG索引构建失败: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
