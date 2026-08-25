#!/usr/bin/env python
"""通用多板型 PCB 异常检测训练系统入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dataset_layout import DatasetLayoutError, discover_category_datasets

BASE_DIR = Path(__file__).parent.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="动态发现任意板型的 OK/NG 数据并训练独立异常检测模型")
    parser.add_argument(
        "--data-root",
        "--prepared-root",
        dest="data_root",
        type=Path,
        default=BASE_DIR / "data" / "pcb_categories",
        help="包含任意板型类别及其 OK/NG 图片的数据根目录",
    )
    parser.add_argument("--categories", nargs="+", help="只训练指定板型类别，例如 --categories C1 C3")
    parser.add_argument("--name", default="pcb_dataset", help="多模型系统名称")
    parser.add_argument("--model", choices=("patchcore", "efficient_ad"), default="patchcore")
    parser.add_argument("--force-retrain", action="store_true", help="已有模型时仍重新训练")
    parser.add_argument("--validate-only", action="store_true", help="只验证 OK/NG 数据，不训练、不启动Web")
    parser.add_argument("--no-auto-train", action="store_true", help="启动Web，但不自动加入训练任务")
    parser.add_argument("--host", default="127.0.0.1", help="Web监听地址")
    parser.add_argument("--port", type=int, default=5000, help="Web监听端口")
    return parser


def _print_datasets(datasets) -> None:
    print("板型类别 OK/NG 数据")
    for dataset in datasets:
        counts = dataset.counts
        print(f"  {dataset.category}: OK={counts['OK']}, NG={counts['NG']}")
        print(f"    {dataset.root}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        datasets = discover_category_datasets(args.data_root, args.categories)
        _print_datasets(datasets)
    except (DatasetLayoutError, OSError, ValueError) as error:
        print(f"训练数据验证失败: {error}", file=sys.stderr)
        print("请先运行 prepare_pcb_dataset.py，或提供直接包含 OK、NG 的类别目录。", file=sys.stderr)
        return 1

    if args.validate_only:
        return 0

    from pcb_training_system import add_training_task, stop_training_system
    from web_interface import create_app

    try:
        if not args.no_auto_train:
            for dataset in datasets:
                task_id = add_training_task(
                    args.name,
                    dataset.root,
                    args.model,
                    args.force_retrain,
                    category=dataset.category,
                )
                print(f"  {dataset.category} 训练任务: {task_id}")
        app = create_app(args.data_root)
        print(f"多板型管理界面: http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        stop_training_system()


if __name__ == "__main__":
    raise SystemExit(main())
