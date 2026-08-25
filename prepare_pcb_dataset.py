#!/usr/bin/env python
"""独立把 Kaggle pcb-dataset 整理为 ``类别/OK、类别/NG``。

本脚本不依赖项目其他 Python 文件，可以单独复制使用；它只整理数据，不划分训练集。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
EXTENSION_PRIORITY = {extension: index for index, extension in enumerate(IMAGE_EXTENSIONS)}
CATEGORY_PATTERN = re.compile(r"(?:^|_)(C\d+)(?:_|$)", re.IGNORECASE)
METADATA_FILENAME = "dataset_metadata.json"
MANIFEST_FILENAME = "manifest.csv"
BASE_DIR = Path(__file__).parent.resolve()


class DatasetError(ValueError):
    """原始数据结构或文件名不符合预期。"""


@dataclass(frozen=True)
class SourceImage:
    path: Path
    relative_path: str
    category: str
    label: str
    group: str
    defect_type: str | None = None


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _deduplicated_images(root: Path) -> list[Path]:
    selected: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not _is_image(path):
            continue
        key = path.relative_to(root).with_suffix("").as_posix().casefold()
        current = selected.get(key)
        if current is None or EXTENSION_PRIORITY[path.suffix.lower()] < EXTENSION_PRIORITY[current.suffix.lower()]:
            selected[key] = path
    return sorted(selected.values(), key=lambda path: path.relative_to(root).as_posix())


def _category(path: Path) -> str:
    match = CATEGORY_PATTERN.search(path.stem)
    if not match:
        raise DatasetError(f"文件名中没有找到类别编号（例如 C1）: {path}")
    return match.group(1).upper()


def load_source_images(root: str | Path) -> list[SourceImage]:
    source = Path(root).expanduser().resolve()
    normal_root = source / "OK"
    abnormal_root = source / "NG"
    if not normal_root.is_dir() or not abnormal_root.is_dir():
        raise DatasetError(f"原始目录必须包含 OK 和 NG: {source}")

    images: list[SourceImage] = []
    for path in _deduplicated_images(normal_root):
        relative = path.relative_to(normal_root)
        images.append(
            SourceImage(
                path=path,
                relative_path=path.relative_to(source).as_posix(),
                category=_category(path),
                label="OK",
                group=relative.parts[0] if len(relative.parts) > 1 else path.stem,
            )
        )

    for path in _deduplicated_images(abnormal_root):
        relative = path.relative_to(abnormal_root)
        if len(relative.parts) < 2:
            raise DatasetError(f"NG图片没有缺陷类别目录: {relative}")
        defect_type = relative.parts[0]
        group_name = relative.parts[1] if len(relative.parts) > 2 else path.stem
        images.append(
            SourceImage(
                path=path,
                relative_path=path.relative_to(source).as_posix(),
                category=_category(path),
                label="NG",
                group=f"{defect_type}/{group_name}",
                defect_type=defect_type,
            )
        )

    if not images or not any(image.label == "OK" for image in images) or not any(image.label == "NG" for image in images):
        raise DatasetError("原始数据没有可用的OK或NG图片")
    return images


def analyze_dataset(root: str | Path) -> dict:
    source = Path(root).expanduser().resolve()
    images = load_source_images(source)
    raw_count = sum(1 for folder in (source / "OK", source / "NG") for path in folder.rglob("*") if _is_image(path))
    categories = sorted({image.category for image in images}, key=str.casefold)
    return {
        "root": str(source),
        "categories": categories,
        "counts_by_category": {
            category: {
                "OK": sum(image.category == category and image.label == "OK" for image in images),
                "NG": sum(image.category == category and image.label == "NG" for image in images),
            }
            for category in categories
        },
        "defect_counts": dict(sorted(Counter(image.defect_type for image in images if image.defect_type).items())),
        "ignored_duplicates": raw_count - len(images),
    }


def _fingerprint(images: list[SourceImage]) -> str:
    digest = hashlib.sha256()
    for image in images:
        stat = image.path.stat()
        digest.update(image.relative_path.encode("utf-8"))
        digest.update(f"|{stat.st_size}|{stat.st_mtime_ns}\n".encode("ascii"))
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path, hard_links: bool) -> str:
    if hard_links:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


def _safe_name(image: SourceImage, used: set[str]) -> str:
    name = image.path.name
    if name.casefold() not in used:
        used.add(name.casefold())
        return name
    prefix = image.group.replace("/", "_").replace("\\", "_")
    name = f"{prefix}_{name}"
    if name.casefold() in used:
        name = f"{uuid.uuid4().hex[:8]}_{name}"
    used.add(name.casefold())
    return name


def _replace_directory(staging: Path, target: Path) -> None:
    backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
    had_target = target.exists()
    if had_target:
        target.replace(backup)
    try:
        staging.replace(target)
    except Exception:
        if had_target and backup.exists() and not target.exists():
            backup.replace(target)
        raise
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def prepare_dataset(
    source_root: str | Path,
    target_root: str | Path,
    *,
    force: bool = False,
    use_hard_links: bool = True,
) -> dict:
    """整理为任意类别名下的平面 OK/NG 图片目录。"""
    source = Path(source_root).expanduser().resolve()
    target = Path(target_root).expanduser().resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("源目录和目标目录不能相同或互相包含")

    images = load_source_images(source)
    source_fingerprint = _fingerprint(images)
    preparation_fingerprint = hashlib.sha256(
        f"{source_fingerprint}|category-ok-ng-v1".encode("utf-8")
    ).hexdigest()
    metadata_path = target / METADATA_FILENAME
    if not force and metadata_path.is_file():
        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                existing = json.load(file)
            if existing.get("preparation_fingerprint") == preparation_fingerprint:
                actual = {
                    category: {
                        label: sum(1 for path in (target / category / label).iterdir() if _is_image(path))
                        for label in ("OK", "NG")
                    }
                    for category in existing["categories"]
                }
                if actual == existing.get("counts_by_category"):
                    existing["reused"] = True
                    return existing
        except (OSError, KeyError, json.JSONDecodeError):
            pass

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    used_names: dict[tuple[str, str], set[str]] = defaultdict(set)
    link_modes = Counter()
    manifest_rows: list[dict[str, str]] = []
    try:
        for image in images:
            destination_dir = staging / image.category / image.label
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / _safe_name(image, used_names[(image.category, image.label)])
            link_modes[_link_or_copy(image.path, destination, use_hard_links)] += 1
            manifest_rows.append(
                {
                    "category": image.category,
                    "label": image.label,
                    "defect_type": image.defect_type or "",
                    "group": image.group,
                    "source": image.relative_path,
                    "target": destination.relative_to(staging).as_posix(),
                }
            )

        with (staging / MANIFEST_FILENAME).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=("category", "label", "defect_type", "group", "source", "target"),
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

        analysis = analyze_dataset(source)
        metadata = {
            "schema_version": 1,
            "layout": "category/OK-or-NG",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source),
            "target_root": str(target),
            "categories": analysis["categories"],
            "counts_by_category": analysis["counts_by_category"],
            "defect_counts": analysis["defect_counts"],
            "ignored_duplicates": analysis["ignored_duplicates"],
            "source_fingerprint": source_fingerprint,
            "preparation_fingerprint": preparation_fingerprint,
            "link_modes": dict(sorted(link_modes.items())),
            "manifest": MANIFEST_FILENAME,
            "reused": False,
        }
        with (staging / METADATA_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        _replace_directory(staging, target)
        return metadata
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="把 pcb-dataset 整理成 类别/OK、类别/NG")
    parser.add_argument("--source", type=Path, default=BASE_DIR / "pcb-dataset", help="Kaggle原始数据目录")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "data" / "pcb_categories", help="整理结果目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重新生成")
    parser.add_argument("--copy-files", action="store_true", help="复制图片，不使用硬链接")
    parser.add_argument("--analyze-only", action="store_true", help="只分析，不生成目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        analysis = analyze_dataset(args.source)
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
        if args.analyze_only:
            return 0
        metadata = prepare_dataset(
            args.source,
            args.output,
            force=args.force,
            use_hard_links=not args.copy_files,
        )
        print(f"已{'复用' if metadata.get('reused') else '生成'}: {Path(args.output).resolve()}")
        print(json.dumps(metadata["counts_by_category"], ensure_ascii=False, indent=2))
        return 0
    except (DatasetError, OSError, ValueError) as error:
        print(f"数据整理失败: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
