"""通用板型 OK/NG 数据发现、验证和自动训练集划分。"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
METADATA_FILENAME = "dataset_metadata.json"
SPLIT_METADATA_FILENAME = "split_metadata.json"


class DatasetLayoutError(ValueError):
    """板型数据目录不符合约定。"""


@dataclass(frozen=True)
class CategoryDataset:
    """一种板型对应的一套 OK/NG 图片。"""

    category: str
    root: str
    counts: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _images(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.iterdir() if _is_image(path)),
        key=lambda path: path.name.casefold(),
    )


def is_ok_ng_dataset(root: str | Path) -> bool:
    """判断目录是否直接包含 ``OK`` 和 ``NG``。"""
    path = Path(root).expanduser().resolve()
    return (path / "OK").is_dir() and (path / "NG").is_dir()


def validate_ok_ng_dataset(root: str | Path) -> dict[str, int]:
    """验证单类别数据；OK至少两张，NG至少一张。"""
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise DatasetLayoutError(f"数据目录不存在: {path}")
    for label in ("OK", "NG"):
        if not (path / label).is_dir():
            raise DatasetLayoutError(f"数据目录缺少: {path / label}")

    counts = {"OK": len(_images(path / "OK")), "NG": len(_images(path / "NG"))}
    if counts["OK"] < 2:
        raise DatasetLayoutError(f"{path / 'OK'} 至少需要两张图片，才能自动划分训练和测试")
    if counts["NG"] < 1:
        raise DatasetLayoutError(f"{path / 'NG'} 至少需要一张图片")
    return counts


def discover_category_datasets(
    root: str | Path,
    categories: Iterable[str] | None = None,
) -> list[CategoryDataset]:
    """动态发现一个单类别目录或根目录下的任意类别。

    系统不预设 C1、C2 等名称；只要子目录中直接存在 OK/NG 就会被识别。
    """
    data_root = Path(root).expanduser().resolve()
    if not data_root.is_dir():
        raise DatasetLayoutError(f"训练数据根目录不存在: {data_root}")

    requested = [str(category).strip() for category in categories or [] if str(category).strip()]
    requested_keys = [category.casefold() for category in requested]
    if len(set(requested_keys)) != len(requested_keys):
        raise DatasetLayoutError("类别列表中存在重复项")

    if is_ok_ng_dataset(data_root):
        category = data_root.name
        if requested_keys and category.casefold() not in requested_keys:
            raise DatasetLayoutError(f"直接数据集类别 {category} 与指定类别不一致")
        return [CategoryDataset(category, str(data_root), validate_ok_ng_dataset(data_root))]

    available = {
        child.name.casefold(): child
        for child in data_root.iterdir()
        if child.is_dir() and is_ok_ng_dataset(child)
    }
    selected_keys = requested_keys or sorted(available)
    if not selected_keys:
        raise DatasetLayoutError(f"{data_root} 下没有找到包含 OK/NG 的板型类别目录")

    missing = [requested[index] for index, key in enumerate(requested_keys) if key not in available]
    if missing:
        raise DatasetLayoutError(f"缺少板型类别: {', '.join(missing)}")

    return [
        CategoryDataset(
            category=available[key].name,
            root=str(available[key].resolve()),
            counts=validate_ok_ng_dataset(available[key]),
        )
        for key in selected_keys
    ]


def _fingerprint(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(f"|{stat.st_size}|{stat.st_mtime_ns}\n".encode("ascii"))
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path, use_hard_links: bool) -> str:
    if use_hard_links:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


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


def prepare_training_split(
    source_root: str | Path,
    target_root: str | Path,
    *,
    normal_train_ratio: float = 0.8,
    seed: int = 42,
    use_hard_links: bool = True,
) -> dict:
    """训练任务内部把 ``OK/NG`` 自动变成 Anomalib 的 train/test 结构。"""
    if not 0 < normal_train_ratio < 1:
        raise ValueError("normal_train_ratio 必须在 0 和 1 之间")

    source = Path(source_root).expanduser().resolve()
    target = Path(target_root).expanduser().resolve()
    validate_ok_ng_dataset(source)
    normal = _images(source / "OK")
    abnormal = _images(source / "NG")
    source_fingerprint = _fingerprint([*normal, *abnormal], source)
    split_fingerprint = hashlib.sha256(
        f"{source_fingerprint}|{normal_train_ratio}|{seed}|generic-ok-ng-v1".encode("utf-8")
    ).hexdigest()

    metadata_path = target / SPLIT_METADATA_FILENAME
    if metadata_path.is_file():
        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                existing = json.load(file)
            if existing.get("split_fingerprint") == split_fingerprint:
                expected = existing["counts"]
                actual = {
                    "train_OK": len(_images(target / "train" / "OK")),
                    "test_OK": len(_images(target / "test" / "OK")),
                    "test_NG": len(_images(target / "test" / "NG")),
                }
                if actual == expected:
                    existing["reused"] = True
                    return existing
        except (OSError, KeyError, json.JSONDecodeError):
            pass

    shuffled_normal = list(normal)
    random.Random(seed).shuffle(shuffled_normal)
    train_count = round(len(shuffled_normal) * normal_train_ratio)
    train_count = min(max(train_count, 1), len(shuffled_normal) - 1)
    train_normal = shuffled_normal[:train_count]
    test_normal = shuffled_normal[train_count:]

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    link_modes: dict[str, int] = {}
    try:
        for subset, label, paths in (
            ("train", "OK", train_normal),
            ("test", "OK", test_normal),
            ("test", "NG", abnormal),
        ):
            destination_dir = staging / subset / label
            destination_dir.mkdir(parents=True, exist_ok=True)
            for path in paths:
                mode = _link_or_copy(path, destination_dir / path.name, use_hard_links)
                link_modes[mode] = link_modes.get(mode, 0) + 1

        metadata = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source),
            "target_root": str(target),
            "normal_train_ratio": normal_train_ratio,
            "seed": seed,
            "source_fingerprint": source_fingerprint,
            "split_fingerprint": split_fingerprint,
            "counts": {
                "train_OK": len(train_normal),
                "test_OK": len(test_normal),
                "test_NG": len(abnormal),
            },
            "link_modes": link_modes,
            "reused": False,
        }
        with (staging / SPLIT_METADATA_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        _replace_directory(staging, target)
        return metadata
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def load_dataset_metadata(root: str | Path) -> dict | None:
    path = Path(root).expanduser().resolve() / METADATA_FILENAME
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def dataset_summary(root: str | Path) -> dict:
    data_root = Path(root).expanduser().resolve()
    datasets = discover_category_datasets(data_root)
    return {
        "root": str(data_root),
        "category_count": len(datasets),
        "categories": [dataset.to_dict() for dataset in datasets],
        "metadata": load_dataset_metadata(data_root),
    }
