"""PCB 异常检测训练任务管理。"""

from __future__ import annotations

import json
import logging
import queue
import re
import shutil
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from dataset_layout import IMAGE_EXTENSIONS, prepare_training_split, validate_ok_ng_dataset


SUPPORTED_MODELS = ("patchcore", "efficient_ad")


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingTask:
    task_id: str
    name: str
    category: str
    data_root: str
    model_type: str
    output_dir: str
    created_at: datetime
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    metrics: dict[str, Any] | None = None
    model_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, Enum):
                data[key] = value.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingTask":
        model_type = data.get("model_type") or _infer_legacy_model_type(data.get("model_config", ""))
        return cls(
            task_id=data["task_id"],
            name=data["name"],
            category=data.get("category") or Path(data["data_root"]).name,
            data_root=data["data_root"],
            model_type=model_type,
            output_dir=data["output_dir"],
            created_at=datetime.fromisoformat(data["created_at"]),
            status=TaskStatus(data.get("status", TaskStatus.PENDING.value)),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            error_message=data.get("error_message"),
            metrics=data.get("metrics"),
            model_path=data.get("model_path"),
        )


def _infer_legacy_model_type(config_path: str) -> str:
    return "efficient_ad" if "efficient" in config_path.lower() else "patchcore"


def _safe_filename(value: str) -> str:
    sanitized = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._")
    return sanitized or "pcb_dataset"


def _model_stem(name: str, category: str, model_type: str) -> str:
    return f"{_safe_filename(name)}_{_safe_filename(category)}_{model_type}"


def _json_value(value: Any) -> Any:
    """把 Torch/NumPy 等指标值转换为 JSON 可序列化对象。"""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return str(value)


class PCBTrainingSystem:
    """串行训练队列；Anomalib 只在真正执行任务时加载。"""

    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        config_path: str | Path | None = None,
        auto_start: bool = True,
    ) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir else Path(__file__).parent.resolve()
        self.config_path = Path(config_path).resolve() if config_path else self.base_dir / "configs" / "training.yaml"
        self.results_dir = self.base_dir / "training_results"
        self.logs_dir = self.base_dir / "logs"
        self.models_dir = self.base_dir / "trained_models"
        for directory in (self.results_dir, self.logs_dir, self.models_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.tasks: dict[str, TrainingTask] = {}
        self.task_queue: queue.Queue[str | None] = queue.Queue()
        self.current_task: TrainingTask | None = None
        self.worker_thread: threading.Thread | None = None
        self.is_running = False
        self._lock = threading.RLock()
        self.logger = self._create_logger()
        self._load_tasks()
        if auto_start:
            self.start_worker()

    @property
    def tasks_file(self) -> Path:
        return self.results_dir / "tasks.json"

    def _create_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"pcb_training.{self.base_dir}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler = logging.FileHandler(
                self.logs_dir / f"training_{datetime.now():%Y%m%d}.log",
                encoding="utf-8",
            )
            stream_handler = logging.StreamHandler()
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
        return logger

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.is_file():
            raise FileNotFoundError(f"训练配置不存在: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        if "data" not in config or "models" not in config:
            raise ValueError(f"训练配置缺少 data 或 models: {self.config_path}")
        return config

    def _generate_task_id(self) -> str:
        return f"pcb_{datetime.now():%Y%m%d_%H%M%S_%f}"

    def add_training_task(
        self,
        name: str,
        data_root: str | Path,
        model_type: str = "patchcore",
        force_retrain: bool = False,
        *,
        category: str | None = None,
    ) -> str:
        model_type = model_type.strip().lower()
        config = self._load_config()
        if model_type not in SUPPORTED_MODELS or model_type not in config["models"]:
            raise ValueError(f"不支持的模型: {model_type}")

        resolved_data_root = Path(data_root).expanduser().resolve()
        validate_ok_ng_dataset(resolved_data_root)
        clean_category = (category or resolved_data_root.name).strip()
        if not clean_category:
            raise ValueError("板型类别不能为空")
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("任务名称不能为空")
        if not force_retrain and self.check_model_exists(clean_name, model_type, clean_category):
            self.logger.info("模型已存在，跳过训练: %s/%s/%s", clean_name, clean_category, model_type)
            return f"existing_model_{_model_stem(clean_name, clean_category, model_type)}"

        with self._lock:
            for task in self.tasks.values():
                if (
                    task.name == clean_name
                    and task.category == clean_category
                    and Path(task.data_root) == resolved_data_root
                    and task.model_type == model_type
                    and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
                ):
                    return task.task_id
            task_id = self._generate_task_id()
            task = TrainingTask(
                task_id=task_id,
                name=clean_name,
                category=clean_category,
                data_root=str(resolved_data_root),
                model_type=model_type,
                output_dir=str(self.results_dir / task_id),
                created_at=datetime.now(),
            )
            self.tasks[task_id] = task
            self._save_tasks()
            if self.is_running:
                self.task_queue.put(task_id)
        self.logger.info("训练任务已入队: %s/%s (%s)", clean_name, clean_category, model_type)
        return task_id

    def start_worker(self) -> None:
        with self._lock:
            if self.worker_thread and self.worker_thread.is_alive():
                return
            self.is_running = True
            self.worker_thread = threading.Thread(
                target=self._worker_loop,
                name="PCBTrainingWorker",
                daemon=True,
            )
            self.worker_thread.start()
            for task in self.tasks.values():
                if task.status == TaskStatus.PENDING:
                    self.task_queue.put(task.task_id)
        self.logger.info("训练工作线程已启动")

    def stop_worker(self) -> None:
        with self._lock:
            if not self.is_running:
                return
            self.is_running = False
            self.task_queue.put(None)
            worker = self.worker_thread
        if worker and worker is not threading.current_thread():
            worker.join(timeout=5)
        self.logger.info("训练工作线程已停止")

    def close(self) -> None:
        """停止后台线程并释放日志文件句柄。"""
        self.stop_worker()
        if self.worker_thread and self.worker_thread.is_alive():
            return
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)

    def _worker_loop(self) -> None:
        while self.is_running:
            try:
                task_id = self.task_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                if task_id is None:
                    return
                with self._lock:
                    task = self.tasks.get(task_id)
                    if not task or task.status != TaskStatus.PENDING:
                        continue
                self._execute_training_task(task)
            except Exception:
                self.logger.exception("训练工作线程发生未处理异常")
            finally:
                self.task_queue.task_done()

    def _execute_training_task(self, task: TrainingTask) -> None:
        with self._lock:
            self.current_task = task
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            task.completed_at = None
            task.error_message = None
            self._save_tasks()
        self.logger.info("开始训练: %s/%s (%s)", task.name, task.category, task.task_id)

        try:
            output_dir = Path(task.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            config = self._load_config()
            data_config = config["data"]
            split_metadata = prepare_training_split(
                task.data_root,
                output_dir / "dataset_split",
                normal_train_ratio=float(data_config.get("normal_train_ratio", 0.8)),
                seed=int(data_config.get("split_seed", 42)),
            )
            with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as file:
                yaml.safe_dump(
                    {
                        "data": data_config,
                        "split": split_metadata,
                        "model": config["models"][task.model_type],
                    },
                    file,
                    allow_unicode=True,
                    sort_keys=False,
                )
            model_path, metrics = self._run_anomalib_training(
                task,
                config,
                output_dir / "dataset_split",
            )
            metrics["split_counts"] = split_metadata["counts"]
            task.model_path = str(model_path)
            task.metrics = metrics
            self._save_trained_model(task)
            task.status = TaskStatus.COMPLETED
            self.logger.info("训练完成: %s", task.task_id)
        except Exception as error:
            task.status = TaskStatus.FAILED
            task.error_message = str(error)
            self.logger.exception("训练失败: %s", task.task_id)
        finally:
            with self._lock:
                task.completed_at = datetime.now()
                self.current_task = None
                self._save_tasks()

    def _run_anomalib_training(
        self,
        task: TrainingTask,
        config: dict[str, Any],
        split_root: Path,
    ) -> tuple[Path, dict[str, Any]]:
        try:
            from anomalib.data import Folder
            from anomalib.engine import Engine
            from anomalib.models import EfficientAd, Patchcore
        except ImportError as error:
            raise RuntimeError("未安装兼容的 anomalib，请先执行 pip install -r requirements.txt") from error

        data_config = config["data"]
        model_config = config["models"][task.model_type]
        image_size = tuple(data_config.get("image_size", (256, 256)))
        datamodule = Folder(
            name=_safe_filename(f"{task.name}_{task.category}"),
            root=str(split_root),
            normal_dir="train/OK",
            abnormal_dir="test/NG",
            normal_test_dir="test/OK",
            task="classification",
            image_size=image_size,
            extensions=tuple(data_config.get("extensions", IMAGE_EXTENSIONS)),
            train_batch_size=int(data_config.get("train_batch_size", 16)),
            eval_batch_size=int(data_config.get("eval_batch_size", 16)),
            num_workers=int(data_config.get("num_workers", 0)),
            test_split_mode="from_dir",
            val_split_mode="from_test",
            val_split_ratio=float(data_config.get("val_split_ratio", 0.5)),
            seed=int(data_config.get("seed", 42)),
        )

        init_args = dict(model_config.get("init_args", {}))
        if task.model_type == "patchcore":
            model = Patchcore(**init_args)
        else:
            model = EfficientAd(**init_args)

        trainer_args = dict(model_config.get("trainer", {}))
        engine = Engine(default_root_dir=task.output_dir, **trainer_args)
        engine.fit(model=model, datamodule=datamodule)
        test_results = engine.test(model=model, datamodule=datamodule)

        checkpoint_path = Path(task.output_dir) / "model.ckpt"
        if getattr(engine, "trainer", None) is None:
            raise RuntimeError("Anomalib 训练完成但未创建 Trainer")
        engine.trainer.save_checkpoint(checkpoint_path)
        if not checkpoint_path.is_file():
            raise RuntimeError("训练完成但未生成模型权重")
        metrics: dict[str, Any] = {"training_completed": True}
        if test_results:
            result = test_results[0] if isinstance(test_results, list) else test_results
            if isinstance(result, dict):
                metrics.update({str(key): _json_value(value) for key, value in result.items()})
        return checkpoint_path, metrics

    def _save_trained_model(self, task: TrainingTask) -> None:
        if not task.model_path or not Path(task.model_path).is_file():
            raise RuntimeError("任务没有有效模型权重")
        stem = _model_stem(task.name, task.category, task.model_type)
        model_path = self.models_dir / f"{stem}.ckpt"
        info_path = self.models_dir / f"{stem}_info.json"
        shutil.copy2(task.model_path, model_path)
        info = {
            "name": task.name,
            "category": task.category,
            "model_type": task.model_type,
            "task_id": task.task_id,
            "created_at": datetime.now().isoformat(),
            "model_path": str(model_path),
            "data_root": task.data_root,
            "metrics": task.metrics,
        }
        with info_path.open("w", encoding="utf-8") as file:
            json.dump(info, file, ensure_ascii=False, indent=2)

    def get_task_info(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self.tasks.get(task_id)
            return task.to_dict() if task else None

    def get_queue_status(self) -> dict[str, Any]:
        with self._lock:
            pending = [task for task in self.tasks.values() if task.status == TaskStatus.PENDING]
            return {
                "queue_size": len(pending),
                "current_task": self.current_task.to_dict() if self.current_task else None,
                "pending_tasks": [task.to_dict() for task in pending],
            }

    def get_all_tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            tasks = sorted(self.tasks.values(), key=lambda task: task.created_at, reverse=True)
            return [task.to_dict() for task in tasks]

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            task = self.tasks.get(task_id)
            if not task or task.status != TaskStatus.PENDING:
                return False
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            self._save_tasks()
            return True

    def check_model_exists(self, name: str, model_type: str, category: str) -> bool:
        stem = _model_stem(name, category, model_type)
        model_path = self.models_dir / f"{stem}.ckpt"
        info_path = self.models_dir / f"{stem}_info.json"
        if not model_path.is_file() or not info_path.is_file():
            return False
        try:
            with info_path.open("r", encoding="utf-8") as file:
                info = json.load(file)
            return (
                info.get("name") == name
                and info.get("category") == category
                and info.get("model_type") == model_type
            )
        except (OSError, json.JSONDecodeError):
            return False

    def get_model_info(
        self,
        name: str,
        model_type: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any] | None:
        matching = [
            info
            for info in self.list_all_models()
            if info.get("name") == name
            and (model_type is None or info.get("model_type") == model_type)
            and (category is None or info.get("category") == category)
        ]
        if model_type and category:
            return matching[0] if matching else None
        return {
            f"{item.get('category', 'legacy')}/{item['model_type']}": item
            for item in matching
        } if matching else None

    def list_all_models(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        for info_path in self.models_dir.glob("*_info.json"):
            try:
                with info_path.open("r", encoding="utf-8") as file:
                    info = json.load(file)
                model_path = Path(info.get("model_path", ""))
                if model_path.is_file():
                    models.append(info)
            except (OSError, json.JSONDecodeError):
                self.logger.warning("忽略损坏的模型信息文件: %s", info_path)
        return sorted(models, key=lambda item: item.get("created_at", ""), reverse=True)

    def _save_tasks(self) -> None:
        data = [task.to_dict() for task in self.tasks.values()]
        temporary = self.tasks_file.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        temporary.replace(self.tasks_file)

    def _load_tasks(self) -> None:
        if not self.tasks_file.is_file():
            return
        try:
            with self.tasks_file.open("r", encoding="utf-8") as file:
                items = json.load(file)
            changed = False
            for item in items:
                task = TrainingTask.from_dict(item)
                if task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now()
                    task.error_message = "进程在训练期间退出，任务未完成"
                    changed = True
                self.tasks[task.task_id] = task
            if changed:
                self._save_tasks()
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            self.logger.error("无法加载历史任务: %s", error)


_system: PCBTrainingSystem | None = None
_system_lock = threading.Lock()


def get_training_system() -> PCBTrainingSystem:
    global _system
    with _system_lock:
        if _system is None:
            _system = PCBTrainingSystem()
        return _system


def stop_training_system() -> None:
    global _system
    with _system_lock:
        system = _system
        _system = None
    if system:
        system.close()


def add_training_task(
    name: str,
    data_root: str | Path,
    model_type: str = "patchcore",
    force_retrain: bool = False,
    *,
    category: str | None = None,
) -> str:
    return get_training_system().add_training_task(
        name,
        data_root,
        model_type,
        force_retrain,
        category=category,
    )


def get_task_info(task_id: str) -> dict[str, Any] | None:
    return get_training_system().get_task_info(task_id)


def get_queue_status() -> dict[str, Any]:
    return get_training_system().get_queue_status()


def get_all_tasks() -> list[dict[str, Any]]:
    return get_training_system().get_all_tasks()


def cancel_task(task_id: str) -> bool:
    return get_training_system().cancel_task(task_id)


def get_model_info(
    name: str,
    model_type: str | None = None,
    category: str | None = None,
) -> dict[str, Any] | None:
    return get_training_system().get_model_info(name, model_type, category)


def list_all_models() -> list[dict[str, Any]]:
    return get_training_system().list_all_models()


def check_model_exists(name: str, model_type: str, category: str) -> bool:
    return get_training_system().check_model_exists(name, model_type, category)


if __name__ == "__main__":
    print(json.dumps(get_training_system().get_queue_status(), ensure_ascii=False, indent=2))
