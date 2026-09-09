"""通用多板型异常检测训练队列的 Web 管理界面。"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from dataset_layout import DatasetLayoutError, dataset_summary, discover_category_datasets
from pcb_training_system import get_training_system
from rag import RAGNotReadyError, get_rag_assistant


BASE_DIR = Path(__file__).parent.resolve()


def create_app(data_root: str | Path | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        PCB_DATA_ROOT=str(Path(data_root).resolve() if data_root else BASE_DIR / "data" / "pcb_categories"),
        JSON_AS_ASCII=False,
    )

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"success": True, "data": {"status": "ok", "mode": "multi-category"}})

    @app.get("/api/dataset")
    def dataset_status():
        try:
            return jsonify({"success": True, "data": dataset_summary(app.config["PCB_DATA_ROOT"])})
        except (DatasetLayoutError, OSError) as error:
            return jsonify({"success": False, "error": str(error)}), 400

    @app.post("/api/train")
    def create_training_tasks():
        payload = request.get_json(silent=True) or {}
        try:
            name = str(payload.get("name", "pcb_dataset")).strip()
            model_type = str(payload.get("model_type", "patchcore")).strip().lower()
            force_retrain = bool(payload.get("force_retrain", False))
            categories = payload.get("categories")
            if not name:
                raise ValueError("系统名称不能为空")
            if categories is not None and not isinstance(categories, list):
                raise ValueError("categories 必须是数组")

            datasets = discover_category_datasets(app.config["PCB_DATA_ROOT"], categories)
            task_ids = {
                dataset.category: get_training_system().add_training_task(
                    name=name,
                    data_root=dataset.root,
                    model_type=model_type,
                    force_retrain=force_retrain,
                    category=dataset.category,
                )
                for dataset in datasets
            }
            return jsonify({"success": True, "data": {"task_ids": task_ids}})
        except (DatasetLayoutError, OSError, TypeError, ValueError) as error:
            return jsonify({"success": False, "error": str(error)}), 400

    @app.get("/api/tasks")
    def tasks():
        return jsonify({"success": True, "data": get_training_system().get_all_tasks()})

    @app.get("/api/tasks/<task_id>")
    def task_detail(task_id: str):
        task = get_training_system().get_task_info(task_id)
        if task is None:
            return jsonify({"success": False, "error": "任务不存在"}), 404
        return jsonify({"success": True, "data": task})

    @app.post("/api/tasks/<task_id>/cancel")
    def cancel_task(task_id: str):
        if not get_training_system().cancel_task(task_id):
            return jsonify({"success": False, "error": "任务不存在或已开始执行"}), 409
        return jsonify({"success": True})

    @app.get("/api/queue")
    def queue_status():
        return jsonify({"success": True, "data": get_training_system().get_queue_status()})

    @app.get("/api/models")
    def models():
        return jsonify({"success": True, "data": get_training_system().list_all_models()})

    @app.get("/api/chat/status")
    def chat_status():
        try:
            return jsonify({"success": True, "data": get_rag_assistant(BASE_DIR).status()})
        except (OSError, RuntimeError, ValueError) as error:
            return jsonify({"success": False, "error": str(error)}), 503

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        question = payload.get("message", "")
        history = payload.get("history", [])
        if not isinstance(question, str):
            return jsonify({"success": False, "error": "message 必须是字符串"}), 400
        if not isinstance(history, list):
            return jsonify({"success": False, "error": "history 必须是数组"}), 400
        try:
            result = get_rag_assistant(BASE_DIR).ask(question, history)
            return jsonify({"success": True, "data": result})
        except RAGNotReadyError as error:
            return jsonify({"success": False, "error": str(error)}), 503
        except ValueError as error:
            return jsonify({"success": False, "error": str(error)}), 400
        except (OSError, RuntimeError) as error:
            app.logger.exception("PCB知识助手调用失败")
            return jsonify({"success": False, "error": str(error)}), 502

    @app.get("/api/models/<path:name>/<category>/<model_type>/download")
    def download_model(name: str, category: str, model_type: str):
        model_info = get_training_system().get_model_info(name, model_type, category)
        if not model_info:
            return jsonify({"success": False, "error": "模型不存在"}), 404
        model_path = Path(model_info["model_path"])
        if not model_path.is_file():
            return jsonify({"success": False, "error": "模型权重文件不存在"}), 404
        return send_file(model_path, as_attachment=True, download_name=f"{model_path.stem}.ckpt")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
