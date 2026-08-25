import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pcb_training_system import PCBTrainingSystem, TaskStatus


def make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(path)


class TrainingQueueTests(unittest.TestCase):
    def test_add_and_cancel_without_loading_anomalib(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data_root = base / "data"
            for relative in ("OK/a.jpg", "OK/b.jpg", "NG/c.jpg"):
                make_image(data_root / relative)
            config = base / "training.yaml"
            config.write_text(
                "data: {}\nmodels:\n  patchcore:\n    init_args: {}\n    trainer: {}\n",
                encoding="utf-8",
            )

            system = PCBTrainingSystem(base, config_path=config, auto_start=False)
            task_id = system.add_training_task("unit_test", data_root, category="board_alpha")
            self.assertEqual(system.tasks[task_id].status, TaskStatus.PENDING)
            self.assertEqual(system.tasks[task_id].category, "board_alpha")
            self.assertTrue(system.cancel_task(task_id))
            self.assertEqual(system.tasks[task_id].status, TaskStatus.CANCELLED)
            self.assertFalse(system.cancel_task(task_id))
            system.close()


if __name__ == "__main__":
    unittest.main()
