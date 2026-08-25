import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dataset_layout import discover_category_datasets, prepare_training_split, validate_ok_ng_dataset
from prepare_pcb_dataset import analyze_dataset, prepare_dataset


def make_image(path: Path, color=(40, 80, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


class GenericCategoryDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "pcb-dataset"
        self.categories = self.root / "pcb_categories"

        for group in ("S0001", "S0002", "S0003"):
            for category in ("C1", "C2"):
                image = self.source / "OK" / group / f"pcb_{group}_OK_{category}.jpg"
                make_image(image)
                make_image(image.with_suffix(".png"))
        for defect, count in (("HS", 2), ("QS", 1)):
            for index in range(count):
                for category in ("C1", "C2"):
                    image = self.source / "NG" / defect / f"S{index:04d}" / f"pcb_NG_{defect}_{category}_{index}.jpg"
                    make_image(image, color=(150, 30, 20))
                    make_image(image.with_suffix(".png"), color=(150, 30, 20))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_standalone_script_outputs_only_category_ok_ng(self) -> None:
        analysis = analyze_dataset(self.source)
        self.assertEqual(analysis["categories"], ["C1", "C2"])
        self.assertEqual(analysis["ignored_duplicates"], 12)

        metadata = prepare_dataset(self.source, self.categories, use_hard_links=False)
        expected = {"C1": {"OK": 3, "NG": 3}, "C2": {"OK": 3, "NG": 3}}
        self.assertEqual(metadata["counts_by_category"], expected)
        self.assertFalse((self.categories / "C1" / "train").exists())
        self.assertEqual(validate_ok_ng_dataset(self.categories / "C1"), {"OK": 3, "NG": 3})

        reused = prepare_dataset(self.source, self.categories, use_hard_links=False)
        self.assertTrue(reused["reused"])

    def test_system_discovers_names_dynamically_and_splits_at_training_time(self) -> None:
        prepare_dataset(self.source, self.categories, use_hard_links=False)
        datasets = discover_category_datasets(self.categories)
        self.assertEqual([dataset.category for dataset in datasets], ["C1", "C2"])

        split_root = self.root / "task" / "dataset_split"
        metadata = prepare_training_split(
            self.categories / "C1",
            split_root,
            normal_train_ratio=2 / 3,
            seed=7,
            use_hard_links=False,
        )
        self.assertEqual(metadata["counts"], {"train_OK": 2, "test_OK": 1, "test_NG": 3})
        self.assertTrue((split_root / "train" / "OK").is_dir())
        self.assertTrue((split_root / "test" / "NG").is_dir())

        reused = prepare_training_split(
            self.categories / "C1",
            split_root,
            normal_train_ratio=2 / 3,
            seed=7,
            use_hard_links=False,
        )
        self.assertTrue(reused["reused"])


if __name__ == "__main__":
    unittest.main()
