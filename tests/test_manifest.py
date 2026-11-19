"""Unit test: đọc manifest.json và kiểm tra cấu trúc task."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.manifest_path = ROOT / "K5-L2-DAY03-NguyenThanhQuang-2A202603083-Image-Segmentation" / "data" / "manifest.json"
        if not self.manifest_path.is_file():
            self.skipTest(f"Optional dataset fixture is not available: {self.manifest_path}")
        with open(self.manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)

    def test_tasks_are_present(self):
        tasks = self.manifest["tasks"]
        expected = {
            "easy_semantic", "medium_instance", "hard_panoptic",
            "cp1_holes", "cp2_slice", "cp5_occlusion",
            "cp3_thin", "cp4_curb", "cp6_coverage",
        }
        self.assertEqual(set(tasks), expected)

    def test_task_types_and_weights(self):
        tasks = self.manifest["tasks"]
        self.assertEqual(tasks["easy_semantic"]["type"], "semantic")
        self.assertEqual(tasks["medium_instance"]["type"], "instance")
        self.assertEqual(tasks["hard_panoptic"]["type"], "panoptic")
        total = sum(i["weight"] for i in tasks.values())
        self.assertEqual(total, 100)

    def test_task_paths_exist(self):
        for name, info in self.manifest["tasks"].items():
            base = ROOT / "K5-L2-DAY03-NguyenThanhQuang-2A202603083-Image-Segmentation" / "data" / info["path"]
            self.assertTrue(base.is_dir(), f"Thiếu thư mục {base}")
            self.assertTrue((base / "classes.json").is_file(), f"Thiếu classes.json của {name}")
            self.assertTrue((base / "cvat-labels.json").is_file(), f"Thiếu cvat-labels.json của {name}")
            images = list((base / "images").glob("*.jpg"))
            self.assertGreater(len(images), 0, f"Không có ảnh của {name}")


if __name__ == "__main__":
    unittest.main()
