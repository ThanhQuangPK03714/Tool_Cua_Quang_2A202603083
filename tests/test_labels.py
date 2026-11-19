"""Unit test: kiểm tra label của từng task khớp với classes.json."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class TestLabels(unittest.TestCase):
    def setUp(self):
        self.repo_root = ROOT / "K5-L2-DAY03-NguyenThanhQuang-2A202603083-Image-Segmentation"
        self.manifest_path = self.repo_root / "data" / "manifest.json"
        if not self.manifest_path.is_file():
            self.skipTest(f"Optional dataset fixture is not available: {self.manifest_path}")
        with open(self.manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)

    def test_labels_match_classes(self):
        """Tên label trong cvat-labels.json phải khớp classes.json của cùng task."""
        for name, info in self.manifest["tasks"].items():
            base = self.repo_root / "data" / info["path"]
            classes = json.loads((base / "classes.json").read_text(encoding="utf-8"))["classes"]
            labels = json.loads((base / "cvat-labels.json").read_text(encoding="utf-8"))
            label_names = [lbl["name"] for lbl in labels]
            self.assertEqual(label_names, classes, f"Task {name}: label không khớp classes")

    def test_label_fields(self):
        """Mỗi label phải có đúng các trường: name, color, type, attributes."""
        for name, info in self.manifest["tasks"].items():
            base = self.repo_root / "data" / info["path"]
            labels = json.loads((base / "cvat-labels.json").read_text(encoding="utf-8"))
            for lbl in labels:
                self.assertEqual(set(lbl), {"name", "color", "type", "attributes"}, f"Task {name}: label fields")
                self.assertEqual(lbl["type"], "any", f"Task {name}: label type")
                self.assertEqual(lbl["attributes"], [], f"Task {name}: label attributes")


if __name__ == "__main__":
    unittest.main()
