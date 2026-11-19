"""Focused tests for CVAT auth, safe syncing, disk storage, and GT metrics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.annotation_processor import draw_cvat_mask, get_cvat_sublabels, keypoints_to_cvat_skeleton, mask_to_cvat_points
from src.cvat_client import CVATClient
from src.evaluation import cvat_shapes_to_records, evaluate_instances
from src.prediction_store import PredictionStore
from src.quality_checker import MaskQCResult


class TestMaskEncodingAndMetrics(unittest.TestCase):
    def test_cvat_mask_round_trip(self):
        mask = np.zeros((11, 13), dtype=np.uint8)
        mask[1:5, 2:6] = 1
        mask[8, 11] = 1
        points = mask_to_cvat_points(mask)
        decoded = np.zeros_like(mask)
        draw_cvat_mask(decoded, points)
        np.testing.assert_array_equal(decoded, mask)

    def test_metrics_use_ground_truth_and_do_not_claim_from_threshold(self):
        gt_mask = np.zeros((12, 12), dtype=np.uint8)
        gt_mask[2:8, 3:9] = 1
        false_mask = np.zeros_like(gt_mask)
        false_mask[0:2, 0:2] = 1
        scores = evaluate_instances(
            [
                {"image_id": "a.jpg", "class_name": "car", "mask": gt_mask, "confidence": 0.8},
                {"image_id": "a.jpg", "class_name": "car", "mask": false_mask, "confidence": 0.1},
            ],
            [{"image_id": "a.jpg", "class_name": "car", "mask": gt_mask}],
        )
        self.assertEqual(scores["tp"], 1)
        self.assertEqual(scores["fp"], 1)
        self.assertEqual(scores["fn"], 0)
        self.assertEqual(scores["precision"], 0.5)
        self.assertEqual(scores["recall"], 1.0)
        self.assertEqual(scores["iou"], 1.0)
        self.assertEqual(scores["dice"], 1.0)

    def test_cvat_annotations_are_rasterized_per_shape(self):
        records = cvat_shapes_to_records(
            {
                "shapes": [
                    {"type": "rectangle", "frame": 4, "label_id": 7, "points": [1, 1, 3, 3]},
                    {"type": "polygon", "frame": 4, "label_id": 7, "points": [5, 5, 8, 5, 8, 8]},
                ]
            },
            [{"id": 7, "name": "car"}],
            [{"number": 4, "name": "frame_4.jpg", "width": 12, "height": 12}],
        )
        self.assertEqual(len(records), 2)
        self.assertEqual({item["image_id"] for item in records}, {"frame_4.jpg"})
        self.assertTrue(all(item["mask"].shape == (12, 12) for item in records))

    def test_pose_keypoints_map_to_matching_cvat_skeleton_sublabels(self):
        shape = keypoints_to_cvat_skeleton(
            [
                {"name": "left_shoulder", "x": 12.5, "y": 8.0, "confidence": 0.9},
                {"name": "right_elbow", "x": 20.0, "y": 9.0, "confidence": 0.1},
                {"name": "nose", "x": 4.0, "y": 3.0, "confidence": None},
            ],
            frame=15,
            label_id=5,
            sublabels=[{"id": 51, "name": "left shoulder"}, {"id": 52, "name": "nose"}],
        )
        self.assertEqual(shape["type"], "skeleton")
        self.assertEqual(shape["frame"], 15)
        self.assertEqual([point["label_id"] for point in shape["elements"]], [51, 52])
        self.assertEqual(shape["elements"][0]["points"], [12.5, 8.0])
        flat_labels = [
            {"id": 5, "name": "person", "type": "skeleton"},
            {"id": 51, "name": "left shoulder", "parent_id": 5},
        ]
        self.assertEqual(get_cvat_sublabels(flat_labels, flat_labels[0]), [flat_labels[1]])


class TestCVATClientSafety(unittest.TestCase):
    def test_pat_uses_bearer_auth(self):
        client = CVATClient("https://example.invalid", access_token="test-token", auth_mode="pat")
        self.assertEqual(client.session.headers["Accept"], "application/vnd.cvat+json")
        with patch.object(client, "is_authenticated", return_value=True):
            client.login()
        self.assertEqual(client.session.headers["Authorization"], "Bearer test-token")

    def test_projects_follow_pagination(self):
        client = CVATClient("https://example.invalid")
        pages = [
            Mock(json=lambda: {"results": [{"id": 1}], "next": "/api/projects?page=2"}),
            Mock(json=lambda: {"results": [{"id": 2}], "next": None}),
        ]
        with patch.object(client, "_request", side_effect=pages):
            projects = client.list_projects()
        self.assertEqual([project["id"] for project in projects], [1, 2])

    def test_append_keeps_old_shapes_and_skips_duplicate(self):
        client = CVATClient("https://example.invalid")
        old_shape = {"type": "rectangle", "frame": 0, "label_id": 3, "points": [1.0, 2.0, 8.0, 9.0]}
        new_shape = {"type": "rectangle", "frame": 1, "label_id": 3, "points": [2, 3, 9, 10]}
        client.get_job_annotations = Mock(return_value={"shapes": [old_shape]})
        client.patch_job_annotations = Mock(return_value=Mock())
        result = client.append_job_shapes(10, [old_shape, new_shape, new_shape])
        self.assertEqual(result, {"added": 1, "duplicates": 1})
        payload = client.patch_job_annotations.call_args.args[1]
        self.assertEqual(payload["shapes"], [new_shape])
        self.assertEqual(client.patch_job_annotations.call_args.kwargs["action"], "create")


class TestPredictionStore(unittest.TestCase):
    def test_frame_is_persisted_and_loaded_lazily(self):
        image = np.zeros((9, 10, 3), dtype=np.uint8)
        image[2:7, 1:5] = (20, 80, 120)
        mask = np.zeros((9, 10), dtype=np.uint8)
        mask[2:7, 1:5] = 1
        qc = MaskQCResult(0, "car", True, 0.82, (9, 10), passed=True, level="high")
        prediction = {
            "image": image,
            "frame_number": 6,
            "type": "instance",
            "mode": "YOLO Only",
            "masks": [{
                "mask": mask,
                "class_name": "car",
                "is_thing": True,
                "confidence": 0.82,
                "bbox": [1, 2, 4, 6],
                "segmentation_available": True,
                "keypoints": [{"name": "nose", "x": 3.0, "y": 4.0, "confidence": 0.8}],
                "approval_key": "1:2:6:0",
                "qc": qc,
            }],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PredictionStore.create(Path(temp_dir), 1, 2)
            store.save_frame("frame_6.jpg", prediction)
            self.assertEqual(store.frame_names, ["frame_6.jpg"])
            restored = store.load_frame("frame_6.jpg")
            np.testing.assert_array_equal(restored["image"], image)
            np.testing.assert_array_equal(restored["masks"][0]["mask"], mask)
            self.assertEqual(restored["masks"][0]["qc"].level, "high")
            self.assertEqual(restored["masks"][0]["keypoints"][0]["name"], "nose")

            updated = {**restored, "masks": [dict(restored["masks"][0])]}
            updated_mask = updated["masks"][0]["mask"].copy()
            updated_mask[0, 0] = 1
            updated["masks"][0]["mask"] = updated_mask
            store.save_frame("frame_6.jpg", updated, replace=True)
            revised = store.load_frame("frame_6.jpg")
            np.testing.assert_array_equal(revised["masks"][0]["mask"], updated_mask)


if __name__ == "__main__":
    unittest.main()
