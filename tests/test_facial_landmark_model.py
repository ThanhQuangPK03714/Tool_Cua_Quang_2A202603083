from types import SimpleNamespace

from src.facial_landmark.mp_face_landmarker import map_mesh_to_facial_landmark50
from src.facial_landmark.annotation_state import merge_prediction_preserving_reviewed
from src.guideline_annotation import SKELETONS


def test_media_mesh_mapping_produces_all_50_guideline_ids_as_review_pending():
    mesh = [
        SimpleNamespace(x=((index * 37) % 478) / 478, y=((index * 73) % 478) / 478)
        for index in range(478)
    ]
    shapes = map_mesh_to_facial_landmark50(mesh, 1280, 720)

    assert [shape["label"] for shape in shapes] == [spec["name"] for spec in SKELETONS]
    assert sum(len(shape["points"]) for shape in shapes) == 50
    assert [point["id"] for shape in shapes for point in shape["points"]] == list(range(50))
    assert all(point["review_status"] == "pending" for shape in shapes for point in shape["points"])
    assert all(point["source"] == "mediapipe_face_landmarker_478" for shape in shapes for point in shape["points"])


def test_prediction_merge_preserves_manual_and_accepted_points():
    existing = [{
        "type": "skeleton", "label": "longmaytrai", "contour": "open", "face_instance_id": "face-01",
        "points": [
            {"id": 0, "x": 10, "y": 11, "source": "manual", "review_status": "pending"},
            {"id": 1, "x": 12, "y": 13, "source": "mediapipe_face_landmarker_478", "review_status": "accepted"},
        ],
    }]
    proposed = [{
        "type": "skeleton", "label": "longmaytrai", "contour": "open",
        "points": [{"id": i, "x": 100 + i, "y": 200 + i, "source": "mediapipe_face_landmarker_478", "review_status": "pending"} for i in range(5)],
    }]
    merged = merge_prediction_preserving_reviewed(existing, proposed, "face-01")
    points = {point["id"]: point for point in merged[0]["points"]}
    assert points[0]["x"] == 10
    assert points[1]["x"] == 12
    assert points[2]["x"] == 102
    assert merged[0]["face_instance_id"] == "face-01"
