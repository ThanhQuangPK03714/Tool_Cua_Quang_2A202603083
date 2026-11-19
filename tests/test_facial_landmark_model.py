from types import SimpleNamespace

from src.facial_landmark.mp_face_landmarker import map_mesh_to_facial_landmark50
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
