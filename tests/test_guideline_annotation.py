from pathlib import Path
from tempfile import TemporaryDirectory
import math

from src.guideline_annotation import (
    export_annotations,
    is_facial_landmark_task,
    POINT_STATES,
    SKELETONS,
    SKELETON_POINT_GUIDANCE,
    STREET_LABELS,
    flags_point_state,
    has_self_intersection,
    load_annotation_file,
    point_state_flags,
    save_annotation_file,
    validate_street,
    validate_facial,
)


def test_street_taxonomy_and_shape_mapping_validation():
    assert sum(map(len, STREET_LABELS.values())) == 20
    assert STREET_LABELS["rectangle"] == ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"]
    assert STREET_LABELS["polygon"] == ["area/direct", "area/alternative"]
    assert STREET_LABELS["polyline"] == ["lane/crosswalk", "lane/double white", "lane/double yellow", "lane/double other", "lane/road curb", "lane/single white", "lane/single yellow", "lane/single other"]
    labels = {
        "rectangle": "car",
        "polygon": "area/direct",
        "polyline": "lane/crosswalk",
    }
    shapes = [
        {"type": kind, "label": label, "occluded": False, "truncated": True,
         "points": ([{"x": 1, "y": 1}, {"x": 10, "y": 1}, {"x": 10, "y": 10}, {"x": 1, "y": 10}] if kind == "rectangle" else
                    [{"x": 2, "y": 2}, {"x": 8, "y": 2}, {"x": 8, "y": 8}] if kind == "polygon" else
                    [{"x": 1, "y": 1}, {"x": 8, "y": 8}])}
        for kind, label in labels.items()
    ]
    assert validate_street(shapes, 20, 20) == []
    overlap = dict(shapes[0], points=[{"x": 5, "y": 5}, {"x": 12, "y": 5}, {"x": 12, "y": 12}, {"x": 5, "y": 12}])
    assert validate_street(shapes + [overlap], 20, 20) == []
    shapes[0]["type"] = "polygon"
    assert any("must use rectangle" in error for error in validate_street(shapes, 20, 20))


def test_polyline_open_and_polygon_self_intersection():
    bow_tie = [{"x": 0, "y": 0}, {"x": 4, "y": 4}, {"x": 0, "y": 4}, {"x": 4, "y": 0}]
    assert has_self_intersection(bow_tie, closed=True)
    assert not has_self_intersection(bow_tie[:2], closed=False)


def test_facial_taxonomy_and_independent_state_flags():
    assert len(SKELETONS) == 7
    assert [s["start"] for s in SKELETONS] == [0, 5, 10, 14, 22, 30, 42]
    assert POINT_STATES == ("Visible", "Occluded", "Outside")
    assert point_state_flags("Occluded") == {"outside": False, "occluded": True}
    assert point_state_flags("Outside") == {"outside": True, "occluded": False}
    assert flags_point_state(False, False) == "Visible"
    assert is_facial_landmark_task([spec["name"] for spec in SKELETONS])
    assert is_facial_landmark_task([spec["name"].upper() for spec in SKELETONS])
    assert not is_facial_landmark_task([spec["name"] for spec in SKELETONS[:-1]])
    assert not is_facial_landmark_task([spec["name"] for spec in SKELETONS] + ["car"])


def test_facial_point_guidance_covers_each_skeleton_in_point_id_order():
    assert set(SKELETON_POINT_GUIDANCE) == {spec["name"] for spec in SKELETONS}
    for spec in SKELETONS:
        guidance = SKELETON_POINT_GUIDANCE[spec["name"]]
        assert len(guidance) == spec["count"]
        assert all(isinstance(cue, str) and cue.strip() for cue in guidance)
    assert "đuôi ngoài" in SKELETON_POINT_GUIDANCE["longmaytrai"][0]
    assert "đầu trong" in SKELETON_POINT_GUIDANCE["longmaytrai"][-1]
    assert "phía trên lỗ mũi" in SKELETON_POINT_GUIDANCE["songmui"][-1]
    assert "84%" in SKELETON_POINT_GUIDANCE["moingoai"][5]


def test_annotation_json_round_trip():
    image = {"key": "image-key", "name": "face.jpg", "width": 31, "height": 19}
    annotations = [{"type": "skeleton", "label": "longmaytrai", "contour": "open", "points": [{"id": 0, "x": 2, "y": 3, "state": "Outside"}]}]
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        path = Path(temp_dir) / "annotation.json"
        save_annotation_file(path, image, "Facial Landmark-50", annotations)
        restored = load_annotation_file(path.read_bytes(), image, "Facial Landmark-50")
        assert restored[0]["points"][0]["state"] == "Outside"
        assert restored[0]["points"][0]["outside"] is True
        assert restored[0]["points"][0]["occluded"] is False


def test_json_export_uses_cvat_native_keypoint_flags():
    points = [{"id": 0, "x": 1, "y": 2, "state": "Outside"}, {"id": 1, "x": 3, "y": 4, "state": "Occluded"}]
    exported = export_annotations([{"type": "skeleton", "label": "longmaytrai", "points": points}])
    assert [(p["outside"], p["occluded"]) for p in exported[0]["points"]] == [(True, False), (False, True)]
    assert "outside" not in points[0]


def test_facial_validator_accepts_complete_clockwise_topology():
    shapes = []
    for spec in SKELETONS:
        count, start, name = spec["count"], spec["start"], spec["name"]
        if spec["contour"] == "closed":
            cx, cy = {"mattrai": (30, 30), "matphai": (70, 30), "moingoai": (50, 80), "moitrong": (50, 80)}[name]
            rx, ry = (8, 4) if name.startswith("mat") else ((20, 10) if name == "moingoai" else (10, 4))
            points = []
            for i in range(count):
                angle = 3.141592653589793 - i * (2 * 3.141592653589793 / count)
                points.append({"id": start + i, "x": cx + rx * math.cos(angle), "y": cy - ry * math.sin(angle), "state": "Visible"})
        else:
            points = [{"id": start + i, "x": 10 + i * 3, "y": 10 + i * 2, "state": "Visible"} for i in range(count)]
        shapes.append({"type": "skeleton", "label": name, "contour": spec["contour"], "points": points})
    assert validate_facial(shapes, 100, 100) == []


    # The guideline explicitly allows closed-eye and closed-mouth lid pairs to coincide.
    for shape in shapes:
        if shape["label"] in ("mattrai", "matphai", "moitrong"):
            points = shape["points"]
            for upper, lower in ((1, 7), (2, 6), (3, 5)):
                points[lower]["x"], points[lower]["y"] = points[upper]["x"], points[upper]["y"]
    assert validate_facial(shapes, 100, 100) == []
    for shape in shapes:
        if shape["label"] == "mattrai":
            for point in shape["points"]:
                point["state"] = "Outside"
                point["x"] = point["y"] = None
    assert validate_facial(shapes, 100, 100) == []


def test_facial_validator_allows_an_empty_or_partial_connected_job_draft():
    assert validate_facial([], 100, 100, allow_incomplete=True) == []
    partial = [{
        "type": "skeleton", "label": "longmaytrai", "contour": "open",
        "points": [{"id": 0, "x": 5.0, "y": 5.0, "state": "Visible"}],
    }]
    assert validate_facial(partial, 100, 100, allow_incomplete=True) == []
    assert any("Missing skeleton" in error for error in validate_facial(partial, 100, 100))
