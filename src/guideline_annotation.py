"""Taxonomy, geometry validation, and persistence for guideline canvas modes."""
from __future__ import annotations

import json
import math
import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

STREET_LABELS = {
    "rectangle": ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"],
    "polygon": ["area/direct", "area/alternative"],
    "polyline": ["lane/crosswalk", "lane/double white", "lane/double yellow", "lane/double other", "lane/road curb", "lane/single white", "lane/single yellow", "lane/single other"],
}
STREET_LABEL_TO_SHAPE = {label: shape for shape, labels in STREET_LABELS.items() for label in labels}
SKELETONS = [
    {"name": "longmaytrai", "start": 0, "count": 5, "contour": "open", "minimum_visible": 3},
    {"name": "longmayphai", "start": 5, "count": 5, "contour": "open", "minimum_visible": 3},
    {"name": "songmui", "start": 10, "count": 4, "contour": "open", "minimum_visible": 2},
    {"name": "mattrai", "start": 14, "count": 8, "contour": "closed", "minimum_visible": 4},
    {"name": "matphai", "start": 22, "count": 8, "contour": "closed", "minimum_visible": 4},
    {"name": "moingoai", "start": 30, "count": 12, "contour": "closed", "minimum_visible": 6},
    {"name": "moitrong", "start": 42, "count": 8, "contour": "closed", "minimum_visible": 4},
]
SKELETON_BY_NAME = {s["name"]: s for s in SKELETONS}
FACIAL_LANDMARK_LABELS = frozenset(SKELETON_BY_NAME)
# Human-readable placement cues shown while annotating. The sequence follows
# the guideline's global Point IDs and must never be used to reorder points.
SKELETON_POINT_GUIDANCE = {
    "longmaytrai": [
        "đuôi ngoài lông mày trái", "bờ trên lông mày trái · 1/4",
        "bờ trên lông mày trái · 1/2", "bờ trên lông mày trái · 3/4",
        "đầu trong lông mày trái",
    ],
    "longmayphai": [
        "đầu trong lông mày phải", "bờ trên lông mày phải · 1/4",
        "bờ trên lông mày phải · 1/2", "bờ trên lông mày phải · 3/4",
        "đuôi ngoài lông mày phải",
    ],
    "songmui": [
        "đỉnh sống mũi · vùng lõm giữa hai mắt",
        "sống mũi · 1/3 trên", "sống mũi · 2/3 trên",
        "chân sống mũi · phía trên lỗ mũi, không phải chóp mũi",
    ],
    "mattrai": [
        "khóe ngoài mắt trái", "mí trên · 1/4 từ trái sang phải",
        "mí trên · giữa", "mí trên · 3/4 từ trái sang phải",
        "khóe trong mắt trái", "mí dưới · 1/4 từ phải sang trái",
        "mí dưới · giữa", "mí dưới · 3/4 từ phải sang trái",
    ],
    "matphai": [
        "khóe trong mắt phải", "mí trên · 1/4 từ trái sang phải",
        "mí trên · giữa", "mí trên · 3/4 từ trái sang phải",
        "khóe ngoài mắt phải", "mí dưới · 1/4 từ phải sang trái",
        "mí dưới · giữa", "mí dưới · 3/4 từ phải sang trái",
    ],
    "moingoai": [
        "khóe miệng trái", "bờ môi trên · 18% chiều rộng",
        "bờ môi trên · 36% chiều rộng", "bờ môi trên · chính giữa",
        "bờ môi trên · 66% chiều rộng", "bờ môi trên · 84% chiều rộng",
        "khóe miệng phải", "bờ môi dưới · 84% chiều rộng từ trái",
        "bờ môi dưới · 66% chiều rộng từ trái", "bờ môi dưới · chính giữa",
        "bờ môi dưới · 36% chiều rộng từ trái",
        "bờ môi dưới · 18% chiều rộng từ trái",
    ],
    "moitrong": [
        "khóe trong miệng trái", "mép trong môi trên · 1/4",
        "mép trong môi trên · giữa", "mép trong môi trên · 3/4",
        "khóe trong miệng phải", "mép trong môi dưới · 3/4 từ phải sang trái",
        "mép trong môi dưới · giữa", "mép trong môi dưới · 1/4 từ phải sang trái",
    ],
}
POINT_STATES = ("Visible", "Occluded", "Outside")
ISSUE_TYPES = ("UNCERTAIN_CLASS", "UNCERTAIN_BOUNDARY", "UNCERTAIN_SCOPE", "ATTRIBUTE_CHECK", "LOW_VISIBILITY")


def is_facial_landmark_task(label_names: Iterable[str]) -> bool:
    """Identify the seven-label Facial Landmark-50 taxonomy by exact names."""
    normalized = {str(name).strip().casefold() for name in label_names}
    return normalized == FACIAL_LANDMARK_LABELS


def new_skeleton(name: str) -> dict[str, Any]:
    spec = SKELETON_BY_NAME[name]
    return {"type": "skeleton", "label": name, "contour": spec["contour"], "points": []}


def point_state_flags(state: str) -> dict[str, bool]:
    if state not in POINT_STATES:
        raise ValueError(f"Invalid keypoint state: {state}")
    return {"outside": state == "Outside", "occluded": state == "Occluded"}


def flags_point_state(outside: bool, occluded: bool) -> str:
    if outside and occluded:
        raise ValueError("CVAT keypoint cannot be both outside and occluded")
    return "Outside" if outside else "Occluded" if occluded else "Visible"


def signed_area(points: Iterable[Mapping[str, Any]]) -> float:
    pts = list(points)
    if len(pts) < 3:
        return 0.0
    return sum(float(pts[i]["x"]) * float(pts[(i + 1) % len(pts)]["y"]) - float(pts[(i + 1) % len(pts)]["x"]) * float(pts[i]["y"]) for i in range(len(pts))) / 2


def _orient(a: Mapping[str, Any], b: Mapping[str, Any], c: Mapping[str, Any]) -> float:
    return (float(b["x"]) - float(a["x"])) * (float(c["y"]) - float(a["y"])) - (float(b["y"]) - float(a["y"])) * (float(c["x"]) - float(a["x"]))


def _segments_intersect(a: Mapping[str, Any], b: Mapping[str, Any], c: Mapping[str, Any], d: Mapping[str, Any]) -> bool:
    eps = 1e-8
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True
    def on(p: Mapping[str, Any], q: Mapping[str, Any], r: Mapping[str, Any]) -> bool:
        return min(float(p["x"]), float(r["x"])) - eps <= float(q["x"]) <= max(float(p["x"]), float(r["x"])) + eps and min(float(p["y"]), float(r["y"])) - eps <= float(q["y"]) <= max(float(p["y"]), float(r["y"])) + eps
    return (abs(o1) <= eps and on(a, c, b)) or (abs(o2) <= eps and on(a, d, b)) or (abs(o3) <= eps and on(c, a, d)) or (abs(o4) <= eps and on(c, b, d))


def has_self_intersection(
    points: list[Mapping[str, Any]],
    closed: bool = True,
    ignored_edge_pairs: set[tuple[int, int]] | None = None,
    allowed_duplicate_point_pairs: set[tuple[int, int]] | None = None,
) -> bool:
    n = len(points)
    edge_count = n if closed else n - 1
    ignored = {tuple(sorted(pair)) for pair in (ignored_edge_pairs or set())}
    duplicate_pairs = {tuple(sorted(pair)) for pair in (allowed_duplicate_point_pairs or set())}
    for i in range(edge_count):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, edge_count):
            if (i, j) in ignored:
                continue
            if j == i or j == i + 1 or (closed and i == 0 and j == edge_count - 1):
                continue
            c, d = points[j], points[(j + 1) % n]
            left_indices = (i, (i + 1) % n)
            right_indices = (j, (j + 1) % n)
            homologous_overlap = any(
                tuple(sorted((left_id, right_id))) in duplicate_pairs
                and math.hypot(float(points[left_id]["x"]) - float(points[right_id]["x"]), float(points[left_id]["y"]) - float(points[right_id]["y"])) <= 1.0
                for left_id in left_indices for right_id in right_indices
            )
            if homologous_overlap:
                continue
            if _segments_intersect(a, b, c, d):
                return True
    return False


def _valid_points(points: list[Mapping[str, Any]], width: int, height: int) -> list[str]:
    issues = []
    for index, point in enumerate(points):
        try:
            x, y = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"Point {index + 1} has invalid coordinates")
            continue
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < width and 0 <= y < height):
            issues.append(f"Point {index + 1} is outside the {width}×{height} image")
    return issues


def validate_street(shapes: list[Mapping[str, Any]], width: int, height: int) -> list[str]:
    issues: list[str] = []
    for index, shape in enumerate(shapes, start=1):
        label, kind, points = shape.get("label"), shape.get("type"), shape.get("points") or []
        expected = STREET_LABEL_TO_SHAPE.get(label)
        if expected is None:
            issues.append(f"Annotation {index}: label '{label}' is outside Street Scene taxonomy")
            continue
        if kind != expected:
            issues.append(f"Annotation {index}: '{label}' must use {expected}, got {kind}")
            continue
        if kind in ("rectangle", "polygon") and len(points) < (4 if kind == "rectangle" else 3):
            issues.append(f"Annotation {index}: {kind} needs more vertices")
        if kind == "polyline" and len(points) < 2:
            issues.append(f"Annotation {index}: polyline needs at least two points")
        issues.extend(f"Annotation {index}: {message}" for message in _valid_points(points, width, height))
        if kind == "rectangle" and len(points) >= 4:
            xs, ys = [float(p["x"]) for p in points], [float(p["y"]) for p in points]
            if max(xs) <= min(xs) or max(ys) <= min(ys):
                issues.append(f"Annotation {index}: rectangle has zero area")
        if kind == "polygon" and len(points) >= 3:
            if abs(signed_area(points)) < 1e-6:
                issues.append(f"Annotation {index}: polygon has zero area")
            if has_self_intersection(points, closed=True):
                issues.append(f"Annotation {index}: polygon self-intersects")
        if kind == "polyline" and has_self_intersection(points, closed=False):
            issues.append(f"Annotation {index}: polyline self-intersects")
        if kind == "rectangle":
            for attr in ("occluded", "truncated"):
                if attr not in shape:
                    issues.append(f"Annotation {index}: missing {attr} attribute")
                elif not isinstance(shape.get(attr), bool):
                    issues.append(f"Annotation {index}: {attr} must be true or false")
    return issues


def validate_facial(
    shapes: list[Mapping[str, Any]],
    width: int,
    height: int,
    *,
    allow_incomplete: bool = False,
) -> list[str]:
    issues: list[str] = []
    by_name: dict[str, Mapping[str, Any]] = {}
    seen_ids: set[int] = set()
    for shape in shapes:
        name = shape.get("label")
        if name not in SKELETON_BY_NAME:
            issues.append(f"Unknown facial skeleton: {name}")
            continue
        if name in by_name:
            issues.append(f"Duplicate skeleton: {name}")
        by_name[name] = shape
        spec = SKELETON_BY_NAME[name]
        if shape.get("type") != "skeleton" or shape.get("contour") != spec["contour"]:
            issues.append(f"{name}: expected {spec['contour']} skeleton topology")
        points = shape.get("points") or []
        ids = [point.get("id") for point in points]
        expected_ids = list(range(spec["start"], spec["start"] + spec["count"]))
        valid_draft_ids = len(ids) <= len(expected_ids) and ids == expected_ids[:len(ids)]
        if (allow_incomplete and not valid_draft_ids) or (not allow_incomplete and ids != expected_ids):
            issues.append(f"{name}: expected Point IDs {expected_ids[0]}–{expected_ids[-1]} in order")
        for point in points:
            pid = point.get("id")
            if isinstance(pid, int):
                if pid in seen_ids:
                    issues.append(f"Duplicate Point ID {pid}")
                seen_ids.add(pid)
            if point.get("state") not in POINT_STATES:
                issues.append(f"Point {pid}: missing or invalid state")
            elif "outside" in point or "occluded" in point:
                expected_flags = point_state_flags(point["state"])
                if bool(point.get("outside")) != expected_flags["outside"] or bool(point.get("occluded")) != expected_flags["occluded"]:
                    issues.append(f"Point {pid}: CVAT outside/occluded flags do not match its state")
        issues.extend(f"{name}: {message}" for message in _valid_points([p for p in points if p.get("state") != "Outside"], width, height))
        wholly_outside = len(points) == spec["count"] and all(point.get("state") == "Outside" for point in points)
        if len(points) == spec["count"] and spec["contour"] == "closed" and not wholly_outside:
            allowed_coincident_pairs = {(1, 7), (2, 6), (3, 5)} if name in ("mattrai", "matphai", "moitrong") else set()
            if has_self_intersection(points, closed=True, allowed_duplicate_point_pairs=allowed_coincident_pairs):
                issues.append(f"{name}: contour self-intersects")
        if len(points) == spec["count"] and spec["contour"] == "closed" and not wholly_outside:
            area = signed_area(points)
            if area < -1e-6:
                issues.append(f"{name}: contour must follow the guideline's clockwise point order")
            min_x = min(float(point["x"]) for point in points)
            if float(points[0]["x"]) > min_x + 1:
                issues.append(f"{name}: contour must start from the leftmost guideline point")
    for spec in SKELETONS:
        if spec["name"] not in by_name and not allow_incomplete:
            issues.append(f"Missing skeleton: {spec['name']}")
    outer, inner = by_name.get("moingoai"), by_name.get("moitrong")
    if outer and inner and len(outer.get("points", [])) == 12 and len(inner.get("points", [])) == 8:
        # For each inner point, require it to be inside or on the outer contour.
        poly = outer["points"]
        for p in inner["points"]:
            if p.get("state") == "Outside":
                continue
            x, y = float(p["x"]), float(p["y"])
            inside = False
            for i, a in enumerate(poly):
                b = poly[(i + 1) % len(poly)]
                if _orient(a, b, p) == 0 and min(a["x"], b["x"]) <= x <= max(a["x"], b["x"]) and min(a["y"], b["y"]) <= y <= max(a["y"], b["y"]):
                    inside = True
                    break
                if (float(a["y"]) > y) != (float(b["y"]) > y):
                    at_x = (float(b["x"]) - float(a["x"])) * (y - float(a["y"])) / (float(b["y"]) - float(a["y"])) + float(a["x"])
                    if x < at_x:
                        inside = not inside
            if not inside:
                issues.append("moitrong: a keypoint lies outside moingoai")
                break
    return issues


def save_annotation_file(path: str | Path, image_record: Mapping[str, Any], mode: str, annotations: list[Mapping[str, Any]], issues: list[Mapping[str, str]] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "mode": mode, "image": {"key": image_record["key"], "name": image_record["name"], "width": int(image_record["width"]), "height": int(image_record["height"])}, "annotations": export_annotations(annotations), "issues": issues or []}
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)


def load_annotation_file(data: bytes, image_record: Mapping[str, Any], mode: str) -> list[dict[str, Any]]:
    payload = json.loads(data.decode("utf-8"))
    if payload.get("schema_version") != 1 or payload.get("mode") != mode:
        raise ValueError("File annotation has an incompatible schema or mode")
    image = payload.get("image", {})
    if (image.get("width"), image.get("height")) != (int(image_record["width"]), int(image_record["height"])):
        raise ValueError("Image dimensions do not match the selected image")
    if image.get("key") != image_record["key"]:
        raise ValueError("File belongs to a different image")
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("Invalid annotation list")
    restored = copy.deepcopy(annotations)
    for shape in restored:
        for point in shape.get("points", []):
            if "state" not in point and ("outside" in point or "occluded" in point):
                point["state"] = flags_point_state(bool(point.get("outside")), bool(point.get("occluded")))
    return restored


def export_annotations(annotations: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep readable states and include CVAT's native per-point flags for migration."""
    exported = copy.deepcopy(list(annotations))
    for shape in exported:
        if shape.get("type") != "skeleton":
            continue
        for point in shape.get("points", []):
            point.update(point_state_flags(point.get("state", "Visible")))
    return exported
