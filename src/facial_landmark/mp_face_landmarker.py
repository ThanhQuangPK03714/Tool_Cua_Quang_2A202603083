"""MediaPipe Face Landmarker adapter and guideline-indexed 50-point proposals."""
from __future__ import annotations

import functools
import logging
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

from src.guideline_annotation import SKELETONS

logger = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# Landmark contours from the official MediaPipe Face Landmarker connections.
_EYES = {
    "left": (
        [263, 466, 388, 387, 386, 385, 384, 398, 362],
        [263, 249, 390, 373, 374, 380, 381, 382, 362],
    ),
    "right": (
        [33, 246, 161, 160, 159, 158, 157, 173, 133],
        [33, 7, 163, 144, 145, 153, 154, 155, 133],
    ),
}
_EYEBROWS = {
    "left": ([276, 283, 282, 295, 285], [300, 293, 334, 296, 336]),
    "right": ([46, 53, 52, 65, 55], [70, 63, 105, 66, 107]),
}
_NOSE = [168, 6, 197, 195, 5]
_OUTER_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
_OUTER_LOWER = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
_INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
_INNER_LOWER = [308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]


def _xy(landmarks: Sequence[Any], index: int, width: int, height: int) -> tuple[float, float]:
    point = landmarks[index]
    return float(point.x) * width, float(point.y) * height


def _orient_by_x(
    indices: Sequence[int], landmarks: Sequence[Any], width: int, height: int, *, ascending: bool
) -> list[int]:
    ordered = list(indices)
    first_x = _xy(landmarks, ordered[0], width, height)[0]
    last_x = _xy(landmarks, ordered[-1], width, height)[0]
    if (last_x >= first_x) != ascending:
        ordered.reverse()
    return ordered


def _sample_arc(
    indices: Sequence[int], landmarks: Sequence[Any], width: int, height: int, fraction: float
) -> tuple[float, float]:
    coords = [_xy(landmarks, index, width, height) for index in indices]
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(coords, coords[1:])]
    total = sum(lengths)
    if total <= 1e-8:
        return coords[0]
    target = min(1.0, max(0.0, fraction)) * total
    covered = 0.0
    for (x1, y1), (x2, y2), length in zip(coords, coords[1:], lengths):
        if covered + length >= target:
            ratio = (target - covered) / max(length, 1e-8)
            return x1 + ratio * (x2 - x1), y1 + ratio * (y2 - y1)
        covered += length
    return coords[-1]


def _sample_horizontal(
    indices: Sequence[int], landmarks: Sequence[Any], width: int, height: int, fraction: float,
    *, upper: bool,
) -> tuple[float, float]:
    """Sample an upper/lower contour at a fraction of its image-space width."""
    coords = [_xy(landmarks, index, width, height) for index in indices]
    x0, x1 = coords[0][0], coords[-1][0]
    target_x = x0 + min(1.0, max(0.0, fraction)) * (x1 - x0)
    candidates = []
    for (xa, ya), (xb, yb) in zip(coords, coords[1:]):
        if min(xa, xb) - 1e-6 <= target_x <= max(xa, xb) + 1e-6 and abs(xb - xa) > 1e-8:
            ratio = (target_x - xa) / (xb - xa)
            candidates.append((target_x, ya + ratio * (yb - ya)))
    if candidates:
        return min(candidates, key=lambda p: p[1]) if upper else max(candidates, key=lambda p: p[1])
    return min(coords, key=lambda p: abs(p[0] - target_x))


def _eye_annotation(
    label: str,
    paths: tuple[Sequence[int], Sequence[int]],
    landmarks: Sequence[Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    first, second = paths
    upper_path, lower_path = sorted(
        (list(first), list(second)),
        key=lambda path: sum(_xy(landmarks, index, width, height)[1] for index in path) / len(path),
    )
    # Guideline order starts at the image-left canthus, crosses the upper lid,
    # then returns along the lower lid in the opposite direction.
    upper_path = _orient_by_x(upper_path, landmarks, width, height, ascending=True)
    lower_path = _orient_by_x(lower_path, landmarks, width, height, ascending=False)
    coords = [_sample_horizontal(upper_path, landmarks, width, height, f, upper=True)
              for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    coords.extend(_sample_horizontal(lower_path, landmarks, width, height, f, upper=False)
                  for f in (0.25, 0.5, 0.75))
    spec = next(item for item in SKELETONS if item["name"] == label)
    return _shape_from_coords(label, spec, coords, width, height)


def _shape_from_coords(
    label: str, spec: dict[str, Any], coords: Sequence[tuple[float, float]], width: int, height: int
) -> dict[str, Any]:
    points = []
    for offset, (x, y) in enumerate(coords):
        state = "Outside" if x < 0 or y < 0 or x >= width or y >= height else "Visible"
        points.append({
            "id": int(spec["start"]) + offset,
            "x": float(x), "y": float(y), "state": state,
            "source": "mediapipe_face_landmarker_478",
            "review_status": "pending",
        })
    return {"type": "skeleton", "label": label, "contour": spec["contour"], "points": points}


def map_mesh_to_facial_landmark50(
    landmarks: Sequence[Any], width: int, height: int
) -> list[dict[str, Any]]:
    """Map official face-mesh contours to the guideline's 7 ordered groups.

    This returns a review-pending proposal. MediaPipe does not estimate CVAT
    visibility state or per-point confidence, so those are never fabricated.
    """
    if len(landmarks) < 468:
        raise ValueError(f"Expected at least 468 MediaPipe landmarks, got {len(landmarks)}")
    shapes: list[dict[str, Any]] = []

    # Image-side assignment follows the project's visible-left/right rule.
    eyebrow_candidates = []
    for side, curves in _EYEBROWS.items():
        curve = min(
            curves,
            key=lambda path: sum(_xy(landmarks, i, width, height)[1] for i in path) / len(path),
        )
        xs = [_xy(landmarks, i, width, height)[0] for i in curve]
        eyebrow_candidates.append((sum(xs) / len(xs), curve))
    eyebrow_candidates.sort(key=lambda item: item[0])
    for label, (_, curve) in zip(("longmaytrai", "longmayphai"), eyebrow_candidates):
        # Point IDs follow the image-side guideline: left brow 0→4 runs
        # from the outer tail toward the inner head; right brow 5→9 runs
        # from the inner head toward the outer tail. Both move left-to-right
        # in image coordinates.
        curve = _orient_by_x(curve, landmarks, width, height, ascending=True)
        spec = next(item for item in SKELETONS if item["name"] == label)
        coords = [_sample_arc(curve, landmarks, width, height, f) for f in (0, .25, .5, .75, 1)]
        shapes.append(_shape_from_coords(label, spec, coords, width, height))

    nose_coords = [_sample_arc(_NOSE, landmarks, width, height, f) for f in (0, 1/3, 2/3, 1)]
    nose_spec = next(item for item in SKELETONS if item["name"] == "songmui")
    shapes.append(_shape_from_coords("songmui", nose_spec, nose_coords, width, height))

    eye_candidates = []
    for paths in _EYES.values():
        eye_x = sum(_xy(landmarks, i, width, height)[0] for i in paths[0]) / len(paths[0])
        eye_candidates.append((eye_x, paths))
    eye_candidates.sort(key=lambda item: item[0])
    for label, (_, paths) in zip(("mattrai", "matphai"), eye_candidates):
        shapes.append(_eye_annotation(label, paths, landmarks, width, height))

    outer_upper = _orient_by_x(_OUTER_UPPER, landmarks, width, height, ascending=True)
    outer_lower = _orient_by_x(_OUTER_LOWER, landmarks, width, height, ascending=False)
    outer_coords = [_sample_horizontal(outer_upper, landmarks, width, height, f, upper=True)
                    for f in (0, .18, .36, .50, .66, .84, 1)]
    outer_coords.extend(_sample_horizontal(outer_lower, landmarks, width, height, f, upper=False)
                        for f in (1/6, 2/6, 3/6, 4/6, 5/6))
    outer_spec = next(item for item in SKELETONS if item["name"] == "moingoai")
    shapes.append(_shape_from_coords("moingoai", outer_spec, outer_coords, width, height))

    inner_paths = [
        _orient_by_x(_INNER_UPPER, landmarks, width, height, ascending=True),
        _orient_by_x(list(reversed(_INNER_LOWER)), landmarks, width, height, ascending=True),
    ]
    inner_coords = []
    for fraction in (0, .25, .5, .75, 1):
        candidates = [
            _sample_horizontal(path, landmarks, width, height, fraction, upper=True)
            for path in inner_paths
        ]
        inner_coords.append(min(candidates, key=lambda point: point[1]))
    for fraction in (.75, .5, .25):
        candidates = [
            _sample_horizontal(path, landmarks, width, height, fraction, upper=False)
            for path in inner_paths
        ]
        inner_coords.append(max(candidates, key=lambda point: point[1]))
    # Both guideline contours share the mouth corner anchors. Reuse the outer
    # corners to keep the inner loop attached and inside the outer contour.
    inner_coords[0] = outer_coords[0]
    inner_coords[4] = outer_coords[6]
    inner_spec = next(item for item in SKELETONS if item["name"] == "moitrong")
    shapes.append(_shape_from_coords("moitrong", inner_spec, inner_coords, width, height))

    return shapes


@functools.lru_cache(maxsize=2)
def _create_landmarker(model_path: str, num_faces: int):
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu MediaPipe. Cài gói tùy chọn mediapipe theo hướng dẫn Facial Landmark."
        ) from exc
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Không thấy model Face Landmarker: {model_path}")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=int(num_faces),
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def propose_facial_landmark50(
    image_rgb: np.ndarray,
    *,
    num_faces: int = 5,
    model_path: str | Path = MODEL_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Detect face(s) and return 50-point proposal for the largest face."""
    try:
        import mediapipe as mp
    except ModuleNotFoundError as exc:
        if exc.name != "mediapipe":
            raise
        raise RuntimeError(
            "Python đang chạy Streamlit chưa có MediaPipe "
            f"({sys.executable}). Chạy setup_facial_landmark.ps1 với -PythonExe trỏ tới Python này."
        ) from exc

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Input must be a 3-channel RGB image")
    height, width = image_rgb.shape[:2]
    landmarker = _create_landmarker(str(Path(model_path).resolve()), int(num_faces))
    try:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb)))
    except Exception:
        logger.exception("MediaPipe face inference failed image_size=%sx%s", width, height)
        raise
    faces = result.face_landmarks or []
    if not faces:
        return [], {"face_count": 0, "model": "MediaPipe Face Landmarker", "status": "no_face"}

    def area(face: Sequence[Any]) -> float:
        xs = [float(point.x) for point in face]
        ys = [float(point.y) for point in face]
        return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))

    chosen = max(faces, key=area)
    shapes = map_mesh_to_facial_landmark50(chosen, width, height)
    meta = {
        "face_count": len(faces),
        "selected_face": "largest_bbox",
        "model": "MediaPipe Face Landmarker",
        "native_landmarks": len(chosen),
        "mapping": "guideline-v1-media-mesh-contours",
        "review_status": "pending",
    }
    if len(faces) > 1:
        meta["warning"] = "Nhiều mặt được phát hiện; app đã chọn mặt lớn nhất, hãy xác nhận đây là người lái."
    return shapes, meta
