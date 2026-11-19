"""CVAT export presets and conversions for mixed annotation geometry."""

from __future__ import annotations

import re
from typing import Any, Iterable

import cv2
import numpy as np


SHAPE_TYPES = {
    "Mask": "mask",
    "Polygon": "polygon",
    "Bounding box": "rectangle",
    "Polyline": "polyline",
    "Keypoints / Skeleton": "skeleton",
}

PRESET_LABELS = {
    "auto": "Tự nhận diện theo taxonomy",
    "road": "Đường phố: box + drivable area + lane marking",
    "semantic": "Semantic segmentation: polygon",
    "custom": "Giữ cấu hình từng label",
}


def normalize_export_label(name: str) -> str:
    """Normalize separators while preserving taxonomy paths such as ``area/direct``."""
    return re.sub(r"[\s_-]+", " ", str(name).strip().casefold())


def annotation_label_key(name: str) -> str:
    """Keep CVAT label identity exact; ``traffic light`` and ``traffic_light`` may coexist."""
    return str(name).strip()


SEMANTIC_LABELS = {
    normalize_export_label(name)
    for name in (
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
        "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    )
}

ROAD_GUIDELINE_SHAPES = {
    **{
        normalize_export_label(name): "rectangle"
        for name in (
            "pedestrian", "rider", "car", "truck", "bus", "train",
            "motorcycle", "bicycle", "traffic light", "traffic sign",
        )
    },
    normalize_export_label("area/direct"): "polygon",
    normalize_export_label("area/alternative"): "polygon",
    **{
        normalize_export_label(name): "polyline"
        for name in (
            "lane/crosswalk", "lane/double white", "lane/double yellow",
            "lane/double other", "lane/road curb", "lane/single white",
            "lane/single yellow", "lane/single other",
        )
    },
}


def detect_export_preset(labels: Iterable[str]) -> str:
    """Choose a starting preset from the task's label names."""
    normalized = {normalize_export_label(name) for name in labels}
    road_region_labels = {
        normalize_export_label("area/direct"),
        normalize_export_label("area/alternative"),
        *(normalize_export_label(name) for name in (
            "lane/crosswalk", "lane/double white", "lane/double yellow",
            "lane/double other", "lane/road curb", "lane/single white",
            "lane/single yellow", "lane/single other",
        )),
    }
    if normalized.intersection(road_region_labels):
        return "road"
    if len(normalized.intersection(SEMANTIC_LABELS)) >= 12:
        return "semantic"
    if normalized.intersection(ROAD_GUIDELINE_SHAPES):
        return "road"
    return "auto"


def canonical_cvat_shape_type(value: Any) -> str:
    """Normalize CVAT label types and common aliases to shape API types."""
    value = str(value or "any").strip().casefold().replace("_", "-")
    aliases = {
        "bounding box": "rectangle",
        "bounding-box": "rectangle",
        "bbox": "rectangle",
        "box": "rectangle",
        "keypoints": "skeleton",
        "keypoints / skeleton": "skeleton",
    }
    return aliases.get(value, value)


def guideline_shape_for_label(label_name: str, preset: str) -> str | None:
    """Return a shape explicitly required by a selected guideline, when known."""
    key = normalize_export_label(label_name)
    if preset == "semantic":
        return "polygon" if key in SEMANTIC_LABELS else None
    if preset in ("auto", "road"):
        if key in ROAD_GUIDELINE_SHAPES:
            return ROAD_GUIDELINE_SHAPES[key]
        if key in SEMANTIC_LABELS:
            return "polygon"
    return None


def recommend_export_shape(
    label_name: str,
    *,
    declared_type: Any = "any",
    preset: str = "auto",
    segmentation_available: bool = True,
) -> str:
    """Return a default shape type, respecting any shape type fixed in CVAT."""
    declared = canonical_cvat_shape_type(declared_type)
    if declared not in ("", "any", "tag"):
        return declared

    key = normalize_export_label(label_name)
    if preset == "semantic":
        return "polygon" if segmentation_available else "rectangle"
    if preset == "road":
        shape = ROAD_GUIDELINE_SHAPES.get(key)
        if shape:
            return shape if segmentation_available or shape == "rectangle" else "rectangle"
        if key in SEMANTIC_LABELS:
            return "polygon" if segmentation_available else "rectangle"
    elif preset == "auto":
        shape = ROAD_GUIDELINE_SHAPES.get(key)
        if shape:
            return shape if segmentation_available or shape == "rectangle" else "rectangle"
        if key in SEMANTIC_LABELS:
            return "polygon" if segmentation_available else "rectangle"

    return "mask" if segmentation_available else "rectangle"


def mask_to_cvat_polylines(
    mask: np.ndarray,
    simplify_tolerance: float = 2.0,
    min_path_length: float = 24.0,
) -> list[list[int]]:
    """Convert a thick lane mask to clean, meaningful-length CVAT centerlines."""
    binary = np.ascontiguousarray(np.asarray(mask) > 0, dtype=np.uint8)
    if binary.ndim != 2 or not binary.any():
        return []

    # Join only tiny segmentation gaps, which otherwise become separate CVAT
    # objects along a single painted marking.
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, bridge_kernel)

    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        skeleton = cv2.ximgproc.thinning(binary * 255)
    else:
        skeleton = _morphological_skeleton(binary)
    pixels = {(int(x), int(y)) for y, x in zip(*np.where(skeleton > 0))}
    if len(pixels) < 2:
        return []

    neighbors = {point: _skeleton_neighbors(point, pixels) for point in pixels}
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def edge_key(left: tuple[int, int], right: tuple[int, int]):
        return (left, right) if left <= right else (right, left)

    paths: list[list[tuple[int, int]]] = []
    junctions = sorted(point for point, adjacent in neighbors.items() if len(adjacent) != 2)
    for start in junctions:
        for next_point in neighbors[start]:
            first_edge = edge_key(start, next_point)
            if first_edge in visited:
                continue
            path = [start]
            previous, current = start, next_point
            visited.add(first_edge)
            path.append(current)
            while len(neighbors[current]) == 2:
                following = next(point for point in neighbors[current] if point != previous)
                edge = edge_key(current, following)
                if edge in visited:
                    break
                visited.add(edge)
                previous, current = current, following
                path.append(current)
            if len(path) >= 2:
                paths.append(path)

    # Closed loops have no endpoints or junctions, so walk any remaining edges.
    for start in sorted(pixels):
        for next_point in neighbors[start]:
            if edge_key(start, next_point) in visited:
                continue
            path = [start]
            previous, current = start, next_point
            visited.add(edge_key(start, next_point))
            path.append(current)
            while True:
                candidates = [point for point in neighbors[current] if point != previous]
                if not candidates:
                    break
                following = candidates[0]
                edge = edge_key(current, following)
                if edge in visited:
                    break
                visited.add(edge)
                previous, current = current, following
                path.append(current)
            if len(path) >= 2:
                paths.append(path)

    result: list[list[int]] = []
    for path in paths:
        points = np.asarray(path, dtype=np.float32).reshape(-1, 1, 2)
        path_length = float(np.linalg.norm(np.diff(points[:, 0, :], axis=0), axis=1).sum())
        if path_length < float(min_path_length):
            continue
        simplified = cv2.approxPolyDP(points, float(simplify_tolerance), False).reshape(-1, 2)
        if len(simplified) < 2:
            continue
        flat = [int(round(value)) for value in simplified.reshape(-1)]
        if len(flat) >= 4:
            result.append(flat)
    return result


def _morphological_skeleton(binary: np.ndarray) -> np.ndarray:
    """Extract a one-pixel centerline without requiring OpenCV contrib modules."""
    working = binary.copy()
    skeleton = np.zeros_like(working)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while working.any():
        eroded = cv2.erode(working, kernel)
        opened = cv2.dilate(eroded, kernel)
        skeleton |= cv2.subtract(working, opened)
        if np.array_equal(eroded, working):
            break
        working = eroded
    return skeleton


def _skeleton_neighbors(
    point: tuple[int, int], pixels: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    x, y = point
    result = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not dx and not dy:
                continue
            candidate = (x + dx, y + dy)
            if candidate not in pixels:
                continue
            # Avoid diagonal shortcuts across a corner when an orthogonal step exists.
            if dx and dy and ((x + dx, y) in pixels or (x, y + dy) in pixels):
                continue
            result.append(candidate)
    return sorted(result)
