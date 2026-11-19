"""Conversions between binary masks and CVAT REST shape coordinates."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def mask_to_cvat_points(mask: np.ndarray) -> list[int]:
    """Encode a binary bitmap with CVAT SDK's canonical mask-RLE format."""
    bitmap = np.ascontiguousarray(np.asarray(mask) > 0)
    if bitmap.ndim != 2:
        raise ValueError("A CVAT mask must be a two-dimensional array.")
    if not bitmap.any():
        return []
    from cvat_sdk.masks import encode_mask

    return [int(value) for value in encode_mask(bitmap)]


def draw_cvat_mask(canvas: np.ndarray, points: Sequence[Any]) -> None:
    """Decode CVAT's flat RLE+bounding-box point list into ``canvas`` in place."""
    try:
        left, top, right, bottom = [int(round(float(value))) for value in points[-4:]]
        counts = [max(0, int(round(float(value)))) for value in points[:-4]]
    except (TypeError, ValueError):
        return
    crop_width = right - left + 1
    crop_height = bottom - top + 1
    if crop_width <= 0 or crop_height <= 0 or not counts:
        return

    flat = np.zeros(crop_width * crop_height, dtype=np.uint8)
    offset = 0
    foreground = False
    for count in counts:
        end = min(offset + count, flat.size)
        if foreground:
            flat[offset:end] = 1
        offset = end
        foreground = not foreground
        if offset >= flat.size:
            break
    crop = flat.reshape(crop_height, crop_width)
    dst_x1, dst_y1 = max(0, left), max(0, top)
    dst_x2, dst_y2 = min(canvas.shape[1], right + 1), min(canvas.shape[0], bottom + 1)
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return
    src = crop[dst_y1 - top : dst_y2 - top, dst_x1 - left : dst_x2 - left]
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = np.maximum(canvas[dst_y1:dst_y2, dst_x1:dst_x2], src)


def keypoints_to_cvat_skeleton(
    keypoints: Sequence[dict[str, Any]],
    *,
    frame: int,
    label_id: int,
    sublabels: Sequence[dict[str, Any]],
    confidence_threshold: float = 0.25,
) -> dict[str, Any] | None:
    """Build a CVAT skeleton shape using only matching model/CVAT keypoints."""
    sublabel_ids = {
        str(label["name"]).strip().replace("_", " ").casefold(): int(label["id"])
        for label in sublabels
        if label.get("id") is not None and label.get("name")
    }
    elements: list[dict[str, Any]] = []
    for point in keypoints:
        name = str(point.get("name", "")).strip().replace("_", " ").casefold()
        child_label_id = sublabel_ids.get(name)
        if child_label_id is None:
            continue
        confidence = point.get("confidence")
        if confidence is not None and float(confidence) < confidence_threshold:
            continue
        x, y = float(point["x"]), float(point["y"])
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        elements.append({
            "type": "points",
            "frame": int(frame),
            "label_id": child_label_id,
            "group": 0,
            "source": "auto",
            "occluded": False,
            "outside": False,
            "z_order": 0,
            "rotation": 0,
            "points": [x, y],
            "attributes": [],
            "elements": [],
        })
    if not elements:
        return None
    return {
        "type": "skeleton",
        "frame": int(frame),
        "label_id": int(label_id),
        "group": 0,
        "source": "auto",
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0,
        "points": [],
        "attributes": [],
        "elements": elements,
    }


def get_cvat_sublabels(
    labels: Sequence[dict[str, Any]],
    parent_label: dict[str, Any],
) -> list[dict[str, Any]]:
    """Read skeleton children from either nested or flat CVAT label payloads."""
    nested = parent_label.get("sublabels")
    if isinstance(nested, list) and nested:
        return nested
    parent_id = parent_label.get("id")
    parent_name = str(parent_label.get("name", "")).casefold()
    result: list[dict[str, Any]] = []
    for label in labels:
        reference = label.get("parent_id", label.get("parent"))
        if isinstance(reference, dict):
            matches = reference.get("id") == parent_id or str(reference.get("name", "")).casefold() == parent_name
        else:
            matches = reference == parent_id or str(reference).casefold() == parent_name
        if matches and label.get("id") is not None:
            result.append(label)
    return result
