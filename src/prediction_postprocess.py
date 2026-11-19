"""Reduce tiny mask fragments and consolidate road-marking predictions."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def postprocess_prediction_masks(
    masks: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    *,
    min_component_area_px: int = 96,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Remove tiny components and merge same-label, untracked lane fragments.

    Instance objects remain separate. Lane predictions without track IDs are
    unioned by label so one review item can cover the whole marking class; the
    polyline exporter later keeps only meaningful-length centerline segments.
    """
    threshold = max(1, int(min_component_area_px))
    height, width = int(image_shape[0]), int(image_shape[1])
    lane_groups: Dict[str, List[Dict[str, Any]]] = {}
    lane_first_index: Dict[str, int] = {}
    retained: List[Tuple[int, Dict[str, Any]]] = []
    stats = {
        "input_masks": len(masks),
        "removed_masks": 0,
        "removed_components": 0,
        "merged_lane_masks": 0,
    }

    for index, item in enumerate(masks):
        raw_mask = np.asarray(item.get("mask", []))
        segmentation_available = bool(item.get("segmentation_available", True))
        class_name = str(item.get("class_name", ""))
        is_untracked_lane = (
            class_name.strip().casefold().startswith("lane/")
            and item.get("track_id") is None
            and segmentation_available
            and raw_mask.ndim == 2
            and raw_mask.shape == (height, width)
        )
        if is_untracked_lane:
            key = class_name.strip().casefold()
            lane_groups.setdefault(key, []).append(item)
            lane_first_index.setdefault(key, index)
            continue

        if not segmentation_available or raw_mask.ndim != 2 or raw_mask.shape != (height, width):
            retained.append((index, item))
            continue

        binary = np.ascontiguousarray(raw_mask > 0, dtype=np.uint8)
        if int(binary.sum()) < threshold:
            stats["removed_masks"] += 1
            continue
        cleaned, removed = _remove_small_components(binary, threshold)
        stats["removed_components"] += removed
        area = int(cleaned.sum())
        if area < threshold:
            stats["removed_masks"] += 1
            continue
        cleaned_item = {**item, "mask": cleaned, "area": area, "bbox": _mask_bbox(cleaned)}
        retained.append((index, cleaned_item))

    for key, items in lane_groups.items():
        merged = np.zeros((height, width), dtype=np.uint8)
        confidence_area: List[Tuple[float, int]] = []
        for item in items:
            source = np.asarray(item["mask"]) > 0
            merged[source] = 1
            area = int(source.sum())
            try:
                confidence = float(item.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            confidence_area.append((confidence, area))

        # Bridge only hairline gaps; this absorbs detector speckle without
        # joining visibly separate parallel road markings.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)
        merged, removed = _remove_small_components(merged, threshold)
        stats["removed_components"] += removed
        area = int(merged.sum())
        if area < threshold:
            stats["removed_masks"] += len(items)
            continue

        total_confidence_area = sum(area_value for _, area_value in confidence_area)
        confidence = (
            sum(score * area_value for score, area_value in confidence_area)
            / max(total_confidence_area, 1)
        )
        combined = {
            **items[0],
            "mask": merged,
            "area": area,
            "bbox": _mask_bbox(merged),
            "confidence": confidence,
            "is_thing": False,
            "merged_prediction_count": len(items),
        }
        retained.append((lane_first_index[key], combined))
        stats["merged_lane_masks"] += max(0, len(items) - 1)

    retained.sort(key=lambda entry: entry[0])
    output = [item for _, item in retained]
    stats["output_masks"] = len(output)
    return output, stats


def _remove_small_components(mask: np.ndarray, min_area_px: int) -> Tuple[np.ndarray, int]:
    binary = np.ascontiguousarray(np.asarray(mask) > 0, dtype=np.uint8)
    ys, xs = np.where(binary > 0)
    if not len(xs):
        return np.zeros_like(binary), 0
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = binary[y1:y2, x1:x2]
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    cleaned_crop = np.zeros_like(crop)
    removed = 0
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            cleaned_crop[labels == component_id] = 1
        else:
            removed += 1
    cleaned = np.zeros_like(binary)
    cleaned[y1:y2, x1:x2] = cleaned_crop
    return cleaned, removed


def _mask_bbox(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
