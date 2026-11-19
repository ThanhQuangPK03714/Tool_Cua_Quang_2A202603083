"""
Mask post-processing: lọc bruit, khử lỗ, đóng mở morphology, tách instance dính nhau.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def fill_holes(mask: np.ndarray, min_hole_area: int = 100) -> np.ndarray:
    """Khử lỗ nhỏ bên trong mask (ví dụ kính xe trong mask xe)."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (1 - mask).astype(np.uint8), connectivity=8
    )
    filled = mask.copy()
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_hole_area:
            filled[labels == i] = 1
    return filled


def morphological_clean(
    mask: np.ndarray,
    open_kernel: int = 3,
    close_kernel: int = 5,
) -> np.ndarray:
    """Lọc nhiễu nhỏ (open) rồi khứu khuyết tách rời (close)."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    result = mask.copy()
    if open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Chỉ giữ thành phần liên kết lớn nhất (dùng cho stuff region)."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def split_instances(mask: np.ndarray, min_area: int = 50) -> List[np.ndarray]:
    """Tách mask nhị phân thành các instance riêng (connected components)."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    instances = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            instances.append((labels == i).astype(np.uint8))
    return instances


def resize_mask(
    mask: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """Resize mask về kích thước mục tiêu (width, height)."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return cv2.resize(mask, target_size, interpolation=interpolation)


def mask_to_polygons(
    mask: np.ndarray,
    simplify_tolerance: float = 1.5,
    min_contour_area_px: float = 32.0,
) -> List[List[float]]:
    """
    Chuyển mask nhị phân thành danh sách polygon (COCO segmentation format).
    Trả về list các polygon, mỗi polygon là list [x1, y1, x2, y2, ...].
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3 or cv2.contourArea(contour) < float(min_contour_area_px):
            continue
        contour = cv2.approxPolyDP(contour, simplify_tolerance, True)
        pts = contour.reshape(-1, 2)
        if len(pts) < 3:
            continue
        poly = []
        for x, y in pts:
            poly.extend([float(x), float(y)])
        polygons.append(poly)
    return polygons


def mask_to_rle(mask: np.ndarray) -> dict:
    """Chuyển mask nhị phân thành COCO RLE format."""
    from pycocotools import mask as cocomask

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # np.asfortranarray required by pycocotools
    rle = cocomask.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle
