"""Image-space transforms for rotating views and face crops."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def rotate_image(image: np.ndarray, degrees_clockwise: int) -> np.ndarray:
    angle = int(degrees_clockwise) % 360
    if angle not in (0, 90, 180, 270):
        raise ValueError("Only 0, 90, 180, and 270 degree rotations are supported")
    turns = (-angle // 90) % 4
    return np.ascontiguousarray(np.rot90(image, k=turns))


def point_to_rotated(x: float, y: float, width: int, height: int, degrees_clockwise: int) -> tuple[float, float]:
    angle = int(degrees_clockwise) % 360
    if angle == 0:
        return float(x), float(y)
    if angle == 90:
        return float(height - 1 - y), float(x)
    if angle == 180:
        return float(width - 1 - x), float(height - 1 - y)
    if angle == 270:
        return float(y), float(width - 1 - x)
    raise ValueError("Only 0, 90, 180, and 270 degree rotations are supported")


def point_from_rotated(x: float, y: float, width: int, height: int, degrees_clockwise: int) -> tuple[float, float]:
    angle = int(degrees_clockwise) % 360
    if angle == 0:
        return float(x), float(y)
    if angle == 90:
        return float(y), float(height - 1 - x)
    if angle == 180:
        return float(width - 1 - x), float(height - 1 - y)
    if angle == 270:
        return float(width - 1 - y), float(x)
    raise ValueError("Only 0, 90, 180, and 270 degree rotations are supported")


def mirror_point(x: float, y: float, width: int) -> tuple[float, float]:
    return float(width - 1 - x), float(y)


def clamp_face_roi(
    x0: float, y0: float, x1: float, y1: float, width: int, height: int, *, padding: float = 0.0
) -> tuple[int, int, int, int]:
    """Return a clamped ``(x, y, width, height)`` ROI with optional fractional padding."""
    if width < 1 or height < 1 or padding < 0:
        raise ValueError("Image dimensions must be positive and padding non-negative")
    left, right = sorted((float(x0), float(x1)))
    top, bottom = sorted((float(y0), float(y1)))
    if right <= left or bottom <= top:
        raise ValueError("Face ROI must have positive width and height")
    padx, pady = (right - left) * padding, (bottom - top) * padding
    left = max(0, int(np.floor(left - padx)))
    top = max(0, int(np.floor(top - pady)))
    right = min(width, int(np.ceil(right + padx)))
    bottom = min(height, int(np.ceil(bottom + pady)))
    return left, top, max(1, right - left), max(1, bottom - top)


def crop_face(image: np.ndarray, roi: Sequence[float], *, padding: float = 0.15) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if len(roi) != 4:
        raise ValueError("ROI must be (x, y, width, height)")
    x, y, width, height = roi
    image_height, image_width = image.shape[:2]
    clamped = clamp_face_roi(x, y, x + width, y + height, image_width, image_height, padding=padding)
    left, top, crop_width, crop_height = clamped
    return image[top:top + crop_height, left:left + crop_width].copy(), clamped
