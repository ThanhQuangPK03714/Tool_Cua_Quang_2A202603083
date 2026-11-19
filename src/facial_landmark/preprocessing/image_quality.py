"""Image quality signals used to suggest, not force, enhancement choices."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    raise ValueError("Expected a grayscale or RGB image")


def assess_image_quality(image: np.ndarray, face_bbox: tuple[int, int, int, int] | None = None) -> dict[str, Any]:
    """Return reproducible image signals and cautious condition tags.

    ``ir_like_grayscale`` only describes appearance. It does not identify the
    sensor or assert that a grayscale image is infrared.
    """
    if image.size == 0:
        raise ValueError("Image is empty")
    gray = _gray(np.asarray(image)).astype(np.uint8, copy=False)
    pixels = gray.astype(np.float32)
    brightness = float(np.mean(pixels))
    contrast = float(np.std(pixels))
    dynamic_range = float(np.percentile(pixels, 99) - np.percentile(pixels, 1))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    high_frequency = pixels - cv2.GaussianBlur(pixels, (0, 0), 1.0)
    noise_estimate = float(np.median(np.abs(high_frequency - np.median(high_frequency))) / 0.6745)
    grayscale = image.ndim == 2 or image.ndim == 3 and image.shape[2] == 1
    if not grayscale and image.ndim == 3 and image.shape[2] >= 3:
        grayscale = bool(np.max(np.abs(image[:, :, :3].astype(np.int16) - image[:, :, :3].mean(axis=2, keepdims=True))) < 2)
    shadow_fraction = float(np.mean(pixels < 32))
    highlight_fraction = float(np.mean(pixels > 245))
    tags: list[str] = []
    if grayscale:
        tags.extend(("grayscale", "ir_like_grayscale"))
    if brightness < 65:
        tags.append("low_light")
    if contrast < 35 or dynamic_range < 90:
        tags.append("low_contrast")
    if blur_score < 35:
        tags.append("possible_blur")
    if noise_estimate > 12:
        tags.append("high_noise")
    if shadow_fraction > 0.6:
        tags.append("underexposed")
    if highlight_fraction > 0.25:
        tags.append("overexposed")
    width, height = int(image.shape[1]), int(image.shape[0])
    face_size_fraction = None
    if face_bbox is not None:
        _, _, face_width, face_height = face_bbox
        face_size_fraction = max(0.0, float(face_width * face_height) / max(1, width * height))
        if face_size_fraction < 0.035:
            tags.append("small_face")
    if not tags:
        tags.append("normal_rgb")
    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2),
        "noise_estimate": round(noise_estimate, 2),
        "dynamic_range_p1_p99": round(dynamic_range, 2),
        "underexposed_fraction": round(shadow_fraction, 4),
        "overexposed_fraction": round(highlight_fraction, 4),
        "is_grayscale_appearance": bool(grayscale),
        "face_size_fraction": None if face_size_fraction is None else round(face_size_fraction, 5),
        "tags": tags,
    }
