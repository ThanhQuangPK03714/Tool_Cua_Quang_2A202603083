"""Dimension-preserving variants for preview and optional inference retries."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .image_quality import assess_image_quality


def _as_u8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array.copy()
    return np.clip(array, 0, 255).astype(np.uint8)


def _luma_transform(image: np.ndarray, transform) -> np.ndarray:
    src = _as_u8(image)
    if src.ndim == 2:
        return transform(src)
    if src.ndim != 3 or src.shape[2] < 3:
        raise ValueError("Expected a grayscale or RGB image")
    yuv = cv2.cvtColor(src[:, :, :3], cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = transform(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    if clip_limit <= 0 or tile_grid_size < 2:
        raise ValueError("CLAHE clip limit must be positive and tile size at least 2")
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    return _luma_transform(image, clahe.apply)


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("Gamma must be greater than zero")
    table = np.clip(((np.arange(256) / 255.0) ** (1.0 / float(gamma))) * 255.0, 0, 255).astype(np.uint8)
    return _luma_transform(image, lambda channel: cv2.LUT(channel, table))


def denoise_image(image: np.ndarray, method: str = "bilateral", strength: int = 5) -> np.ndarray:
    if strength < 1 or strength > 20:
        raise ValueError("Denoising strength must be between 1 and 20")
    src = _as_u8(image)
    if method == "bilateral":
        return cv2.bilateralFilter(src, d=5, sigmaColor=float(strength * 4), sigmaSpace=float(strength * 3))
    if method == "nlmeans":
        if src.ndim == 2:
            return cv2.fastNlMeansDenoising(src, None, float(strength), 7, 21)
        return cv2.fastNlMeansDenoisingColored(src[:, :, :3], None, float(strength), float(strength), 7, 21)
    if method == "gaussian":
        return cv2.GaussianBlur(src, (3, 3), sigmaX=min(0.8, strength / 20))
    raise ValueError(f"Unknown denoising method: {method}")


def sharpen_image(image: np.ndarray, amount: float = 0.35) -> np.ndarray:
    if amount < 0 or amount > 1:
        raise ValueError("Sharpen amount must be between 0 and 1")
    src = _as_u8(image)
    blurred = cv2.GaussianBlur(src, (0, 0), 1.0)
    return cv2.addWeighted(src, 1.0 + amount, blurred, -amount, 0)


def make_preprocessing_variants(
    image: np.ndarray,
    *,
    clahe_enabled: bool = True,
    gamma_values: tuple[float, ...] = (1.25, 1.6),
    gamma_enabled: bool | None = None,
    denoise_enabled: bool = True,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
    denoise_strength: int = 5,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create conservative alternatives while keeping every output size fixed."""
    source = _as_u8(image)
    quality = assess_image_quality(source)
    variants = {"Original": source}
    should_raise_gamma = "low_light" in quality["tags"]
    clahe_image = None
    if clahe_enabled:
        clahe_image = apply_clahe(source, clip_limit, tile_grid_size)
        variants["CLAHE"] = clahe_image
    if (should_raise_gamma if gamma_enabled is None else gamma_enabled):
        for gamma in gamma_values:
            variants[f"Gamma {gamma:g}"] = apply_gamma(source, gamma)
    denoised = denoise_image(source, "bilateral", denoise_strength) if denoise_enabled else None
    if denoised is not None:
        variants["Denoised"] = denoised
    if clahe_image is not None and denoised is not None:
        variants["CLAHE + denoise"] = denoise_image(clahe_image, "bilateral", denoise_strength)
    return variants, quality
