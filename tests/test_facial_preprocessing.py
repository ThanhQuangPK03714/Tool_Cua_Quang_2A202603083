import numpy as np
import pytest

from src.facial_landmark.coordinates import (
    clamp_face_roi,
    crop_face,
    mirror_point,
    point_from_rotated,
    point_to_rotated,
    rotate_image,
)
from src.facial_landmark.preprocessing import assess_image_quality, make_preprocessing_variants
from src.facial_landmark.preprocessing.variants import apply_clahe, apply_gamma, denoise_image, sharpen_image


def test_quality_assessment_handles_rgb_and_grayscale_appearance():
    gray = np.tile(np.arange(96, dtype=np.uint8), (64, 1))
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    assert "grayscale" in assess_image_quality(gray)["tags"]
    assert "ir_like_grayscale" in assess_image_quality(rgb)["tags"]
    assert assess_image_quality(rgb)["width"] == 96


def test_preprocessing_variants_keep_source_dimensions_and_original():
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[:, 48:] = 30
    variants, quality = make_preprocessing_variants(image)
    assert "Original" in variants
    assert "CLAHE" in variants
    assert all(item.shape == image.shape for item in variants.values())
    assert quality["width"] == image.shape[1]
    assert np.array_equal(variants["Original"], image)


def test_enhancement_parameters_and_color_channels():
    image = np.full((32, 32, 3), 24, dtype=np.uint8)
    assert apply_clahe(image).shape == image.shape
    assert apply_gamma(image, 1.6).mean() > image.mean()
    assert denoise_image(image).shape == image.shape
    assert sharpen_image(image).shape == image.shape
    with pytest.raises(ValueError):
        apply_gamma(image, 0)


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_rotate_and_inverse_point_transforms(angle):
    image = np.arange(4 * 7, dtype=np.uint8).reshape(4, 7)
    rotated = rotate_image(image, angle)
    x, y = 2.0, 1.0
    rx, ry = point_to_rotated(x, y, image.shape[1], image.shape[0], angle)
    back = point_from_rotated(rx, ry, image.shape[1], image.shape[0], angle)
    assert back == (x, y)
    assert rotated[int(ry), int(rx)] == image[int(y), int(x)]


def test_manual_face_roi_is_clamped_and_crop_retains_expected_region():
    image = np.arange(20 * 30, dtype=np.uint8).reshape(20, 30)
    roi = clamp_face_roi(-4, 2, 10, 12, 30, 20, padding=0.1)
    crop, actual = crop_face(image, (roi[0], roi[1], roi[2], roi[3]), padding=0)
    assert actual == roi
    assert crop.shape == (roi[3], roi[2])
    assert mirror_point(2, 3, 10) == (7.0, 3.0)
