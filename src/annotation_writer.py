"""
Chuyển đổi mask sang định dạng annotation của CVAT và ghi qua API.

Hỗ trợ:
- COCO 1.0 (instance, panoptic) - RLE + polygon
- Segmentation mask 1.1 (semantic) - color PNG + labelmap.txt
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.mask_processor import mask_to_polygons, mask_to_rle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# COCO 1.0 helpers
# ---------------------------------------------------------------------- #
def build_coco_annotation(
    annotation_id: int,
    image_id: int,
    category_id: int,
    mask: np.ndarray,
    confidence: float = 1.0,
    use_rle: bool = True,
) -> dict:
    """Tạo một annotation COCO 1.0 từ mask nhị phân."""
    h, w = mask.shape
    if use_rle:
        rle = mask_to_rle(mask)
        segmentation: Any = rle
    else:
        segmentation = mask_to_polygons(mask)

    area = int(mask.sum())
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        bbox = [0, 0, 0, 0]
    else:
        bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())]

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
        "attributes": {"confidence": confidence},
        "metadata": {"confidence": confidence},
        "group": 0,
        "frame": 0,
        "source": "auto",
    }


def build_coco_payload(
    images: List[dict],
    categories: List[dict],
    annotations: List[dict],
) -> dict:
    """Gói toàn bộ thành payload COCO 1.0 cho CVAT."""
    return {
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }


# ---------------------------------------------------------------------- #
# Segmentation mask 1.1 helpers
# ---------------------------------------------------------------------- #
def build_labelmap(classes: List[str], colors: Optional[List[Tuple[int, int, int]]] = None) -> str:
    """Tạo nội dung labelmap.txt cho Segmentation mask 1.1."""
    lines = []
    if colors is None:
        colors = [(0, 0, 0)] * len(classes)
    for cls, rgb in zip(classes, colors):
        lines.append(f"{cls}:{rgb[0]},{rgb[1]},{rgb[2]}::")
    return "\n".join(lines)


def semantic_mask_to_png(
    label_map: np.ndarray,
    class_colors: Dict[str, Tuple[int, int, int]],
    class_names: List[str],
) -> Image.Image:
    """
    Chuyển label_map (H×W int32, class id) thành ảnh PNG màu (Segmentation mask 1.1).

    Mỗi pixel được tô màu theo class.
    """
    h, w = label_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_name, class_id in {
        name: i for i, name in enumerate(class_names)
    }.items():
        if class_name in class_colors:
            r, g, b = class_colors[class_name]
            mask = label_map == class_id
            rgb[mask] = [r, g, b]
    return Image.fromarray(rgb, mode="RGB")


# ---------------------------------------------------------------------- #
# CVAT annotation upload
# ---------------------------------------------------------------------- #
def annotations_to_cvat_job_format(
    annotations: List[dict],
    image_width: int,
    image_height: int,
) -> dict:
    """
    Chuyển danh sách annotation sang định dạng CVAT job annotations (JSON cho PUT/PATCH).

    Mỗi annotation: {
        "id": int,
        "label_id": int,
        "type": "mask",  # CVAT dùng 'mask' cho segmentation
        "points": [...],  # polygon points hoặc RLE
        "occluded": false,
        "outside": false,
        "z_order": 0,
        "frame": 0,
        "group": 0,
        "attributes": [...],
    }
    """
    cvat_shapes = []
    for ann in annotations:
        mask = ann["mask"]
        label_id = ann["label_id"]
        confidence = ann.get("confidence", 1.0)

        # CVAT mask format sử dụng RLE hoặc polygon
        rle = mask_to_rle(mask)

        cvat_shapes.append({
            "id": ann.get("id", 0),
            "label_id": label_id,
            "type": "mask",
            "points": [],  # mask không dùng points, dùng rle trong data
            "occluded": False,
            "outside": False,
            "z_order": 0,
            "frame": 0,
            "group": 0,
            "attributes": [
                {"value": str(confidence), "spec_id": None}  # confidence as attribute
            ],
            "data": {
                "rle": rle,
            },
        })

    return {
        "version": 1,
        "tags": [],
        "shapes": cvat_shapes,
        "tracks": [],
        "intervals": [],
    }
