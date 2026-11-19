"""
Export dataset từ CVAT annotation sang định dạng ZIP (Segmentation mask 1.1 / COCO 1.0).

Tuân thủ quy định của repository Day 3:
- Semantic: Segmentation mask 1.1 (labelmap.txt + SegmentationClass/*.png)
- Instance/Panoptic: COCO 1.0 (annotations/instances_default.json)
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Semantic mask 1.1 export
# ---------------------------------------------------------------------- #
def export_semantic_mask_1_1(
    outputs: Dict[str, np.ndarray],
    class_colors: Dict[str, Tuple[int, int, int]],
    class_names: List[str],
    output_zip: Optional[str] = None,
) -> bytes:
    """
    Tạo ZIP Segmentation mask 1.1 từ dict {image_stem: label_map}.

    parameters
    ----------
    outputs : {stem: H×W int32 label map}
    class_colors : {class_name: (r, g, b)}
    class_names : list of class names
    output_zip : nếu cung cấp, ghi ra file zip này

    Returns bytes của ZIP.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # labelmap.txt
        lines = []
        for cls in class_names:
            r, g, b = class_colors.get(cls, (0, 0, 0))
            lines.append(f"{cls}:{r},{g},{b}::")
        zf.writestr("labelmap.txt", "\n".join(lines) + "\n")

        # SegmentationClass/<stem>.png
        for stem, label_map in outputs.items():
            h, w = label_map.shape
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for class_name in class_names:
                class_id = class_names.index(class_name)
                if class_name in class_colors:
                    mask = label_map == class_id
                    rgb[mask] = class_colors[class_name]
            img = Image.fromarray(rgb, mode="RGB")
            img_buf = io.BytesIO()
            img.save(img_buf, format="PNG")
            zf.writestr(f"SegmentationClass/{stem}.png", img_buf.getvalue())

    data = buf.getvalue()
    if output_zip:
        Path(output_zip).write_bytes(data)
        logger.info("Đã xuất semantic mask 1.1: %s", output_zip)
    return data


# ---------------------------------------------------------------------- #
# COCO 1.0 export
# ---------------------------------------------------------------------- #
def export_coco_1_0(
    images: List[dict],
    categories: List[dict],
    annotations: List[dict],
    output_zip: Optional[str] = None,
) -> bytes:
    """
    Tạo ZIP COCO 1.0 từ images, categories, annotations.

    parameters
    ----------
    images: list of {id, file_name, width, height}
    categories: list of {id, name, supercategory}
    annotations: list of COCO annotation dicts
    """
    coco = {
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("annotations/instances_default.json", json.dumps(coco, ensure_ascii=False, indent=2))

    data = buf.getvalue()
    if output_zip:
        Path(output_zip).write_bytes(data)
        logger.info("Đã xuất COCO 1.0: %s", output_zip)
    return data


# ---------------------------------------------------------------------- #
# QC structural check (dùng trước khi nộp)
# ---------------------------------------------------------------------- #
def validate_zip_structure(zip_path: str, task_type: str, expected_images: List[str], expected_classes: List[str]) -> dict:
    """
    Kiểm tra cấu trúc ZIP trước khi nộp bài.
    Trả về dict {errors: [], warnings: []}.
    """
    errors: List[str] = []
    warnings: List[str] = []

    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
    except Exception as exc:
        return {"errors": [f"Không đọc được ZIP: {exc}"], "warnings": []}

    if task_type == "semantic":
        if "labelmap.txt" not in names:
            errors.append("Thiếu labelmap.txt")
        mask_files = [n for n in names if "SegmentationClass/" in n and n.endswith(".png")]
        if not mask_files:
            errors.append("Thiếu file PNG trong SegmentationClass/")
        # Check image names match
        mask_stems = {Path(n).stem for n in mask_files}
        expected_stems = {Path(n).stem for n in expected_images}
        missing = expected_stems - mask_stems
        extra = mask_stems - expected_stems
        if missing:
            errors.append(f"Thiếu mask: {sorted(missing)}")
        if extra:
            errors.append(f"Mask không thuộc task: {sorted(extra)}")

    elif task_type in ("instance", "panoptic"):
        coco_files = [n for n in names if n.endswith("instances_default.json")]
        if not coco_files:
            errors.append("Thiếu annotations/instances_default.json")

    return {"errors": errors, "warnings": warnings}
