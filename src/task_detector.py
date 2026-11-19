"""
Phát hiện loại segmentation dựa trên tên task và danh sách label từ CVAT.

Không được mặc định task nào. Phải kiểm tra tên task + class labels trước khi
lựa chọn mô hình.
"""

from __future__ import annotations

from typing import List, Dict, Any

# Tên class chuẩn theo bài segmentation (không được suy đoán thêm).
SEMANTIC_CLASSES = {"road", "sidewalk", "building", "vegetation", "sky"}
INSTANCE_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
PANOPTIC_CLASSES = {
    "road", "sidewalk", "building", "vegetation", "sky",
    "person", "car", "bus", "truck", "motorcycle", "bicycle", "traffic light",
}
PANOPTIC_STUFF = {"road", "sidewalk", "building", "vegetation", "sky"}
PANOPTIC_THINGS = {"person", "car", "bus", "truck", "motorcycle", "bicycle", "traffic light"}

# Checkpoint dùng class list riêng; nhận diện bằng tên task trước khi xét labels.
CHECKPOINT_TYPES = {
    "cp1_holes": "instance",
    "cp2_slice": "instance",
    "cp3_thin": "semantic",
    "cp4_curb": "semantic",
    "cp5_occlusion": "instance",
    "cp6_coverage": "semantic",
}


def detect_segmentation_type(task_name: str, labels: List[Dict[str, Any]]) -> str:
    """
    Phát hiện loại segmentation từ tên task và labels.

    Trả về một trong: 'semantic', 'instance', 'panoptic', 'unknown'.
    """
    label_names = {lbl.get("name", "").strip().lower() for lbl in labels}
    task_lower = task_name.strip().lower()

    if task_lower in CHECKPOINT_TYPES:
        return CHECKPOINT_TYPES[task_lower]

    # Dựa trên tên task (manifest)
    if "panoptic" in task_lower or "hard" in task_lower:
        if PANOPTIC_CLASSES.issubset(label_names):
            return "panoptic"
        return "panoptic"

    if "semantic" in task_lower or "easy" in task_lower:
        if SEMANTIC_CLASSES.issubset(label_names):
            return "semantic"
        return "semantic"

    if "instance" in task_lower or "medium" in task_lower:
        return "instance"

    # Dựa trên labels nếu tên task không rõ ràng
    if PANOPTIC_CLASSES.issubset(label_names):
        return "panoptic"
    if SEMANTIC_CLASSES.issubset(label_names) and not INSTANCE_CLASSES.intersection(label_names):
        return "semantic"
    if INSTANCE_CLASSES.issubset(label_names) and not SEMANTIC_CLASSES.intersection(label_names):
        return "instance"

    return "unknown"


def get_stuff_classes(label_names: List[str]) -> List[str]:
    """Trả về danh sách classStuff (vùng, không đếm instance)."""
    stuff = {"road", "sidewalk", "building", "vegetation", "sky"}
    return [c for c in label_names if c.lower() in stuff]


def get_thing_classes(label_names: List[str]) -> List[str]:
    """Trả về danh sách classThing (vật đếm được)."""
    things = {"person", "car", "bus", "truck", "motorcycle", "bicycle", "traffic light"}
    return [c for c in label_names if c.lower() in things]


def is_checkpoint_task(task_name: str) -> bool:
    """Kiểm tra task có phải checkpoint (cp1..cp6) không."""
    import re
    return bool(re.match(r"^cp[1-6]_", task_name.strip().lower()))
