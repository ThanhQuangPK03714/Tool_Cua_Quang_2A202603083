"""
Model Router: chọn mô hình AI phù hợp cho từng loại segmentation.

Ưu tiên mô hình có trọng số pretrained phù hợp với ảnh giao thông đường phố.
Chạy được trên CPU (môi trường lab không có GPU).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SegmentationType(Enum):
    SEMANTIC = "semantic"
    INSTANCE = "instance"
    PANOPTIC = "panoptic"


@dataclass
class ModelConfig:
    segmentation_type: SegmentationType
    model_name: str
    model_class: str  # tên lớp Python để khởi tạo
    weights: Optional[str] = None
    input_size: tuple = (512, 512)
    confidence_threshold: float = 0.25
    device: str = "auto"
    description: str = ""


# Bản đồ mô hình khuyến nghị cho từng loại segmentation
MODEL_REGISTRY = {
    SegmentationType.SEMANTIC: ModelConfig(
        segmentation_type=SegmentationType.SEMANTIC,
        model_name="segformer_cityscapes",
        model_class="CityscapesSegformer",
        weights=None,  # sẽ tải từ URL công khai
        input_size=(512, 512),
        confidence_threshold=0.25,
        device="auto",
        description="SegFormer B0 pretrained trên Cityscapes (19 lớp; lọc theo labels của task)",
    ),
    SegmentationType.INSTANCE: ModelConfig(
        segmentation_type=SegmentationType.INSTANCE,
        model_name="yolov8n-seg",
        model_class="YOLOv8Seg",
        weights="yolov8n-seg.pt",
        input_size=(640, 640),
        confidence_threshold=0.25,
        device="auto",
        description="YOLOv8n-seg pretrained trên COCO (các thing class của bài, gồm traffic light)",
    ),
    SegmentationType.PANOPTIC: ModelConfig(
        segmentation_type=SegmentationType.PANOPTIC,
        model_name="combined_panoptic",
        model_class="PanopticSegmenter",
        weights=None,
        input_size=(512, 512),
        confidence_threshold=0.25,
        device="auto",
        description="Kết hợp SegFormer Cityscapes (stuff) + YOLOv8-seg (things) cho panoptic",
    ),
}


def get_model_config(seg_type: SegmentationType) -> ModelConfig:
    """Lấy cấu hình mô hình cho loại segmentation."""
    return MODEL_REGISTRY[seg_type]


def describe_model(seg_type: SegmentationType) -> str:
    """Mô tả mô hình sẽ dùng cho loại segmentation."""
    cfg = MODEL_REGISTRY[seg_type]
    return f"{cfg.model_name} ({cfg.description})"
