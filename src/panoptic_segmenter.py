"""
Panoptic Segmentation: kết hợp Semantic (stuff) + Instance (things).

Stuff: road, sidewalk, building, vegetation, sky
Things: person, car, bus, truck, motorcycle, bicycle, traffic light
Đầu ra: panoptic map (mỗi pixel có segment_id) + danh sách segment (id, class_name, is_thing, instance_id).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.instance_segmenter import YOLOv8Seg
from src.semantic_segmenter import CITYSCAPES_TRAINID_TO_CLASS, CityscapesSegformer
from src.device_utils import resolve_device

logger = logging.getLogger(__name__)

CITYSCAPES_CLASSES = CITYSCAPES_TRAINID_TO_CLASS
CITYSCAPES_THING_CLASSES = {
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
}
STUFF_CLASSES = set(CITYSCAPES_CLASSES.values()) - CITYSCAPES_THING_CLASSES
THING_CLASSES = CITYSCAPES_THING_CLASSES | {"traffic light"}


class PanopticSegmenter:
    """Kết hợp SegFormer Cityscapes (stuff) và YOLOv8-seg (things)."""

    def __init__(
        self,
        device: str = "auto",
        semantic_input_size: Tuple[int, int] = (512, 512),
        instance_input_size: Tuple[int, int] = (640, 640),
        semantic_conf: float = 0.25,
        instance_conf: float = 0.25,
        instance_iou: float = 0.45,
        instance_weights: str = "yolov8n-seg.pt",
        semantic_model: Optional[CityscapesSegformer] = None,
        instance_model: Optional[YOLOv8Seg] = None,
    ):
        self.device = resolve_device(device)
        self.semantic = (
            semantic_model
            if semantic_model is not None
            else CityscapesSegformer(
                device=self.device,
                input_size=semantic_input_size,
                confidence_threshold=semantic_conf,
            )
        )
        self.instance = (
            instance_model
            if instance_model is not None
            else YOLOv8Seg(
                device=self.device,
                input_size=instance_input_size,
                confidence_threshold=instance_conf,
                iou_threshold=instance_iou,
                weights=instance_weights,
            )
        )
        logger.info("Panoptic segmenter initialized (stuff + things).")

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Dự đoán panoptic segmentation.

        Returns
        -------
        dict: {
            "panoptic_map": np.ndarray (H×W int32), mỗi pixel là segment_id (0 = background/unknown),
            "segments": list of dict {
                "id": int,
                "class_name": str,
                "is_thing": bool,
                "instance_id": int or None,
                "area": int,
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
            },
            "label_map": np.ndarray (H×W int32), class id cho mỗi pixel (255 = unknown),
            "semantic_confidence": np.ndarray (H×W float32),
        }
        """
        h, w = image.shape[:2]

        # 1. Semantic segmentation over all Cityscapes classes.
        label_map, sem_conf = self.semantic.predict(
            image,
            class_names=list(CITYSCAPES_CLASSES.values()),
            confidence_threshold=confidence_threshold,
        )

        # 2. Instance segmentation (things)
        instances = self.instance.predict(image, confidence_threshold=confidence_threshold)

        # 3. Build panoptic map
        panoptic_map = np.full((h, w), 0, dtype=np.int32)  # 0 = unknown/background
        segments: List[Dict] = []
        segment_id = 1

        # 3a. Gán things trước để object masks không bị semantic stuff che mất.
        thing_instance_counts: Dict[str, int] = {}
        for inst in instances:
            class_name = inst["class_name"]
            if class_name not in THING_CLASSES:
                continue
            mask = inst["mask"] > 0
            unassigned_thing = (panoptic_map == 0) & (mask == 1)
            if unassigned_thing.sum() == 0:
                continue
            panoptic_map[unassigned_thing] = segment_id
            thing_instance_counts[class_name] = thing_instance_counts.get(class_name, 0) + 1
            segments.append({
                "id": segment_id,
                "class_name": class_name,
                "is_thing": True,
                "instance_id": thing_instance_counts[class_name],
                "area": int(unassigned_thing.sum()),
                "bbox": self._mask_bbox(unassigned_thing),
                "confidence": inst["confidence"],
                "source": "instance",
            })
            segment_id += 1

        # 3b. Add remaining semantic regions, including Cityscapes classes that
        # are absent from the default YOLO instance model (for example rider,
        # train, wall, fence, pole, traffic sign, and terrain).
        for class_id, class_name in CITYSCAPES_CLASSES.items():
            unassigned = (panoptic_map == 0) & (label_map == class_id)
            if not unassigned.any():
                continue

            is_thing = class_name in THING_CLASSES
            components: List[np.ndarray]
            if is_thing:
                component_count, component_map, stats, _ = cv2.connectedComponentsWithStats(
                    unassigned.astype(np.uint8), connectivity=8
                )
                components = [
                    component_map == component_id
                    for component_id in range(1, component_count)
                    if int(stats[component_id, cv2.CC_STAT_AREA]) >= 20
                ]
            else:
                components = [unassigned]

            for component in components:
                component = (panoptic_map == 0) & component
                if not component.any():
                    continue
                panoptic_map[component] = segment_id
                segments.append({
                    "id": segment_id,
                    "class_name": class_name,
                    "is_thing": is_thing,
                    "instance_id": None,
                    "area": int(component.sum()),
                    "bbox": self._mask_bbox(component),
                    "confidence": float(sem_conf[component].mean()),
                    "source": "semantic",
                })
                segment_id += 1

        return {
            "panoptic_map": panoptic_map,
            "segments": segments,
            "label_map": label_map,
            "semantic_confidence": sem_conf,
        }

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> List[int]:
        """Tính bounding box từ mask nhị phân."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return [0, 0, 0, 0]
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
