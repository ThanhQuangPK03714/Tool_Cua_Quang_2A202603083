"""
Instance Segmentation using YOLOv8-seg (ultralytics).

The default Ultralytics COCO weights support the standard 80 object classes.
The active CVAT task filters this model output to its own label list.
Đầu ra: mỗi instance là một mask riêng với class id và confidence.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from src.device_utils import resolve_device

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# COCO instance segmentation class ids, in the order used by Ultralytics.
COCO_TO_CLASS = dict(enumerate(CLASS_NAMES))


class YOLOv8Seg:
    """YOLOv8-seg nano pretrained trên COCO, chỉ giữ các class của bài."""

    def __init__(
        self,
        device: str = "auto",
        input_size: Tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        weights: str = "yolov8n-seg.pt",
    ):
        self.device = resolve_device(device)
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self._predict_lock = threading.RLock()
        logger.info("Đang tải YOLOv8-seg weights: %s (device=%s) ...", weights, self.device)
        self.model = YOLO(weights)
        names = self.model.names
        self.class_names = list(names.values()) if isinstance(names, dict) else list(names)
        self.class_to_id = {name: index for index, name in enumerate(self.class_names)}
        logger.info("YOLOv8-seg loaded.")

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Dự đoán instance segmentation.

        Parameters
        ----------
        image : np.ndarray
            H×W×3, RGB uint8.

        Returns
        -------
        instances : list of dict
            Mỗi dict: {
                "class_name": str,
                "class_id": int,
                "confidence": float,
                "mask": np.ndarray (H×W uint8, 0/1),
                "bbox": [x1, y1, x2, y2],
                "area": int,
            }
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Ultralytics nhận ndarray theo thứ tự BGR; toàn bộ pipeline dùng RGB.
        image = np.ascontiguousarray(image[:, :, ::-1])

        # YOLO expects BGR, but if already RGB we convert
        prediction_confidence = (
            self.confidence_threshold
            if confidence_threshold is None
            else float(confidence_threshold)
        )
        # Ultralytics keeps a mutable predictor on the YOLO object. This model
        # is shared by Streamlit's resource cache, so serialize access to it.
        with self._predict_lock:
            results = self.model(
                image,
                imgsz=self.input_size,
                conf=prediction_confidence,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

        instances: List[Dict] = []
        if not results or len(results) == 0:
            return instances

        result = results[0]
        h, w = image.shape[:2]

        if result.boxes is None:
            return instances

        boxes = result.boxes.data  # N×6 (x1, y1, x2, y2, conf, cls)
        keypoint_data = None
        if result.keypoints is not None:
            keypoint_data = result.keypoints.data.detach().cpu().numpy()

        model_names = self.model.names
        count = len(boxes)
        for i in range(count):
            cls_id = int(boxes[i, 5].item())
            conf = float(boxes[i, 4].item())
            class_name = model_names.get(cls_id, str(cls_id)) if isinstance(model_names, dict) else model_names[cls_id]
            class_name = str(class_name)

            x1, y1, x2, y2 = [float(v) for v in boxes[i, :4].tolist()]
            # Ultralytics đã đưa tọa độ boxes về kích thước ảnh gốc.
            x1 = max(0, min(w - 1, int(round(x1))))
            y1 = max(0, min(h - 1, int(round(y1))))
            x2 = max(0, min(w - 1, int(round(x2))))
            y2 = max(0, min(h - 1, int(round(y2))))

            segmentation_available = result.masks is not None and i < len(result.masks.data)
            if segmentation_available:
                mask = result.masks.data[i].cpu().numpy()
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0.5).astype(np.uint8)
            else:
                # Preserve usable detection output for box-only YOLO weights.
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y1 : y2 + 1, x1 : x2 + 1] = 1

            area = int(mask.sum())
            if area == 0:
                continue

            instances.append({
                "class_name": class_name,
                "class_id": cls_id,
                "confidence": conf,
                "mask": mask,
                "bbox": [x1, y1, x2, y2],
                "area": area,
                "segmentation_available": segmentation_available,
                **({"keypoints": _format_keypoints(keypoint_data[i])} if keypoint_data is not None and i < len(keypoint_data) else {}),
            })

        # Sắp xếp theo confidence giảm dần
        instances.sort(key=lambda x: x["confidence"], reverse=True)
        return instances


def _format_keypoints(values: np.ndarray) -> List[Dict[str, Optional[float] | str]]:
    """Convert Ultralytics pose output to named COCO points when available."""
    output: List[Dict[str, Optional[float] | str]] = []
    for index, point in enumerate(np.asarray(values)):
        if index >= len(COCO_KEYPOINT_NAMES) or len(point) < 2:
            break
        confidence = float(point[2]) if len(point) > 2 else None
        output.append({
            "name": COCO_KEYPOINT_NAMES[index],
            "x": float(point[0]),
            "y": float(point[1]),
            "confidence": confidence,
        })
    return output
