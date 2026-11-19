"""
Semantic Segmentation using SegFormer B0 pretrained on Cityscapes.

Classes: Cityscapes 19-class taxonomy. The active CVAT task selects the labels
that are relevant to its own class list.
Đầu ra: mask mỗi pixel gán nhãn theo class (không phân biệt instance).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.device_utils import resolve_device

logger = logging.getLogger(__name__)

CITYSCAPES_TRAINID_TO_CLASS = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}
CLASS_NAMES = list(CITYSCAPES_TRAINID_TO_CLASS.values())
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

CITYSCAPES_MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


class CityscapesSegformer:
    """SegFormer B0 fine-tuned trên 19 train classes của Cityscapes."""

    def __init__(
        self,
        device: str = "auto",
        input_size: Tuple[int, int] = (512, 512),
        confidence_threshold: float = 0.25,
        model_id: str = CITYSCAPES_MODEL_ID,
    ):
        self.device = torch.device(resolve_device(device))
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.class_names = list(CLASS_NAMES)
        self.num_classes = len(CLASS_NAMES)

        self.processor, self.model = self._load_model(model_id)
        self.model.to(self.device)
        self.model.eval()
        logger.info(
            "Semantic model loaded: model=%s, device=%s, input_size=%s, classes=%d",
            model_id, self.device, input_size, len(self.class_names),
        )
    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def _load_model(self, model_id: str):
        """Tải model và bộ tiền xử lý Cityscapes 19 lớp từ Hugging Face Hub."""
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

            processor = SegformerImageProcessor.from_pretrained(model_id)
            model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu thư viện transformers cho model semantic. Chạy "
                "python -m pip install -r requirements.txt rồi khởi động lại app."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Không tải được model semantic '{model_id}'. Kiểm tra mạng và "
                "cài dependencies trong requirements.txt."
            ) from exc

        if model.config.num_labels != len(CLASS_NAMES):
            raise RuntimeError(
                f"Model semantic có {model.config.num_labels} class; cần đủ "
                f"{len(CLASS_NAMES)} class Cityscapes."
            )
        model_labels = model.config.id2label
        ordered_labels = [
            model_labels.get(str(index), model_labels.get(index, "")).casefold()
            for index in range(len(CLASS_NAMES))
        ]
        if ordered_labels != CLASS_NAMES:
            raise RuntimeError("Thứ tự class trong model không khớp Cityscapes trainId 0..18.")
        return processor, model

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        class_names: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dự đoán semantic mask.

        Parameters
        ----------
        image : np.ndarray
            H×W×3, RGB, uint8.
        class_names : list[str], optional
            Chỉ xét class thuộc task CVAT hiện tại. Nếu bỏ trống, xét cả 19 lớp.

        Returns
        -------
        label_map : np.ndarray
            H×W int32, mỗi pixel là class id (0-18, 255 = unknown).
        confidence : np.ndarray
            H×W float32, độ tin cậy pixel (max softmax).
        """
        h, w = image.shape[:2]
        pil_img = Image.fromarray(image)
        inputs = self.processor(
            images=pil_img,
            return_tensors="pt",
            size={"height": self.input_size[0], "width": self.input_size[1]},
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        logits = self.model(**inputs).logits[0]  # 19×h×w

        # Cityscapes output dùng trainId 0..18, trùng với CLASS_TO_ID.
        if logits.shape[0] == 19:
            probs = F.softmax(logits, dim=0)
            probs_resized = F.interpolate(
                probs.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            )[0]

            requested_names = CLASS_NAMES if class_names is None else class_names
            requested = {name.strip().casefold() for name in requested_names}
            class_ids = [
                class_id for class_id, name in enumerate(CLASS_NAMES)
                if name.casefold() in requested
            ]
            label_map = torch.full((h, w), 255, dtype=torch.int32, device=self.device)
            confidence = torch.zeros((h, w), dtype=torch.float32, device=self.device)
            if not class_ids:
                logger.warning("Task không có class semantic được model hỗ trợ: %s", class_names)
                return label_map.cpu().numpy(), confidence.cpu().numpy()

            selected_class_ids = torch.as_tensor(class_ids, dtype=torch.int32, device=self.device)
            best_confidence, best_class_id = probs_resized.max(dim=0)
            selected_class = (best_class_id.unsqueeze(0) == selected_class_ids[:, None, None]).any(dim=0)
            prediction_confidence = (
                self.confidence_threshold
                if confidence_threshold is None
                else float(confidence_threshold)
            )
            accepted = selected_class & (best_confidence >= prediction_confidence)
            label_map[accepted] = best_class_id[accepted].to(torch.int32)
            confidence[accepted] = best_confidence[accepted]
            return label_map.cpu().numpy(), confidence.cpu().numpy()

        logger.warning("Model có %d kênh đầu ra; cần đúng 19 class Cityscapes.", logits.shape[0])
        return np.full((h, w), 255, dtype=np.int32), np.zeros((h, w), dtype=np.float32)

    def predict_class_mask(
        self, image: np.ndarray, class_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Trả về mask nhị phân + confidence cho một class cụ thể."""
        label_map, confidence = self.predict(image)
        if class_name not in CLASS_TO_ID:
            raise ValueError(f"Class '{class_name}' không thuộc danh sách semantic: {CLASS_NAMES}")
        class_id = CLASS_TO_ID[class_name]
        mask = (label_map == class_id).astype(np.uint8)
        # Confidence trung bình trong vùng mask
        conf_map = np.where(mask == 1, confidence, 0.0)
        return mask, conf_map
