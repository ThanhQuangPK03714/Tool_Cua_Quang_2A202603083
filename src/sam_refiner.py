"""
SAM2 (Segment Anything Model 2) refinement module.

SAM2 không tự hiểu tên class, nên chỉ dùng để tinh chỉnh biên mask dựa trên
điểm hoặc vùng gợi ý từ người dùng. Kết hợp với mô hình phân loại/phân đoạn chính.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _SAM2_AVAILABLE = True
except Exception as exc:
    _SAM2_AVAILABLE = False
    logger.warning("SAM2 không khả dụng: %s", exc)


class SAM2Refiner:
    """Tinh chỉnh biên mask bằng SAM2 dựa trên điểm gợi ý."""

    def __init__(
        self,
        device: str = "cpu",
        model_cfg: str = "sam2_hiera_t.yaml",
        checkpoint: Optional[str] = None,
    ):
        self.device = device
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint
        self.predictor: Optional[object] = None
        self._initialized = False

        if not _SAM2_AVAILABLE:
            logger.warning("SAM2 không được cài đặt. Chế độ refine sẽ bị vô hiệu hóa.")
            return

        if checkpoint is None:
            logger.warning("Chưa cung cấp checkpoint SAM2. Chế độ refine sẽ bị vô hiệu hóa.")
            return

        self._initialize()

    def _initialize(self):
        try:
            sam2_model = build_sam2(self.model_cfg, self.checkpoint, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            self._initialized = True
            logger.info("SAM2 refiner initialized on %s", self.device)
        except Exception as exc:
            logger.error("Không khởi tạo được SAM2: %s", exc)
            self._initialized = False

    def is_available(self) -> bool:
        return self._initialized and self.predictor is not None

    @torch.no_grad()
    def refine(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        points: Optional[List[Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """
        Tinh chỉnh mask bằng SAM2.

        Parameters
        ----------
        image : np.ndarray
            H×W×3, RGB.
        mask : np.ndarray
            H×W uint8, mask đầu vào (0/1).
        points : list of (x, y, label)
            label=1 là điểm foreground, label=0 là điểm background.

        Returns
        -------
        refined_mask : np.ndarray
            H×W uint8, mask sau khi tinh chỉnh.
        """
        if not self.is_available():
            logger.warning("SAM2 chưa sẵn sàng, trả về mask gốc.")
            return mask

        h, w = mask.shape[:2]
        self.predictor.set_image(image)

        # Chuẩn bị điểm gợi ý
        point_coords = []
        point_labels = []
        if points:
            for x, y, lbl in points:
                point_coords.append([x, y])
                point_labels.append(lbl)

        if point_coords:
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array(point_coords),
                point_labels=np.array(point_labels),
                mask_input=mask[None, :, :].astype(np.float32),
                multimask_output=False,
            )
            refined = masks[0].astype(np.uint8)
        else:
            # Không có điểm, chỉ dùng mask đầu vào làm mask_input
            masks, scores, logits = self.predictor.predict(
                mask_input=mask[None, :, :].astype(np.float32),
                multimask_output=False,
            )
            refined = masks[0].astype(np.uint8)

        return refined
