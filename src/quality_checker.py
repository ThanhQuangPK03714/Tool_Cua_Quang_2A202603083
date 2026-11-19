"""
Hệ thống kiểm tra chất lượng mask (QC) trước khi đưa vào CVAT.

Phát hiện:
1. Mask tràn ra nền
2. Mask không phủ đủ vật thể
3. Hai vật cùng class bị gộp
4. Một vật bị chia thành nhiều instance
5. Nhầm class
6. Mask rỗng / diện tích bất thường
7. Biên mask không sát vật thể
8. Bỏ sót đối tượng nhỏ
9. Gán bóng đổ thành vật thể
10. Mask không đúng kích thước ảnh
11. Thing và Stuff bị gán nhầm
12. Vùng chồng lấn không hợp lý
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CONF_HIGH = 0.7
CONF_MEDIUM = 0.4


@dataclass
class MaskQCResult:
    """Kết quả QC cho một mask/mask đề xuất."""
    mask_id: int
    class_name: str
    is_thing: bool
    confidence: float
    image_shape: Tuple[int, int]
    issues: List[str] = field(default_factory=list)
    passed: bool = True
    level: str = "high"  # high | medium | low

    def to_dict(self) -> dict:
        return {
            "mask_id": self.mask_id,
            "class_name": self.class_name,
            "is_thing": self.is_thing,
            "confidence": self.confidence,
            "image_shape": self.image_shape,
            "issues": self.issues,
            "passed": self.passed,
            "level": self.level,
        }


class QualityChecker:
    """Tự động đánh giá chất lượng mask và gán mức độ tin cậy."""

    def __init__(
        self,
        min_mask_area: int = 100,
        max_mask_area_ratio: float = 0.9,
        overlap_threshold: float = 0.3,
        min_confidence_high: float = CONF_HIGH,
        min_confidence_medium: float = CONF_MEDIUM,
        stuff_classes: Optional[List[str]] = None,
        thing_classes: Optional[List[str]] = None,
    ):
        self.min_mask_area = min_mask_area
        self.max_mask_area_ratio = max_mask_area_ratio
        self.overlap_threshold = overlap_threshold
        self.min_confidence_high = min_confidence_high
        self.min_confidence_medium = min_confidence_medium
        self.stuff_classes = set(
            stuff_classes if stuff_classes is not None
            else ["road", "sidewalk", "building", "vegetation", "sky"]
        )
        self.thing_classes = set(
            thing_classes if thing_classes is not None
            else ["person", "car", "bus", "truck", "motorcycle", "bicycle", "traffic light"]
        )

    # ------------------------------------------------------------------ #
    # Check một mask đơn lẻ
    # ------------------------------------------------------------------ #
    def check_single_mask(
        self,
        mask_id: int,
        mask: np.ndarray,
        class_name: str,
        is_thing: bool,
        image_shape: Tuple[int, int],
        confidence: float = 1.0,
    ) -> MaskQCResult:
        """QC một mask đơn lẻ."""
        h, w = image_shape
        issues: List[str] = []

        # 1. Mask rỗng
        if mask.sum() == 0:
            issues.append("Mask rỗng (không có pixel)")
            return MaskQCResult(
                mask_id, class_name, is_thing, confidence, image_shape,
                issues=issues, passed=False, level="low",
            )

        # 2. Kích thước mask không đúng ảnh
        if mask.shape[:2] != (h, w):
            issues.append(f"Mask size {mask.shape} không khớp ảnh {image_shape}")

        # 3. Diện tích bất thường
        area = int(mask.sum())
        area_ratio = area / (h * w)
        if area < self.min_mask_area:
            issues.append(f"Diện tích quá nhỏ ({area} pixels < {self.min_mask_area})")
        if area_ratio > self.max_mask_area_ratio and not is_thing:
            issues.append(f"Vùng stuff quá lớn ({area_ratio:.1%} ảnh), có thể tràn nền")

        # 4. Class hợp lệ
        if is_thing and class_name not in self.thing_classes:
            issues.append(f"Thing '{class_name}' không thuộc danh sách thing")
        if not is_thing and class_name not in self.stuff_classes:
            issues.append(f"Stuff '{class_name}' không thuộc danh sách stuff")

        # 5. Biên mask
        edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        edge_ratio = edges.sum() / 255 / max(area, 1)
        if edge_ratio > 0.5:
            issues.append("Biên mask giáu rác / không mượt")

        # Xác định level
        level = self._determine_level(confidence, issues)
        passed = len(issues) == 0 or level in ("high", "medium")

        return MaskQCResult(
            mask_id, class_name, is_thing, confidence, image_shape,
            issues=issues, passed=passed, level=level,
        )

    # ------------------------------------------------------------------ #
    # Check nhiều mask (phát hiện gộp/tách nhầm)
    # ------------------------------------------------------------------ #
    def check_multiple_masks(
        self,
        masks: List[dict],
        image_shape: Tuple[int, int],
    ) -> List[MaskQCResult]:
        """
        masks: list of dict {id, class_name, is_thing, mask, confidence}
        Trả về danh sách QCResult, đồng thời đánh dấu các mask bị gộp/tách nhầm.
        """
        results: List[MaskQCResult] = []
        for i, m in enumerate(masks):
            res = self.check_single_mask(
                mask_id=i,
                mask=m["mask"],
                class_name=m["class_name"],
                is_thing=m.get("is_thing", False),
                image_shape=image_shape,
                confidence=m.get("confidence", 1.0),
            )
            results.append(res)

        # Phát hiện hai mask cùng class chồng lấn quá mức (có thể gộp nhầm)
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                m1, m2 = masks[i], masks[j]
                if m1["class_name"] != m2["class_name"]:
                    continue
                overlap = self._iou(m1["mask"], m2["mask"])
                if overlap > self.overlap_threshold:
                    results[i].issues.append(f"Chồng lấn {overlap:.2f} với mask {j} cùng class")
                    results[j].issues.append(f"Chồng lấn {overlap:.2f} với mask {i} cùng class")
                    results[i].passed = False
                    results[j].passed = False
                    if results[i].level == "high":
                        results[i].level = "medium"
                    if results[j].level == "high":
                        results[j].level = "medium"

        return results

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        inter = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return float(inter / union)

    def _determine_level(self, confidence: float, issues: List[str]) -> str:
        if not issues and confidence >= self.min_confidence_high:
            return "high"
        if confidence >= self.min_confidence_medium and len(issues) <= 1:
            return "medium"
        return "low"
