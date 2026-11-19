"""Optional, real SAM 3 inference through Ultralytics' supported SAM API.

This module intentionally does not return synthetic masks. Inference is enabled
only when a SAM 3 checkpoint is present and the CUDA runtime is available.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from src.device_utils import resolve_device

logger = logging.getLogger(__name__)


class SAM3Unavailable(RuntimeError):
    """Raised when a real SAM 3 checkpoint/runtime is not ready."""


class SAM3Engine:
    """Promptable SAM 3 image segmenter (visual and concept prompts)."""

    def __init__(
        self,
        weights: Optional[str] = None,
        device: str = "auto",
        confidence_threshold: float = 1.0,
        model: Optional[Any] = None,
    ):
        self.device = resolve_device(device)
        self.confidence_threshold = float(confidence_threshold)
        self.weights = self._resolve_weights(weights)
        self.model: Optional[Any] = None
        self._semantic_predictor: Optional[Any] = None
        if not self.weights.is_file():
            raise SAM3Unavailable(
                f"Không tìm thấy checkpoint SAM 3: {self.weights}. Tải checkpoint có quyền sử dụng "
                "từ trang chính thức của Meta/Hugging Face, sau đó đặt SAM3_WEIGHTS trong .env "
                "hoặc đặt file trong thư mục models/."
            )
        try:
            import torch
        except Exception as exc:
            raise SAM3Unavailable("SAM 3 cần PyTorch; hãy cài runtime SAM 3 trong môi trường riêng.") from exc
        if not torch.cuda.is_available() or not self.device.startswith("cuda"):
            raise SAM3Unavailable(
                "SAM 3 upstream hiện yêu cầu GPU CUDA tương thích. Môi trường hiện tại không có CUDA; "
                "YOLO vẫn chạy được trên CPU."
            )
        try:
            from packaging.version import Version

            torch_version = Version(torch.__version__.split("+", 1)[0])
            cuda_version = tuple(int(part) for part in str(torch.version.cuda or "0.0").split(".")[:2])
        except Exception as exc:
            raise SAM3Unavailable("Không xác định được phiên bản PyTorch/CUDA cho SAM 3.") from exc
        if torch_version < Version("2.7") or cuda_version < (12, 6):
            raise SAM3Unavailable(
                f"SAM 3 cần PyTorch >= 2.7 và CUDA >= 12.6; hiện có torch {torch.__version__}, "
                f"CUDA {torch.version.cuda}. Cài runtime theo hướng dẫn upstream."
            )
        if model is not None:
            self.model = model
        else:
            # Build the model only when using visual prompts. Concept prompts
            # have a separate predictor and should not duplicate a 3.45 GB model.
            self.model = None

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update thresholds without reloading the cached model weights."""
        self.confidence_threshold = float(threshold)
        if self._semantic_predictor is not None:
            self._semantic_predictor.args.conf = self.confidence_threshold

    @staticmethod
    def _resolve_weights(weights: Optional[str]) -> Path:
        configured = weights or os.getenv("SAM3_WEIGHTS", "")
        if configured:
            return Path(configured).expanduser().resolve()
        root = Path(__file__).resolve().parents[1]
        candidates = (root / "models" / "sam3.pt", root / "sam3.pt")
        return next((path for path in candidates if path.is_file()), candidates[0])

    def predict(
        self,
        image: np.ndarray,
        *,
        bbox: Optional[Sequence[float]] = None,
        point_coords: Optional[Sequence[Sequence[float]]] = None,
        point_labels: Optional[Sequence[int]] = None,
        text_prompt: Optional[str] = None,
        exemplar_boxes: Optional[Sequence[Sequence[float]]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Return model-produced masks for one image and one prompt.

        SAM scores are exposed only if the installed predictor returns them;
        absent scores remain ``None`` and are not represented as confidence 1.0.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("SAM 3 expects an H×W×3 RGB image.")
        has_points = point_coords is not None and len(point_coords) > 0
        has_text = text_prompt is not None and bool(text_prompt.strip())
        has_exemplars = exemplar_boxes is not None and len(exemplar_boxes) > 0
        if not (bbox is not None or has_points or has_text or has_exemplars or mask is not None):
            raise ValueError("SAM 3 requires a visual prompt, a text prompt, or an exemplar box.")
        if has_points and (point_labels is None or len(point_coords) != len(point_labels)):
            raise ValueError("Each SAM 3 point must have a positive/negative label.")

        image_bgr = np.ascontiguousarray(image[:, :, ::-1])
        if has_text or has_exemplars:
            self._release_visual_predictor()
            return self._predict_concept(image_bgr, text_prompt, exemplar_boxes)
        self._release_semantic_predictor()

        kwargs: Dict[str, Any] = {
            "source": image_bgr,
            "device": self.device,
            "conf": self.confidence_threshold,
            "imgsz": 640,
            "quantize": 16,
            "verbose": False,
        }
        if bbox is not None:
            x1, y1, x2, y2 = [float(value) for value in bbox]
            kwargs["bboxes"] = [[x1, y1, x2, y2]]
        if has_points:
            kwargs["points"] = [[float(x), float(y)] for x, y in point_coords]
            kwargs["labels"] = [int(value) for value in point_labels] if point_labels is not None else []
        if mask is not None:
            kwargs["masks"] = [np.asarray(mask, dtype=np.uint8)]
        results = self._get_visual_model().predict(**kwargs)
        if not results:
            return []
        result = results[0]
        if result.masks is None:
            return []

        h, w = image.shape[:2]
        score_values: Optional[np.ndarray] = None
        boxes = getattr(result, "boxes", None)
        if boxes is not None and getattr(boxes, "conf", None) is not None:
            score_values = boxes.conf.detach().cpu().numpy().astype(float)
        output: List[Dict[str, Any]] = []
        for index, raw_mask in enumerate(result.masks.data):
            mask = raw_mask.detach().cpu().numpy()
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask >= 0.5).astype(np.uint8)
            if not mask.any():
                continue
            score = float(score_values[index]) if score_values is not None and index < len(score_values) else None
            output.append({"mask": mask, "confidence": score, "confidence_source": "sam3" if score is not None else None})
        return output

    def predict_concepts(
        self,
        image: np.ndarray,
        text_prompts: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """Run several open-vocabulary label prompts with one image encoding."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("SAM 3 expects an H×W×3 RGB image.")
        prompts = [str(prompt).strip() for prompt in text_prompts if str(prompt).strip()]
        if not prompts:
            return []

        image_bgr = np.ascontiguousarray(image[:, :, ::-1])
        self._release_visual_predictor()
        try:
            predictor = self._get_semantic_predictor()
            predictor.set_image(image_bgr)
            results = predictor(text=prompts)
        except Exception as exc:
            raise SAM3Unavailable(
                "SAM 3 không chạy được các text prompt. Kiểm tra checkpoint, CLIP và VRAM."
            ) from exc
        if not results or results[0].masks is None:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "cls", None) is None:
            if len(prompts) > 1:
                raise SAM3Unavailable("SAM 3 không trả class ID để ghép mask với từng text prompt.")
            class_ids = np.zeros(len(result.masks.data), dtype=np.int32)
        else:
            class_ids = boxes.cls.detach().cpu().numpy().astype(np.int32)
        scores = (
            boxes.conf.detach().cpu().numpy().astype(float)
            if boxes is not None and getattr(boxes, "conf", None) is not None
            else None
        )

        h, w = image.shape[:2]
        output: List[Dict[str, Any]] = []
        for index, raw_mask in enumerate(result.masks.data):
            class_index = int(class_ids[index]) if index < len(class_ids) else -1
            if class_index < 0 or class_index >= len(prompts):
                continue
            mask = cv2.resize(
                raw_mask.detach().cpu().numpy(),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = (mask >= 0.5).astype(np.uint8)
            if not mask.any():
                continue
            score = float(scores[index]) if scores is not None and index < len(scores) else None
            output.append({
                "prompt_index": class_index,
                "mask": mask,
                "confidence": score,
                "confidence_source": "sam3" if score is not None else None,
            })
        return output

    def track_video(
        self,
        video_path: str,
        *,
        bbox: Optional[Sequence[float]] = None,
        point_coords: Optional[Sequence[Sequence[float]]] = None,
        point_labels: Optional[Sequence[int]] = None,
        mask: Optional[np.ndarray] = None,
        text_prompt: Optional[str] = None,
        exemplar_boxes: Optional[Sequence[Sequence[float]]] = None,
    ) -> Any:
        """Run SAM 3 tracking and yield per-frame tracked masks and identities."""
        if "sam3.1_multiplex" in self.weights.name.casefold():
            raise SAM3Unavailable(
                "Checkpoint SAM 3.1 Multiplex hiện dùng được cho inference ảnh; "
                "video tracking cần checkpoint sam3.pt."
            )
        has_points = point_coords is not None and len(point_coords) > 0
        has_text = text_prompt is not None and bool(text_prompt.strip())
        has_exemplars = exemplar_boxes is not None and len(exemplar_boxes) > 0
        if not (bbox is not None or has_points or has_text or has_exemplars or mask is not None):
            raise ValueError("SAM 3 video tracking requires a text, visual, or mask prompt.")
        if has_points and (point_labels is None or len(point_coords) != len(point_labels)):
            raise ValueError("Each SAM 3 point must have a positive/negative label.")

        self._release_visual_predictor()
        self._release_semantic_predictor()
        semantic_mode = bool(has_text or has_exemplars)
        try:
            if semantic_mode:
                from ultralytics.models.sam import SAM3VideoSemanticPredictor

                predictor = SAM3VideoSemanticPredictor(
                    overrides={
                        "model": str(self.weights),
                        "task": "segment",
                        "mode": "predict",
                        "device": self.device,
                        "conf": self.confidence_threshold,
                        "imgsz": 640,
                        "quantize": 16,
                        "verbose": False,
                        "save": False,
                    }
                )
            else:
                from ultralytics.models.sam import SAM3VideoPredictor

                predictor = SAM3VideoPredictor(
                    overrides={
                        "model": str(self.weights),
                        "task": "segment",
                        "mode": "predict",
                        "device": self.device,
                        "conf": self.confidence_threshold,
                        "imgsz": 640,
                        "quantize": 16,
                        "verbose": False,
                        "save": False,
                    }
                )
            kwargs: Dict[str, Any] = {"source": video_path, "stream": True}
            if has_text:
                kwargs["text"] = [text_prompt]
            boxes = exemplar_boxes if has_exemplars else ([bbox] if bbox is not None else None)
            if boxes is not None and len(boxes) > 0:
                kwargs["bboxes"] = [[float(value) for value in box] for box in boxes]
            if has_points:
                kwargs["points"] = [[float(x), float(y)] for x, y in point_coords]
                kwargs["labels"] = [int(value) for value in point_labels] if point_labels is not None else []
            if mask is not None:
                kwargs["masks"] = [np.asarray(mask, dtype=np.uint8)]
            results = predictor(**kwargs)
            for result in results:
                if result.masks is None:
                    yield {"image_bgr": result.orig_img, "instances": []}
                    continue
                h, w = result.orig_img.shape[:2]
                boxes_result = getattr(result, "boxes", None)
                scores = None
                track_ids = None
                if boxes_result is not None:
                    if getattr(boxes_result, "conf", None) is not None:
                        scores = boxes_result.conf.detach().cpu().numpy().astype(float)
                    if getattr(boxes_result, "id", None) is not None:
                        track_ids = boxes_result.id.detach().cpu().numpy().astype(int)
                instances = []
                for index, raw_mask in enumerate(result.masks.data):
                    raw = raw_mask.detach().cpu().numpy()
                    binary = (raw >= 0.5).astype(np.uint8)
                    if binary.shape != (h, w):
                        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
                    if not binary.any():
                        continue
                    instances.append({
                        "mask": binary,
                        "confidence": float(scores[index]) if scores is not None and index < len(scores) else None,
                        "confidence_source": "sam3" if scores is not None and index < len(scores) else None,
                        "track_id": int(track_ids[index]) if track_ids is not None and index < len(track_ids) else None,
                    })
                yield {"image_bgr": result.orig_img, "instances": instances}
        except Exception as exc:
            raise SAM3Unavailable(
                "Không chạy được SAM 3 video tracking. Kiểm tra checkpoint, VRAM và phiên bản Ultralytics."
            ) from exc

    def _predict_concept(
        self,
        image_bgr: np.ndarray,
        text_prompt: Optional[str],
        exemplar_boxes: Optional[Sequence[Sequence[float]]],
    ) -> List[Dict[str, Any]]:
        """Find all instances matching a text concept, visual exemplar, or both."""
        try:
            predictor = self._get_semantic_predictor()
            predictor.set_image(image_bgr)
            prompts: Dict[str, Any] = {}
            if text_prompt:
                prompts["text"] = [text_prompt]
            if exemplar_boxes is not None and len(exemplar_boxes) > 0:
                prompts["bboxes"] = [[float(value) for value in box] for box in exemplar_boxes]
            results = predictor(**prompts)
        except Exception as exc:
            raise SAM3Unavailable(
                "SAM 3 concept prompt cần SAM3SemanticPredictor và checkpoint tương thích. "
                "Kiểm tra Ultralytics, CLIP và quyền truy cập checkpoint."
            ) from exc
        if not results or results[0].masks is None:
            return []
        h, w = image_bgr.shape[:2]
        output: List[Dict[str, Any]] = []
        result = results[0]
        scores = None
        if getattr(result, "boxes", None) is not None and getattr(result.boxes, "conf", None) is not None:
            scores = result.boxes.conf.detach().cpu().numpy().astype(float)
        for index, raw_mask in enumerate(result.masks.data):
            mask = cv2.resize(raw_mask.detach().cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask >= 0.5).astype(np.uint8)
            if mask.any():
                score = float(scores[index]) if scores is not None and index < len(scores) else None
                output.append({
                    "mask": mask,
                    "confidence": score,
                    "confidence_source": "sam3" if score is not None else None,
                })
        return output

    def _release_visual_predictor(self) -> None:
        """Avoid keeping a second SAM 3 model in VRAM when switching prompt APIs."""
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and getattr(predictor, "model", None) is not None:
            predictor.model = None
            self.model.predictor = None
            self._empty_cuda_cache()

    def _get_visual_model(self) -> Any:
        if self.model is None:
            try:
                from ultralytics import SAM

                self.model = SAM(str(self.weights))
            except Exception as exc:
                raise SAM3Unavailable(
                    "Không khởi tạo được SAM 3 từ checkpoint. Kiểm tra Ultralytics, "
                    "weights và quyền truy cập checkpoint."
                ) from exc
        return self.model

    def _release_semantic_predictor(self) -> None:
        if self._semantic_predictor is not None:
            self._semantic_predictor.model = None
            self._semantic_predictor = None
            self._empty_cuda_cache()

    @staticmethod
    def _empty_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _get_semantic_predictor(self) -> Any:
        """Load and retain the concept predictor for repeated frame queries."""
        if self._semantic_predictor is None:
            try:
                from ultralytics.models.sam import SAM3SemanticPredictor

                self._semantic_predictor = SAM3SemanticPredictor(
                    overrides={
                        "model": str(self.weights),
                        "task": "segment",
                        "mode": "predict",
                        "device": self.device,
                        "conf": self.confidence_threshold,
                        "imgsz": 640,
                        "quantize": 16,
                        "verbose": False,
                        "save": False,
                    }
                )
            except Exception as exc:
                raise SAM3Unavailable(
                    "Không khởi tạo được SAM 3 concept predictor. Kiểm tra Ultralytics và dependency CLIP."
                ) from exc
        self._semantic_predictor.args.conf = self.confidence_threshold
        return self._semantic_predictor
