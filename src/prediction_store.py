"""Disk-backed prediction runs to keep whole-task image arrays out of RAM."""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping

import numpy as np
from PIL import Image

from src.quality_checker import MaskQCResult

logger = logging.getLogger(__name__)


class PredictionStore:
    """Persist one frame at a time and load only the frame currently reviewed."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.manifest_path = self.path / "manifest.json"
        self.path.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self._write_manifest({"version": 1, "frames": []})

    @classmethod
    def create(cls, root: Path, task_id: int, job_id: int) -> "PredictionStore":
        safe_root = Path(root) / "predictions" / f"task_{int(task_id)}" / f"job_{int(job_id)}"
        safe_root.mkdir(parents=True, exist_ok=True)
        run_path = safe_root / uuid.uuid4().hex
        run_path.mkdir()
        return cls(run_path)

    def _read_manifest(self) -> Dict[str, Any]:
        with self.manifest_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write_manifest(self, payload: Dict[str, Any]) -> None:
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.manifest_path)

    @staticmethod
    def _safe_name(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
        return cleaned[:96] or "frame"

    def save_frame(
        self,
        frame_name: str,
        prediction: Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> None:
        manifest = self._read_manifest()
        frame_index = next(
            (index for index, item in enumerate(manifest["frames"]) if item["name"] == frame_name),
            None,
        )
        if frame_index is not None and not replace:
            raise ValueError(f"Frame {frame_name!r} is already saved in this prediction run.")
        if frame_index is None:
            frame_index = len(manifest["frames"])
            prefix = f"{frame_index:06d}_{self._safe_name(frame_name)}"
        else:
            old_entry = manifest["frames"][frame_index]
            prefix = Path(old_entry["image_file"]).stem
        image_file = f"{prefix}.png"
        masks_file = f"{prefix}.npz"
        image_tmp = self.path / f"{prefix}.tmp.png"
        Image.fromarray(np.asarray(prediction["image"], dtype=np.uint8)).save(image_tmp, format="PNG", optimize=True)
        image_tmp.replace(self.path / image_file)

        arrays: Dict[str, np.ndarray] = {}
        masks_metadata: List[Dict[str, Any]] = []
        for mask_index, item in enumerate(prediction.get("masks", [])):
            array_key = f"mask_{mask_index}"
            arrays[array_key] = np.asarray(item["mask"], dtype=np.uint8)
            qc = item["qc"].to_dict() if hasattr(item.get("qc"), "to_dict") else dict(item.get("qc", {}))
            masks_metadata.append({
                "array_key": array_key,
                "class_name": item["class_name"],
                "is_thing": bool(item.get("is_thing", False)),
                "confidence": float(item.get("confidence", 0.0)),
                "confidence_source": item.get("confidence_source"),
                "bbox": item.get("bbox"),
                "keypoints": item.get("keypoints"),
                "track_id": item.get("track_id"),
                "merged_prediction_count": int(item.get("merged_prediction_count", 1)),
                "segmentation_available": bool(item.get("segmentation_available", True)),
                "approval_key": item["approval_key"],
                "qc": qc,
            })
        masks_tmp = self.path / f"{prefix}.tmp.npz"
        with masks_tmp.open("wb") as file:
            np.savez_compressed(file, **arrays)
        masks_tmp.replace(self.path / masks_file)

        frame_entry = {
            "name": frame_name,
            "frame_number": int(prediction["frame_number"]),
            "type": prediction["type"],
            "mode": prediction.get("mode", "Manual + AI"),
            "inference_confidence_threshold": prediction.get("inference_confidence_threshold"),
            "minimum_component_area_px": prediction.get("minimum_component_area_px"),
            "postprocess_summary": prediction.get("postprocess_summary"),
            "image_file": image_file,
            "masks_file": masks_file,
            "masks": masks_metadata,
        }
        if frame_index == len(manifest["frames"]):
            manifest["frames"].append(frame_entry)
        else:
            manifest["frames"][frame_index] = frame_entry
        self._write_manifest(manifest)

    @property
    def frame_names(self) -> List[str]:
        return [item["name"] for item in self._read_manifest()["frames"]]

    def frame_metadata(self, frame_name: str) -> Dict[str, Any]:
        for item in self._read_manifest()["frames"]:
            if item["name"] == frame_name:
                return item
        raise KeyError(frame_name)

    def iter_metadata(self) -> Iterator[Dict[str, Any]]:
        yield from self._read_manifest()["frames"]

    def load_frame(self, frame_name: str) -> Dict[str, Any]:
        metadata = self.frame_metadata(frame_name)
        image = np.asarray(Image.open(self.path / metadata["image_file"]).convert("RGB"))
        with np.load(self.path / metadata["masks_file"]) as stored_masks:
            masks = []
            for item in metadata["masks"]:
                qc_data = item["qc"]
                qc = MaskQCResult(
                    mask_id=int(qc_data.get("mask_id", 0)),
                    class_name=str(qc_data["class_name"]),
                    is_thing=bool(qc_data["is_thing"]),
                    confidence=float(qc_data["confidence"]),
                    image_shape=tuple(qc_data["image_shape"]),
                    issues=list(qc_data.get("issues", [])),
                    passed=bool(qc_data.get("passed", True)),
                    level=str(qc_data.get("level", "medium")),
                )
                masks.append({
                    **item,
                    "mask": np.asarray(stored_masks[item["array_key"]], dtype=np.uint8),
                    "qc": qc,
                })
        return {
            "image": image,
            "frame_number": int(metadata["frame_number"]),
            "type": metadata["type"],
            "mode": metadata["mode"],
            "inference_confidence_threshold": metadata.get("inference_confidence_threshold"),
            "minimum_component_area_px": metadata.get("minimum_component_area_px"),
            "postprocess_summary": metadata.get("postprocess_summary", {}),
            "masks": masks,
        }
