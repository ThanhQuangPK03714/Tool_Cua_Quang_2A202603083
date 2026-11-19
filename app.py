"""Công cụ AI tự động gán nhãn Image Segmentation tích hợp CVAT local.

Giao diện Streamlit:
- Kết nối CVAT, đọc task/job/labels
- Chạy mô hình AI (semantic / instance / panoptic)
- Xem trước mask, QC, duyệt/từ chối
- Đồng bộ annotation lên CVAT
"""

from __future__ import annotations

import io
import base64
import hashlib
import importlib
import json
import logging
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import numpy as np
import cv2
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

# Thêm thư mục gốc vào sys.path để import src
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_loader import load_dotenv

load_dotenv(ROOT / ".env")

def _reload_changed_module(module: Any) -> Any:
    """Reload a local module only when its source changed since the last rerun."""
    source = Path(module.__file__).resolve()
    signature = (source.stat().st_mtime_ns, source.stat().st_size)
    if getattr(module, "__streamlit_source_signature__", None) != signature:
        module = importlib.reload(module)
        module.__streamlit_source_signature__ = signature
    return module


# Streamlit reruns this file but can keep imported modules and their objects in
# memory. Refresh the local modules that provide CVAT calls, prediction cleanup,
# and annotation export whenever their source changes.
import src.cvat_client as cvat_client_module
cvat_client_module = _reload_changed_module(cvat_client_module)
CVATClient = cvat_client_module.CVATClient
from src.task_detector import (
    get_stuff_classes,
    get_thing_classes,
)
from src.model_router import SegmentationType, get_model_config, describe_model
from src.semantic_segmenter import CLASS_NAMES as SEMANTIC_CLASS_NAMES
from src.semantic_segmenter import CityscapesSegformer
from src.instance_segmenter import CLASS_NAMES as YOLO_CLASS_NAMES
from src.instance_segmenter import YOLOv8Seg
from src.panoptic_segmenter import PanopticSegmenter
from src.quality_checker import QualityChecker
import src.mask_processor as mask_processor_module
mask_processor_module = _reload_changed_module(mask_processor_module)
mask_to_polygons = mask_processor_module.mask_to_polygons
from src.sam3_engine import SAM3Engine, SAM3Unavailable
from src.sam3_prompts import is_region_label, sam3_prompt_for_label
from src.device_utils import available_devices, get_device_status
from src.evaluation import (
    cvat_shapes_to_records,
    evaluate_instances,
    recommend_precision_threshold,
)
import src.prediction_store as prediction_store_module
prediction_store_module = _reload_changed_module(prediction_store_module)
PredictionStore = prediction_store_module.PredictionStore
import src.prediction_postprocess as prediction_postprocess_module
prediction_postprocess_module = _reload_changed_module(prediction_postprocess_module)
postprocess_prediction_masks = prediction_postprocess_module.postprocess_prediction_masks
from src.annotation_processor import get_cvat_sublabels, keypoints_to_cvat_skeleton, mask_to_cvat_points
from src.guideline_annotation import (
    ISSUE_TYPES,
    is_facial_landmark_task,
    POINT_STATES,
    SKELETONS,
    SKELETON_BY_NAME,
    SKELETON_POINT_GUIDANCE,
    STREET_LABELS,
    export_annotations,
    load_annotation_file,
    new_skeleton,
    save_annotation_file,
    validate_facial,
    validate_street,
)
import src.facial_landmark.cvat as facial_cvat_module
facial_cvat_module = _reload_changed_module(facial_cvat_module)
FacialTaxonomyError = facial_cvat_module.FacialTaxonomyError
facial_annotations_to_cvat_shapes = facial_cvat_module.facial_annotations_to_cvat_shapes
next_cvat_group = facial_cvat_module.next_cvat_group
resolve_facial_taxonomy = facial_cvat_module.resolve_facial_taxonomy
save_facial_shapes_to_cvat = facial_cvat_module.save_facial_shapes_to_cvat
import src.facial_landmark.mp_face_landmarker as facial_mp_module
facial_mp_module = _reload_changed_module(facial_mp_module)
import src.annotation_export as annotation_export_module
annotation_export_module = _reload_changed_module(annotation_export_module)
PRESET_LABELS = annotation_export_module.PRESET_LABELS
SHAPE_TYPES = annotation_export_module.SHAPE_TYPES
annotation_label_key = annotation_export_module.annotation_label_key
canonical_cvat_shape_type = annotation_export_module.canonical_cvat_shape_type
detect_export_preset = annotation_export_module.detect_export_preset
guideline_shape_for_label = annotation_export_module.guideline_shape_for_label
mask_to_cvat_polylines = annotation_export_module.mask_to_cvat_polylines
normalize_export_label = annotation_export_module.normalize_export_label
recommend_export_shape = annotation_export_module.recommend_export_shape

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cvat_auto_labeler")

# ---------------------------------------------------------------------------
# Cấu hình trang
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Auto Labeler - Image Segmentation (CVAT Local)",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "cvat_client" not in st.session_state:
    st.session_state.cvat_client = None
if "connected" not in st.session_state:
    st.session_state.connected = False
if "task_context" not in st.session_state:
    st.session_state.task_context = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "qc_results" not in st.session_state:
    st.session_state.qc_results = None
if "approved_mask_ids" not in st.session_state:
    st.session_state.approved_mask_ids = set()
if "synced_mask_ids" not in st.session_state:
    st.session_state.synced_mask_ids = set()
if "projects" not in st.session_state:
    st.session_state.projects = []
if "sam_points" not in st.session_state:
    st.session_state.sam_points = []
if "sam_boxes" not in st.session_state:
    st.session_state.sam_boxes = []

# Preserve the authenticated requests.Session when a changed module replaces
# CVATClient's class object during a Streamlit rerun.
stored_cvat_client = st.session_state.get("cvat_client")
if stored_cvat_client is not None and not isinstance(stored_cvat_client, CVATClient):
    try:
        stored_cvat_client.__class__ = CVATClient
    except TypeError:
        migrated_cvat_client = CVATClient.__new__(CVATClient)
        migrated_cvat_client.__dict__.update(stored_cvat_client.__dict__)
        st.session_state.cvat_client = migrated_cvat_client

stored_predictions = st.session_state.get("predictions")
if stored_predictions is not None and not isinstance(stored_predictions, PredictionStore):
    try:
        stored_predictions.__class__ = PredictionStore
    except TypeError:
        migrated_predictions = PredictionStore.__new__(PredictionStore)
        migrated_predictions.__dict__.update(stored_predictions.__dict__)
        st.session_state.predictions = migrated_predictions


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_client() -> CVATClient:
    if st.session_state.cvat_client is None:
        st.session_state.cvat_client = CVATClient(
            base_url=st.session_state.get("cvat_url", "http://localhost:8080"),
            username=st.session_state.get("cvat_user", ""),
            password=st.session_state.get("cvat_pass", ""),
            access_token=st.session_state.get("cvat_access_token", ""),
            auth_mode=st.session_state.get("cvat_auth_mode", "session"),
            organization_slug=st.session_state.get("cvat_organization", ""),
        )
    return st.session_state.cvat_client


def parse_cvat_task_job_url(value: str) -> Optional[Tuple[str, int, int]]:
    """Return the CVAT server URL, task ID, and job ID from a task/job link."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        parsed = urlsplit(value)
        match = re.fullmatch(r"/tasks/(\d+)/jobs/(\d+)/?", parsed.path)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.netloc or not match:
        return None
    return (
        f"{parsed.scheme}://{parsed.netloc}",
        int(match.group(1)),
        int(match.group(2)),
    )


def _apply_task_job_url() -> None:
    """Use the server from a pasted task/job link as the CVAT endpoint."""
    target = parse_cvat_task_job_url(st.session_state.get("cvat_task_job_url", ""))
    if target is None:
        return
    server_url, _, _ = target
    st.session_state.cvat_url = server_url
    if server_url.rstrip("/").casefold() == CVATClient.ONLINE_URL.casefold():
        st.session_state.cvat_environment = "CVAT Online"
    elif server_url.rstrip("/").casefold() == CVATClient.DEFAULT_URL.casefold():
        st.session_state.cvat_environment = "CVAT Local"
    else:
        st.session_state.cvat_environment = "Custom"


@st.cache_resource(show_spinner=False, max_entries=1)
def get_yolo_model(weights: str, device: str) -> YOLOv8Seg:
    return YOLOv8Seg(
        weights=weights,
        device=device,
        input_size=(640, 640),
    )


@st.cache_resource(show_spinner=False, max_entries=1)
def get_semantic_model(device: str) -> CityscapesSegformer:
    """Keep the large SegFormer weights loaded across inference runs."""
    return CityscapesSegformer(
        device=device,
        input_size=(512, 512),
    )


@st.cache_resource(show_spinner=False, max_entries=1)
def get_sam3_model(weights: str, device: str) -> SAM3Engine:
    """Keep SAM 3 loaded across Streamlit reruns and frame prompts."""
    return SAM3Engine(weights=weights, device=device)


@st.cache_data(ttl=30, show_spinner=False)
def get_cached_device_status() -> Dict[str, Any]:
    return get_device_status()


def model_supported_labels(
    mode: str,
    seg_type: str,
    yolo_label_mapping: Optional[Dict[str, str]] = None,
) -> Optional[set[str]]:
    """Return labels the selected built-in model route can produce."""
    if mode in ("YOLO Only", "YOLO + SAM3"):
        if yolo_label_mapping is not None:
            return {name.casefold() for name in yolo_label_mapping.values() if name.strip()}
        return {name.casefold() for name in YOLO_CLASS_NAMES}
    if mode != "Manual + AI":
        return None
    if seg_type == "semantic":
        return {name.casefold() for name in SEMANTIC_CLASS_NAMES}
    if seg_type == "instance":
        return {name.casefold() for name in YOLO_CLASS_NAMES}
    if seg_type == "panoptic":
        return {
            name.casefold()
            for name in SEMANTIC_CLASS_NAMES + ["pedestrian", "traffic_light", "traffic_sign"]
        }
    return set()


def sam3_fallback_labels(
    mode: str,
    seg_type: str,
    label_names: List[str],
    yolo_label_mapping: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Route labels outside a fixed model taxonomy to SAM 3 text prompts."""
    if mode not in ("Manual + AI", "YOLO + SAM3"):
        return []
    supported = model_supported_labels(mode, seg_type, yolo_label_mapping)
    if supported is None:
        return []
    return [name for name in label_names if name.casefold() not in supported]


def normalize_label_name(name: str) -> str:
    """Normalize separators so names such as traffic_light match traffic light."""
    return re.sub(r"[\s_-]+", " ", str(name).strip().casefold())


def companion_model_device(device: str, use_sam3_fallback: bool) -> str:
    """Leave small-VRAM GPUs for SAM 3 when a custom-label fallback is active."""
    if not use_sam3_fallback or not device.startswith("cuda"):
        return device
    try:
        import torch

        device_index = int(device.split(":", 1)[1]) if ":" in device else 0
        total_memory = int(torch.cuda.get_device_properties(device_index).total_memory)
        return "cpu" if total_memory <= 8 * 1024**3 else device
    except Exception:
        return "cpu"


def load_image_from_cvat(
    client: CVATClient,
    task_id: int,
    frame_number: int,
    job_id: Optional[int] = None,
) -> np.ndarray:
    """Tải ảnh từ CVAT và trả về numpy array (RGB)."""
    if job_id is not None:
        content = client.get_job_frame_image(job_id, frame_number, quality="original")
    else:
        content = client.get_frame_image(task_id, frame_number, quality="original")
    img = Image.open(io.BytesIO(content)).convert("RGB")
    return np.array(img)


def frame_number(frame: Dict[str, Any], index: int) -> int:
    """Lấy số frame từ metadata, fallback về vị trí trong task."""
    return int(frame.get("number", index))


def collect_prediction_masks(prediction: Dict[str, Any], labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chuẩn hóa đầu ra các model về danh sách mask dùng chung cho QC/UI."""
    label_lookup = {label["name"].strip().casefold(): label["name"] for label in labels}
    masks: List[Dict[str, Any]] = []

    if prediction["type"] == "semantic":
        label_map = prediction["label_map"]
        confidence = prediction["confidence"]
        for class_id, class_name in enumerate(SEMANTIC_CLASS_NAMES):
            if class_name.casefold() not in label_lookup:
                continue
            binary_mask = (label_map == class_id).astype(np.uint8)
            if not binary_mask.any():
                continue
            masks.append({
                "class_name": label_lookup[class_name.casefold()],
                "is_thing": False,
                "mask": binary_mask,
                "confidence": float(confidence[binary_mask > 0].mean()),
            })

    elif prediction["type"] == "instance":
        for instance in prediction["instances"]:
            name = instance["class_name"].strip().casefold()
            if name not in label_lookup:
                continue
            masks.append({
                "class_name": label_lookup[name],
                "is_thing": True,
                "mask": instance["mask"].astype(np.uint8),
                "confidence": float(instance["confidence"]),
                "bbox": instance.get("bbox"),
                "confidence_source": instance.get("confidence_source", "yolo"),
                "segmentation_available": instance.get("segmentation_available", True),
                "keypoints": instance.get("keypoints"),
                "track_id": instance.get("track_id"),
            })

    elif prediction["type"] == "panoptic":
        for segment in prediction["panoptic"]["segments"]:
            name = segment["class_name"].strip().casefold()
            source = segment.get("source")
            if name == "person" and source == "instance" and "pedestrian" in label_lookup:
                name = "pedestrian"
            elif name == "traffic light":
                # The job carries both BDD object labels (space-separated)
                # and Cityscapes semantic labels (underscore-separated).
                # Keep each model source aligned with its taxonomy.
                alias = "traffic_light" if source == "semantic" else "traffic light"
                if alias in label_lookup:
                    name = alias
            elif name == "traffic sign" and source == "semantic":
                alias = "traffic_sign"
                if alias in label_lookup:
                    name = alias
            if name not in label_lookup:
                continue
            masks.append({
                "class_name": label_lookup[name],
                "is_thing": bool(segment["is_thing"]),
                "mask": (prediction["panoptic"]["panoptic_map"] == segment["id"]).astype(np.uint8),
                "confidence": float(segment["confidence"]),
                "confidence_source": source,
            })

    return masks


def run_prediction(
    ctx: Dict[str, Any],
    client: CVATClient,
    conf_threshold: float,
    device: str,
    mode: str = "Manual + AI",
    yolo_weights: str = "yolov8n-seg.pt",
    sam3_weights: Optional[str] = None,
    current_only: bool = False,
    selected_frame_number: Optional[int] = None,
    sam_prompts: Optional[Dict[str, Any]] = None,
    yolo_label_mapping: Optional[Dict[str, str]] = None,
    min_component_area_px: int = 96,
) -> None:
    """Chạy model, lưu dự đoán và kết quả QC vào Streamlit session."""
    seg_type = ctx.get("segmentation_type")
    frames = ctx.get("frames") or []
    fallback_labels = sam3_fallback_labels(
        mode,
        seg_type,
        ctx.get("label_names", []),
        yolo_label_mapping,
    )
    if not frames:
        st.error("Task không có ảnh. Hãy tải lại ngữ cảnh task.")
        return
    if is_facial_landmark_task(ctx.get("label_names", [])):
        st.error("Task này dùng 7 label Facial Landmark-50. Model hiện tại chỉ dự đoán mask phân đoạn, không dự đoán các skeleton/keypoint khuôn mặt.")
        st.info("Không chạy YOLO/SAM/semantic segmentation cho task này. Hãy vẽ skeleton trực tiếp trong CVAT, hoặc dùng Canvas local → Facial Landmark-50 để tạo annotation JSON. Canvas local hiện không tự đồng bộ JSON lên CVAT.")
        return
    if mode == "Manual + AI" and seg_type not in ("semantic", "instance", "panoptic") and not fallback_labels:
        st.error(f"Không nhận diện được loại segmentation: {seg_type!r}.")
        return

    indexed_frames = [(frame_number(frame, idx), frame) for idx, frame in enumerate(frames)]
    if mode == "SAM3 Video Track":
        start_index = next(
            (index for index, item in enumerate(indexed_frames) if item[0] == selected_frame_number),
            0,
        )
        indexed_frames = indexed_frames[start_index:]
    if current_only:
        selected = next(
            (item for item in indexed_frames if item[0] == selected_frame_number),
            indexed_frames[0],
        )
        indexed_frames = [selected]

    if mode == "SAM3 Video Track" and current_only:
        st.error("SAM 3 video tracking cần chạy từ frame được chọn tới hết các frame còn lại.")
        return

    if mode == "SAM3 Only" and not current_only and not (
        sam_prompts
        and sam_prompts.get("text_prompt")
        and not sam_prompts.get("bbox")
        and not sam_prompts.get("point_coords")
        and not sam_prompts.get("exemplar_boxes")
    ):
        st.error("Chạy SAM 3 trên toàn bộ frame cần text prompt. Box, điểm và exemplar áp dụng cho frame đang chọn.")
        return

    store: Optional[PredictionStore] = None
    postprocess_totals = {
        "removed_masks": 0,
        "removed_components": 0,
        "merged_lane_masks": 0,
    }
    try:
        with st.spinner("Đang tải model và chạy dự đoán..."):
            yolo_model = None
            sam3_model = None
            base_device = companion_model_device(device, bool(fallback_labels))
            if base_device == "cpu" and device.startswith("cuda"):
                get_yolo_model.clear()
                get_semantic_model.clear()
            elif mode in ("SAM3 Only", "SAM3 Video Track") and device.startswith("cuda"):
                # Do not keep unused Torch models in VRAM while SAM 3 is active.
                get_yolo_model.clear()
                get_semantic_model.clear()
            elif mode == "YOLO + SAM3" and device.startswith("cuda"):
                get_semantic_model.clear()
            has_yolo_task_label = any(
                name.casefold() in {item.casefold() for item in YOLO_CLASS_NAMES}
                for name in ctx["label_names"]
            )
            if mode == "YOLO Only" or (
                mode in ("YOLO + SAM3", "Manual + AI")
                and (mode == "YOLO + SAM3" or seg_type == "instance")
                and has_yolo_task_label
            ):
                yolo_model = get_yolo_model(yolo_weights, base_device)
            if mode in ("SAM3 Only", "YOLO + SAM3", "SAM3 Video Track") or fallback_labels:
                sam3_model = get_sam3_model(sam3_weights or "", device)
                sam3_model.set_confidence_threshold(conf_threshold)

            model = None
            if mode == "Manual + AI" and seg_type == "semantic" and any(
                name.casefold() in {item.casefold() for item in SEMANTIC_CLASS_NAMES}
                for name in ctx["label_names"]
            ):
                model = get_semantic_model(base_device)
            elif mode == "Manual + AI" and seg_type == "panoptic" and any(
                name.casefold() in model_supported_labels(mode, seg_type)
                for name in ctx["label_names"]
            ):
                semantic_model = get_semantic_model(base_device)
                instance_model = get_yolo_model(yolo_weights, base_device)
                model = PanopticSegmenter(
                    device=base_device,
                    semantic_input_size=(512, 512),
                    instance_input_size=(640, 640),
                    instance_weights=yolo_weights,
                    semantic_model=semantic_model,
                    instance_model=instance_model,
                )

            store = PredictionStore.create(ROOT / "outputs", ctx["task_id"], int(ctx.get("job_id") or 0))
            stuff_names = [name.lower() for name in get_stuff_classes(ctx["label_names"])]
            thing_names = [name.lower() for name in get_thing_classes(ctx["label_names"])]
            region_names = [name.lower() for name in ctx["label_names"] if is_region_label(name)]
            if mode == "Manual + AI" and seg_type == "semantic":
                stuff_names = [name.lower() for name in ctx["label_names"]]
            else:
                known_names = set(stuff_names + thing_names)
                thing_names.extend(
                    name.lower()
                    for name in ctx["label_names"]
                    if name.lower() not in known_names and name.lower() not in region_names
                )
                stuff_names.extend(name for name in region_names if name not in stuff_names)
            quality_checker = QualityChecker(stuff_classes=stuff_names, thing_classes=thing_names)
            progress = st.progress(0, text="Bắt đầu inference")

            if mode == "SAM3 Video Track":
                tracked_count, postprocess_totals = _run_sam3_video_prediction(
                    ctx=ctx,
                    client=client,
                    frames=indexed_frames,
                    model=sam3_model,
                    store=store,
                    quality_checker=quality_checker,
                    conf_threshold=conf_threshold,
                    prompts=sam_prompts or {},
                    progress=progress,
                    min_component_area_px=min_component_area_px,
                )
                st.session_state.predictions = store
                st.session_state.qc_results = None
                st.session_state.approved_mask_ids = set()
                st.session_state.synced_mask_ids = set()
                st.success(
                    f"Đã theo dõi {len(store.frame_names)} frame, nhận diện {tracked_count} mask có track ID. "
                    "Ảnh và mask được lưu theo frame. Confidence không phải độ chính xác đã kiểm chứng."
                )
                _show_postprocess_summary(postprocess_totals)
                return

            for frame_index, (number, frame) in enumerate(indexed_frames):
                image = load_image_from_cvat(client, ctx["task_id"], number, ctx.get("job_id"))
                frame_name = frame.get("name") or f"frame_{number}"
                prediction: Dict[str, Any] = {
                    "image": image,
                    "frame_number": number,
                    "type": "instance" if mode != "Manual + AI" else seg_type,
                    "mode": mode,
                    "inference_confidence_threshold": float(conf_threshold),
                }

                if mode == "SAM3 Only":
                    prompt = sam_prompts or {}
                    outputs = sam3_model.predict(
                        image,
                        bbox=prompt.get("bbox"),
                        point_coords=prompt.get("point_coords"),
                        point_labels=prompt.get("point_labels"),
                        text_prompt=prompt.get("text_prompt"),
                        exemplar_boxes=prompt.get("exemplar_boxes"),
                    )
                    class_name = prompt.get("class_name") or (ctx["label_names"][0] if ctx["label_names"] else "")
                    prediction["instances"] = [
                        {
                            "class_name": class_name,
                            "confidence": float(output["confidence"] or 0.0),
                            "confidence_source": output.get("confidence_source"),
                            "mask": output["mask"],
                            "bbox": mask_bbox(output["mask"]),
                            "segmentation_available": True,
                        }
                        for output in outputs
                    ]
                elif mode in ("YOLO Only", "YOLO + SAM3") or (mode == "Manual + AI" and seg_type == "instance"):
                    instances = (
                        yolo_model.predict(image, confidence_threshold=conf_threshold)
                        if yolo_model is not None
                        else []
                    )
                    if mode == "YOLO + SAM3":
                        refined_instances = []
                        for instance in instances:
                            refined = sam3_model.predict(image, bbox=instance.get("bbox"))
                            if not refined:
                                continue
                            best = max(refined, key=lambda item: float(item.get("confidence") or 0.0))
                            refined_instances.append({
                                **instance,
                                "mask": best["mask"],
                                "area": int(best["mask"].sum()),
                                "segmentation_available": True,
                                "confidence_source": "yolo detection; SAM 3 mask score unavailable",
                            })
                        instances = refined_instances
                    if yolo_label_mapping is not None:
                        mapping_lookup = {
                            source_name.casefold(): target_name
                            for source_name, target_name in yolo_label_mapping.items()
                            if target_name.strip()
                        }
                        mapped_instances = []
                        for instance in instances:
                            source_name = str(instance["class_name"])
                            target_name = mapping_lookup.get(source_name.casefold())
                            if target_name is None:
                                continue
                            mapped_instances.append({
                                **instance,
                                "model_class_name": source_name,
                                "class_name": target_name,
                            })
                        instances = mapped_instances
                    prediction["instances"] = instances
                elif seg_type == "semantic":
                    if model is not None:
                        prediction["label_map"], prediction["confidence"] = model.predict(
                            image,
                            class_names=ctx["label_names"],
                            confidence_threshold=conf_threshold,
                        )
                    else:
                        prediction["label_map"] = np.full(image.shape[:2], 255, dtype=np.int32)
                        prediction["confidence"] = np.zeros(image.shape[:2], dtype=np.float32)
                elif seg_type == "panoptic" and model is not None:
                    prediction["panoptic"] = model.predict(
                        image,
                        confidence_threshold=conf_threshold,
                    )
                elif seg_type == "panoptic":
                    prediction["panoptic"] = {
                        "panoptic_map": np.zeros(image.shape[:2], dtype=np.int32),
                        "segments": [],
                    }
                else:
                    prediction["instances"] = []

                masks = collect_prediction_masks(prediction, ctx["labels"])
                if fallback_labels:
                    progress.progress(
                        (frame_index + 1) / len(indexed_frames),
                        text=f"Frame {frame_index + 1}/{len(indexed_frames)} · SAM 3: {len(fallback_labels)} label tùy biến",
                    )
                    concept_outputs = sam3_model.predict_concepts(
                        image,
                        [sam3_prompt_for_label(name) for name in fallback_labels],
                    )
                    for output in concept_outputs:
                        label_index = int(output["prompt_index"])
                        if label_index >= len(fallback_labels):
                            continue
                        label_name = fallback_labels[label_index]
                        masks.append({
                            "class_name": label_name,
                            "is_thing": not is_region_label(label_name),
                            "mask": output["mask"].astype(np.uint8),
                            "confidence": float(output.get("confidence") or 0.0),
                            "confidence_source": output.get("confidence_source"),
                            "bbox": mask_bbox(output["mask"]),
                            "segmentation_available": True,
                        })
                masks, frame_postprocess = postprocess_prediction_masks(
                    masks,
                    image.shape[:2],
                    min_component_area_px=min_component_area_px,
                )
                for key in postprocess_totals:
                    postprocess_totals[key] += frame_postprocess[key]
                prediction["minimum_component_area_px"] = int(min_component_area_px)
                prediction["postprocess_summary"] = frame_postprocess
                results = quality_checker.check_multiple_masks(masks, image.shape[:2])
                prediction["masks"] = []
                for idx, (mask_info, result) in enumerate(zip(masks, results)):
                    approval_key = f"{ctx['task_id']}:{ctx['job_id']}:{number}:{idx}"
                    prediction["masks"].append({**mask_info, "qc": result, "approval_key": approval_key})
                store.save_frame(frame_name, prediction)
                progress.progress((frame_index + 1) / len(indexed_frames), text=f"Đã xử lý {frame_index + 1}/{len(indexed_frames)} ảnh")
                del image, prediction, masks, results

        st.session_state.predictions = store
        st.session_state.qc_results = None
        st.session_state.approved_mask_ids = set()
        st.session_state.synced_mask_ids = set()
        st.success(f"Đã dự đoán {len(store.frame_names)} ảnh. Ảnh và mask được lưu từng frame xuống đĩa để giảm RAM. Confidence không phải độ chính xác đã kiểm chứng.")
        _show_postprocess_summary(postprocess_totals)
    except Exception as exc:
        if store is not None and store.frame_names:
            st.session_state.predictions = store
        logger.exception("Prediction failed")
        st.error(f"Lỗi chạy mô hình: {exc}")


def _run_sam3_video_prediction(
    *,
    ctx: Dict[str, Any],
    client: CVATClient,
    frames: List[Tuple[int, Dict[str, Any]]],
    model: SAM3Engine,
    store: PredictionStore,
    quality_checker: QualityChecker,
    conf_threshold: float,
    prompts: Dict[str, Any],
    progress: Any,
    min_component_area_px: int = 96,
) -> Tuple[int, Dict[str, int]]:
    """Encode selected CVAT frames to a temporary clip and store SAM 3 tracks."""
    if not frames:
        raise ValueError("Không có frame nào từ vị trí bắt đầu đã chọn.")
    output_root = ROOT / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    height = width = None
    with tempfile.TemporaryDirectory(prefix="sam3-track-", dir=output_root) as temp_dir:
        video_path = Path(temp_dir) / "cvat_job.mp4"
        writer = None
        try:
            for frame_index, (number, _) in enumerate(frames):
                image = load_image_from_cvat(client, ctx["task_id"], number, ctx.get("job_id"))
                frame_height, frame_width = image.shape[:2]
                if height is None:
                    height, width = frame_height, frame_width
                    video_width, video_height = width + width % 2, height + height % 2
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        5.0,
                        (video_width, video_height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Không tạo được video tạm để chạy SAM 3 tracking.")
                elif (frame_height, frame_width) != (height, width):
                    raise RuntimeError("Các frame trong Job khác kích thước; video tracking cần kích thước frame đồng nhất.")

                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if width % 2 or height % 2:
                    image_bgr = cv2.copyMakeBorder(
                        image_bgr,
                        0,
                        height % 2,
                        0,
                        width % 2,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )
                writer.write(image_bgr)
                progress.progress(
                    min((frame_index + 1) / (len(frames) * 2), 0.5),
                    text=f"Chuẩn bị video {frame_index + 1}/{len(frames)} frame",
                )
        finally:
            if writer is not None:
                writer.release()

        outputs = model.track_video(
            str(video_path),
            bbox=prompts.get("bbox"),
            point_coords=prompts.get("point_coords"),
            point_labels=prompts.get("point_labels"),
            mask=prompts.get("mask"),
            text_prompt=prompts.get("text_prompt"),
            exemplar_boxes=prompts.get("exemplar_boxes"),
        )
        total_instances = 0
        postprocess_totals = {
            "removed_masks": 0,
            "removed_components": 0,
            "merged_lane_masks": 0,
        }
        for frame_index, output in enumerate(outputs):
            if frame_index >= len(frames):
                break
            number, frame = frames[frame_index]
            image_bgr = output["image_bgr"][:height, :width]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            frame_name = frame.get("name") or f"frame_{number}"
            prediction: Dict[str, Any] = {
                "image": image_rgb,
                "frame_number": number,
                "type": "instance",
                "mode": "SAM3 Video Track",
                "inference_confidence_threshold": float(conf_threshold),
            }
            class_name = prompts.get("class_name") or (ctx["label_names"][0] if ctx["label_names"] else "")
            prediction["instances"] = [
                {
                    "class_name": class_name,
                    "confidence": float(instance["confidence"] or 0.0),
                    "confidence_source": instance.get("confidence_source"),
                    "mask": instance["mask"][:height, :width],
                    "bbox": mask_bbox(instance["mask"][:height, :width]),
                    "segmentation_available": True,
                    "track_id": instance.get("track_id"),
                }
                for instance in output["instances"]
            ]
            masks = collect_prediction_masks(prediction, ctx["labels"])
            masks, frame_postprocess = postprocess_prediction_masks(
                masks,
                image_rgb.shape[:2],
                min_component_area_px=min_component_area_px,
            )
            for key in postprocess_totals:
                postprocess_totals[key] += frame_postprocess[key]
            prediction["minimum_component_area_px"] = int(min_component_area_px)
            prediction["postprocess_summary"] = frame_postprocess
            results = quality_checker.check_multiple_masks(masks, image_rgb.shape[:2])
            prediction["masks"] = []
            for index, (mask_info, result) in enumerate(zip(masks, results)):
                approval_key = f"{ctx['task_id']}:{ctx['job_id']}:{number}:{index}"
                prediction["masks"].append({**mask_info, "qc": result, "approval_key": approval_key})
                total_instances += int(mask_info.get("track_id") is not None)
            store.save_frame(frame_name, prediction)
            progress.progress(
                0.5 + 0.5 * (frame_index + 1) / len(frames),
                text=f"SAM 3 theo dõi {frame_index + 1}/{len(frames)} frame",
            )
        if len(store.frame_names) < len(frames):
            st.warning(f"SAM 3 trả về {len(store.frame_names)}/{len(frames)} frame.")
        return total_instances, postprocess_totals


def _show_postprocess_summary(stats: Dict[str, int]) -> None:
    removed = int(stats.get("removed_masks", 0))
    components = int(stats.get("removed_components", 0))
    merged = int(stats.get("merged_lane_masks", 0))
    st.caption(
        f"Lọc sau inference: bỏ {removed} mask quá nhỏ, loại {components} mảnh nhiễu, "
        f"gộp {merged} mask vạch đường cùng label."
    )


def mask_bbox(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def find_mask_overlap_pairs(items: List[Dict[str, Any]]) -> List[Tuple[int, int, int]]:
    """Return mask pairs sharing pixels; -1 means incompatible mask shapes."""
    overlaps: List[Tuple[int, int, int]] = []
    masks = [np.asarray(item["mask"]) for item in items]
    bounds = [mask_bbox(mask) if mask.ndim == 2 and np.any(mask) else None for mask in masks]
    for left_index, left_mask in enumerate(masks):
        for right_index in range(left_index + 1, len(masks)):
            right_mask = masks[right_index]
            if left_mask.shape != right_mask.shape:
                overlaps.append((left_index, right_index, -1))
                continue
            left_bounds, right_bounds = bounds[left_index], bounds[right_index]
            if left_bounds is None or right_bounds is None:
                continue
            left_x1, left_y1, left_x2, left_y2 = left_bounds
            right_x1, right_y1, right_x2, right_y2 = right_bounds
            x1, y1 = max(left_x1, right_x1), max(left_y1, right_y1)
            x2, y2 = min(left_x2, right_x2), min(left_y2, right_y2)
            if x1 > x2 or y1 > y2:
                continue
            left_region = left_mask[y1 : y2 + 1, x1 : x2 + 1] > 0
            right_region = right_mask[y1 : y2 + 1, x1 : x2 + 1] > 0
            pixel_count = int(np.count_nonzero(left_region & right_region))
            if pixel_count:
                overlaps.append((left_index, right_index, pixel_count))
    return overlaps


def sync_approved_mask_ids_from_widgets(predictions: PredictionStore) -> None:
    """Merge any live checkbox values without losing approvals from unmounted widgets."""
    run_id = predictions.path.name
    for frame_meta in predictions.iter_metadata():
        for item in frame_meta["masks"]:
            approval_key = item["approval_key"]
            widget_key = f"approve_{run_id}_{approval_key}"
            if approval_key in st.session_state.synced_mask_ids:
                st.session_state.approved_mask_ids.discard(approval_key)
            elif widget_key in st.session_state:
                if bool(st.session_state[widget_key]):
                    st.session_state.approved_mask_ids.add(approval_key)
                else:
                    st.session_state.approved_mask_ids.discard(approval_key)


def _on_mask_approval_change(approval_key: str, widget_key: str) -> None:
    """Persist checkbox changes before Streamlit can unmount the review fragment."""
    if approval_key in st.session_state.synced_mask_ids:
        st.session_state.approved_mask_ids.discard(approval_key)
    elif bool(st.session_state.get(widget_key, False)):
        st.session_state.approved_mask_ids.add(approval_key)
    else:
        st.session_state.approved_mask_ids.discard(approval_key)


def compute_ground_truth_evaluation(
    ctx: Dict[str, Any],
    predictions: PredictionStore,
    target_precision: float,
) -> Dict[str, Any]:
    """Evaluate only frames present in the selected prediction run."""
    frame_names = set(predictions.frame_names)
    ground_truth = [
        item
        for item in cvat_shapes_to_records(ctx["annotations"], ctx["labels"], ctx["frames"])
        if item["image_id"] in frame_names
    ]
    result: Dict[str, Any] = {
        "ground_truth_count": len(ground_truth),
        "evaluated_frame_count": len(frame_names),
        "job_frame_count": int(ctx.get("image_count") or len(ctx["frames"])),
        "scores": None,
        "missing_threshold_metadata": False,
        "mixed_thresholds": False,
        "calibration": None,
    }
    if not ground_truth:
        return result

    predicted_records = []
    frame_thresholds = []
    for frame_name in predictions.frame_names:
        frame_meta = predictions.frame_metadata(frame_name)
        frame_threshold = frame_meta.get("inference_confidence_threshold")
        if frame_threshold is None:
            result["missing_threshold_metadata"] = True
        else:
            frame_thresholds.append(float(frame_threshold))
        frame_prediction = predictions.load_frame(frame_name)
        for item in frame_prediction["masks"]:
            confidence_value = item.get("confidence")
            if (
                frame_prediction["mode"] in ("SAM3 Only", "SAM3 Video Track")
                and item.get("confidence_source") is None
            ):
                confidence_value = None
            predicted_records.append({
                "image_id": frame_name,
                "class_name": item["class_name"],
                "mask": item["mask"],
                "confidence": confidence_value,
            })
        del frame_prediction

    result["scores"] = evaluate_instances(predicted_records, ground_truth, iou_threshold=0.5)
    result["mixed_thresholds"] = len(set(frame_thresholds)) > 1
    if not result["missing_threshold_metadata"] and not result["mixed_thresholds"]:
        result["calibration"] = recommend_precision_threshold(
            predicted_records,
            ground_truth,
            target_precision=target_precision,
            iou_threshold=0.5,
            minimum_threshold=frame_thresholds[0] if frame_thresholds else 0.0,
        )
    return result


def ground_truth_evaluation_cache_key(
    ctx: Dict[str, Any],
    predictions: PredictionStore,
    target_precision: float,
) -> Tuple[Any, ...]:
    """Invalidate cached metrics when annotations, predictions, or target change."""
    annotations = ctx.get("annotations") or {}
    manifest_stat = predictions.manifest_path.stat()
    return (
        int(ctx["task_id"]),
        int(ctx["job_id"]),
        str(annotations.get("version", "unknown")),
        len(annotations.get("shapes", [])),
        len(annotations.get("tracks", [])),
        str(predictions.path),
        manifest_stat.st_mtime_ns,
        manifest_stat.st_size,
        round(float(target_precision), 2),
    )


def _select_all_non_overlapping_masks(
    notice_key: str,
    candidates: Tuple[Tuple[str, str, float, Tuple[str, ...]], ...],
) -> None:
    """Keep current approvals and add as many non-overlapping masks as possible."""
    locked_keys = set(st.session_state.synced_mask_ids)
    already_selected = {
        approval_key
        for approval_key, widget_key, _, _ in candidates
        if approval_key in st.session_state.approved_mask_ids
        or bool(st.session_state.get(widget_key, False))
    }
    locked_keys.update(already_selected)
    kept = len(already_selected)
    accepted = skipped = 0
    for approval_key, widget_key, _, overlapping_keys in sorted(
        candidates,
        key=lambda item: item[2],
        reverse=True,
    ):
        if approval_key in st.session_state.synced_mask_ids:
            continue
        if approval_key in already_selected:
            st.session_state[widget_key] = True
            st.session_state.approved_mask_ids.add(approval_key)
            continue
        if any(other_key in locked_keys for other_key in overlapping_keys):
            st.session_state[widget_key] = False
            st.session_state.approved_mask_ids.discard(approval_key)
            skipped += 1
            continue
        st.session_state[widget_key] = True
        st.session_state.approved_mask_ids.add(approval_key)
        locked_keys.add(approval_key)
        accepted += 1

    st.session_state[notice_key] = (
        f"Giữ nguyên {kept} mask đã chọn, tích thêm {accepted} mask không chồng lấn; "
        f"bỏ qua {skipped} mask giao nhau."
    )


def _select_all_masks_including_overlaps(
    notice_key: str,
    candidates: Tuple[Tuple[str, str, float, Tuple[str, ...]], ...],
) -> None:
    """Select every available mask for review, even when masks overlap."""
    for approval_key, widget_key, _, _ in candidates:
        if approval_key in st.session_state.synced_mask_ids:
            continue
        st.session_state[widget_key] = True
        st.session_state.approved_mask_ids.add(approval_key)
    st.session_state[notice_key] = (
        f"Đã tích {len(candidates)} mask trong frame này, kể cả mask giao pixel."
    )


def _select_masks_across_all_frames(
    predictions: PredictionStore,
    *,
    include_overlaps: bool,
) -> Dict[str, int]:
    """Approve every usable mask in a prediction run, frame by frame."""
    sync_approved_mask_ids_from_widgets(predictions)
    run_id = predictions.path.name
    stats = {"eligible": 0, "selected": 0, "added": 0, "skipped_overlap": 0}

    for frame_meta in predictions.iter_metadata():
        prediction = predictions.load_frame(frame_meta["name"])
        mask_items = prediction["masks"]
        overlap_pairs = find_mask_overlap_pairs(mask_items)
        overlaps_by_index: Dict[int, List[int]] = {}
        for left_index, right_index, _ in overlap_pairs:
            overlaps_by_index.setdefault(left_index, []).append(right_index)
            overlaps_by_index.setdefault(right_index, []).append(left_index)

        candidates = []
        for index, item in enumerate(mask_items):
            approval_key = item["approval_key"]
            if approval_key in st.session_state.synced_mask_ids:
                continue
            if not np.asarray(item["mask"]).any():
                continue
            widget_key = f"approve_{run_id}_{approval_key}"
            overlap_keys = tuple(
                mask_items[other_index]["approval_key"]
                for other_index in overlaps_by_index.get(index, [])
            )
            candidates.append((
                approval_key,
                widget_key,
                float(item.get("confidence") or 0.0),
                overlap_keys,
            ))

        stats["eligible"] += len(candidates)
        if include_overlaps:
            for approval_key, widget_key, _, _ in candidates:
                was_selected = (
                    approval_key in st.session_state.approved_mask_ids
                    or bool(st.session_state.get(widget_key, False))
                )
                st.session_state[widget_key] = True
                st.session_state.approved_mask_ids.add(approval_key)
                stats["selected"] += 1
                stats["added"] += int(not was_selected)
            continue

        already_selected = {
            approval_key
            for approval_key, widget_key, _, _ in candidates
            if approval_key in st.session_state.approved_mask_ids
            or bool(st.session_state.get(widget_key, False))
        }
        locked_keys = set(st.session_state.synced_mask_ids) | already_selected
        stats["selected"] += len(already_selected)
        for approval_key, widget_key, _, overlapping_keys in sorted(
            candidates,
            key=lambda item: item[2],
            reverse=True,
        ):
            if approval_key in already_selected:
                st.session_state[widget_key] = True
                st.session_state.approved_mask_ids.add(approval_key)
                continue
            if any(other_key in locked_keys for other_key in overlapping_keys):
                st.session_state[widget_key] = False
                st.session_state.approved_mask_ids.discard(approval_key)
                stats["skipped_overlap"] += 1
                continue
            st.session_state[widget_key] = True
            st.session_state.approved_mask_ids.add(approval_key)
            locked_keys.add(approval_key)
            stats["selected"] += 1
            stats["added"] += 1

    return stats


def _clear_all_pending_approvals(predictions: PredictionStore) -> int:
    """Unapprove masks across every frame without changing synchronized masks."""
    sync_approved_mask_ids_from_widgets(predictions)
    run_id = predictions.path.name
    cleared = 0
    for frame_meta in predictions.iter_metadata():
        for item in frame_meta["masks"]:
            approval_key = item["approval_key"]
            if approval_key in st.session_state.synced_mask_ids:
                continue
            widget_key = f"approve_{run_id}_{approval_key}"
            was_selected = (
                approval_key in st.session_state.approved_mask_ids
                or bool(st.session_state.get(widget_key, False))
            )
            st.session_state[widget_key] = False
            st.session_state.approved_mask_ids.discard(approval_key)
            cleared += int(was_selected)
    return cleared


@st.fragment
def show_mask_approval_ui(
    predictions: PredictionStore,
    mask_items: List[Dict[str, Any]],
    overlap_pairs: List[Tuple[int, int, int]],
    image: np.ndarray,
    image_width: int,
    frame_number: int,
) -> None:
    """Render fast, isolated mask approval controls for the selected frame."""
    overlaps_by_index: Dict[int, List[Tuple[int, int]]] = {}
    overlaps_by_key: Dict[str, List[Tuple[str, int, str, int]]] = {}
    for left_index, right_index, pixel_count in overlap_pairs:
        left_item, right_item = mask_items[left_index], mask_items[right_index]
        overlaps_by_index.setdefault(left_index, []).append((right_index, pixel_count))
        overlaps_by_index.setdefault(right_index, []).append((left_index, pixel_count))
        overlaps_by_key.setdefault(left_item["approval_key"], []).append(
            (right_item["approval_key"], right_index + 1, right_item["class_name"], pixel_count)
        )
        overlaps_by_key.setdefault(right_item["approval_key"], []).append(
            (left_item["approval_key"], left_index + 1, left_item["class_name"], pixel_count)
        )

    run_id = predictions.path.name
    notice_key = f"approve_all_notice_{run_id}_{frame_number}"
    candidate_rows = []
    for idx, item in enumerate(mask_items):
        if item["approval_key"] in st.session_state.synced_mask_ids:
            continue
        if not np.asarray(item["mask"]).any():
            st.session_state.approved_mask_ids.discard(item["approval_key"])
            st.session_state[f"approve_{run_id}_{item['approval_key']}"] = False
            continue
        candidate_rows.append((
            item["approval_key"],
            f"approve_{run_id}_{item['approval_key']}",
            float(item.get("confidence") or 0.0),
            tuple(entry[0] for entry in overlaps_by_key.get(item["approval_key"], [])),
        ))
    for approval_key, widget_key, _, _ in candidate_rows:
        if widget_key not in st.session_state:
            st.session_state[widget_key] = approval_key in st.session_state.approved_mask_ids
    sync_approved_mask_ids_from_widgets(predictions)

    st.markdown("**Duyệt mask để cho phép đồng bộ lên CVAT:**")
    select_nonoverlap_col, select_all_col = st.columns(2)
    with select_nonoverlap_col:
        if st.button(
            "Tích tất cả mask không chồng lấn",
            key=f"select_all_nonoverlap_masks_{run_id}_{frame_number}",
            disabled=not candidate_rows,
            help="Giữ nguyên mask đã chọn; bổ sung các mask không giao pixel, ưu tiên confidence cao hơn.",
            width="stretch",
        ):
            _select_all_non_overlapping_masks(notice_key, tuple(candidate_rows))
    with select_all_col:
        if st.button(
            "Tích tất cả mask, kể cả chồng lấn",
            key=f"select_all_masks_with_overlaps_{run_id}_{frame_number}",
            disabled=not candidate_rows,
            help="Tích mọi mask không rỗng trong frame đang xem, kể cả các mask giao pixel.",
            width="stretch",
        ):
            _select_all_masks_including_overlaps(notice_key, tuple(candidate_rows))
    st.caption(
        "Có thể tích và duyệt mask kể cả khi mask giao pixel. Nếu gửi đồng thời các mask chồng nhau, "
        "bước đồng bộ sẽ yêu cầu xử lý cặp giao nhau để tránh lớp chồng trong CVAT."
    )
    notice = st.session_state.pop(notice_key, None)
    if notice:
        st.info(notice)

    for start_index in range(0, len(mask_items), 2):
        columns = st.columns(2)
        for column, idx in zip(columns, range(start_index, min(start_index + 2, len(mask_items)))):
            item = mask_items[idx]
            qc = item["qc"]
            mask_is_empty = not bool(np.asarray(item["mask"]).any())
            approval_key = item["approval_key"]
            with column:
                with st.container(border=True):
                    st.markdown(f"**Mask {idx + 1} · {item['class_name']}**")
                    st.caption(
                        f"QC {qc.level} · confidence {float(item.get('confidence') or 0.0):.2f}"
                        + (f" · track {item['track_id']}" if item.get("track_id") is not None else "")
                    )
                    if int(item.get("merged_prediction_count", 1)) > 1:
                        st.caption(
                            f"Mask đã gộp {int(item['merged_prediction_count'])} mảnh cùng label để duyệt chung."
                        )
                    if approval_key in st.session_state.synced_mask_ids:
                        st.success("Đã đồng bộ lên CVAT")
                    elif mask_is_empty:
                        st.warning("Mask rỗng · không thể duyệt")
                    else:
                        widget_key = f"approve_{run_id}_{approval_key}"
                        if widget_key not in st.session_state:
                            st.session_state[widget_key] = (
                                approval_key in st.session_state.approved_mask_ids
                            )
                        error_key = f"approval_error_{run_id}_{approval_key}"
                        st.session_state.pop(error_key, None)
                        st.checkbox(
                            f"Duyệt · {item['class_name']}"
                            + (f" · track {item['track_id']}" if item.get("track_id") is not None else "")
                            + f" · QC {qc.level} · confidence {float(item.get('confidence') or 0.0):.2f}",
                            key=widget_key,
                            help="Chỉ mask được tích chọn mới được gửi lên CVAT.",
                            on_change=_on_mask_approval_change,
                            args=(approval_key, widget_key),
                        )
                    if item.get("confidence_source"):
                        st.caption(f"Confidence source: {item['confidence_source']}")
                    if float(item.get("confidence") or 0.0) < 0.4:
                        st.warning("Confidence thấp · kiểm tra kỹ mask này.")
                    item_overlaps = overlaps_by_index.get(idx, [])
                    if item_overlaps:
                        overlap_names = [
                            f"mask {other_index + 1} ({pixel_count:,} px)"
                            if pixel_count >= 0
                            else f"mask {other_index + 1} (kích thước khác)"
                            for other_index, pixel_count in item_overlaps[:3]
                        ]
                        if len(item_overlaps) > 3:
                            overlap_names.append(f"và {len(item_overlaps) - 3} mask khác")
                        st.caption(
                            "Mask này giao pixel với " + ", ".join(overlap_names)
                            + ". Mask vẫn có thể được tích duyệt riêng."
                        )
                    detail_key = f"review_detail_{run_id}_{approval_key}"
                    with st.expander(
                        f"Chi tiết mask {idx + 1}",
                        key=detail_key,
                        on_change="rerun",
                    ) as details:
                        if details.open:
                            st.caption(
                                f"QC: {qc.level} · confidence: {item['confidence']:.3f} · "
                                f"{int(np.asarray(item['mask']).sum()):,} pixel"
                            )
                            if qc.issues:
                                for issue in qc.issues:
                                    st.write(f"- {issue}")
                            st.image(
                                overlay_mask(image, item["mask"], class_color(item["class_name"]), alpha=0.5),
                                caption=f"{item['class_name']} mask",
                                width=image_width,
                            )


def show_prediction_review(
    predictions: PredictionStore,
    task_label_names: List[str],
    ctx: Dict[str, Any],
) -> None:
    """Hiển thị ảnh phủ mask và cho phép duyệt từng mask."""
    frame_names = predictions.frame_names
    if not frame_names:
        st.info("Chưa có ảnh dự đoán để duyệt.")
        return
    review_stuff_classes = get_stuff_classes(task_label_names)
    known_stuff = {name.casefold() for name in review_stuff_classes}
    review_stuff_classes.extend(
        name for name in task_label_names
        if is_region_label(name) and name.casefold() not in known_stuff
    )
    review_thing_classes = get_thing_classes(task_label_names)

    review_key = f"review_frame_{ctx['task_id']}_{ctx['job_id']}"
    current_name = st.session_state.get(review_key, frame_names[0])
    if current_name not in frame_names:
        current_name = frame_names[0]
    current_index = frame_names.index(current_name)
    run_id = predictions.path.name
    pending_mask_count = sum(
        1
        for frame_meta in predictions.iter_metadata()
        for item in frame_meta["masks"]
        if item["approval_key"] not in st.session_state.synced_mask_ids
    )
    with st.container(border=True):
        st.subheader("Duyệt nhanh tất cả ảnh")
        st.caption(
            f"{len(frame_names)} ảnh · {pending_mask_count} mask chưa đồng bộ. "
            "Các nút chỉ đánh dấu duyệt; chưa gửi dữ liệu lên CVAT."
        )
        bulk_notice_key = f"approve_all_frames_notice_{run_id}"
        approve_nonoverlap_col, approve_all_col, clear_all_col = st.columns(3)
        with approve_nonoverlap_col:
            if st.button(
                "Duyệt mask không chồng lấn · mọi ảnh",
                key=f"approve_nonoverlap_all_frames_{run_id}",
                disabled=pending_mask_count == 0,
                width="stretch",
                help="Chọn mask confidence cao hơn trước và bỏ qua mask giao pixel trong cùng frame.",
            ):
                stats = _select_masks_across_all_frames(predictions, include_overlaps=False)
                st.session_state[bulk_notice_key] = (
                    f"Đã duyệt {stats['selected']} mask trên mọi ảnh, thêm {stats['added']}; "
                    f"bỏ qua {stats['skipped_overlap']} mask chồng lấn."
                )
        with approve_all_col:
            if st.button(
                "Duyệt tất cả mask · mọi ảnh",
                key=f"approve_all_frames_{run_id}",
                disabled=pending_mask_count == 0,
                type="primary",
                width="stretch",
                help="Chọn mọi mask còn lại trên mọi ảnh, kể cả mask chồng lấn.",
            ):
                stats = _select_masks_across_all_frames(predictions, include_overlaps=True)
                st.session_state[bulk_notice_key] = (
                    f"Đã duyệt {stats['selected']} mask trên mọi ảnh, thêm {stats['added']}, "
                    "kể cả mask chồng lấn."
                )
        with clear_all_col:
            if st.button(
                "Bỏ duyệt mọi ảnh",
                key=f"clear_approvals_all_frames_{run_id}",
                disabled=pending_mask_count == 0,
                width="stretch",
                help="Bỏ chọn mọi mask chưa đồng bộ; mask đã gửi lên CVAT không bị thay đổi.",
            ):
                cleared = _clear_all_pending_approvals(predictions)
                st.session_state[bulk_notice_key] = f"Đã bỏ duyệt {cleared} mask chưa đồng bộ trên mọi ảnh."
        bulk_notice = st.session_state.pop(bulk_notice_key, None)
        if bulk_notice:
            st.success(bulk_notice)

    nav_left, selector, nav_right = st.columns([1, 4, 1])
    with nav_left:
        if st.button("← Prev", disabled=current_index == 0, key=f"prev_{ctx['task_id']}_{ctx['job_id']}"):
            st.session_state[review_key] = frame_names[current_index - 1]
            st.rerun()
    with nav_right:
        if st.button("Next →", disabled=current_index == len(frame_names) - 1, key=f"next_{ctx['task_id']}_{ctx['job_id']}"):
            st.session_state[review_key] = frame_names[current_index + 1]
            st.rerun()
    with selector:
        selected_name = st.selectbox("Ảnh xem trước", frame_names, index=current_index, key=review_key)
    prediction = predictions.load_frame(selected_name)
    frame_cleanup = prediction.get("postprocess_summary") or {}
    if frame_cleanup and any(int(frame_cleanup.get(key, 0)) for key in ("removed_masks", "removed_components", "merged_lane_masks")):
        st.caption(
            "Lọc ở frame này: "
            f"bỏ {int(frame_cleanup.get('removed_masks', 0))} mask nhỏ, "
            f"loại {int(frame_cleanup.get('removed_components', 0))} mảnh nhiễu, "
            f"gộp {int(frame_cleanup.get('merged_lane_masks', 0))} mask vạch đường."
        )
    image = prediction["image"]
    overlay = image.copy()
    for item in prediction["masks"]:
        overlay = overlay_mask(overlay, item["mask"], class_color(item["class_name"]), alpha=0.45)

    zoom = st.slider("Zoom ảnh", min_value=0.5, max_value=1.5, value=0.75, step=0.1)
    image_width = max(250, int(min(720, image.shape[1]) * zoom))
    image_col, overlay_col = st.columns([1, 1])
    with image_col:
        show_pan_zoom_view(image, "Ảnh gốc", zoom)
    with overlay_col:
        show_pan_zoom_view(overlay, "Mask dự đoán", zoom)

    if not prediction["masks"]:
        if is_facial_landmark_task(task_label_names):
            st.warning("Task này là Facial Landmark-50: các label là skeleton khuôn mặt, không phải lớp mask phân đoạn.")
            st.info("Dùng công cụ skeleton của CVAT để đặt keypoint. Model trong app chưa tạo dự đoán cho 50 facial landmark.")
        else:
            st.warning("Model không tìm thấy mask phù hợp với label của task trong ảnh này.")
            st.info("Kiểm tra ảnh và dùng công cụ vẽ của CVAT cho các lớp cần gán.")
        return

    suggested = {item["class_name"].casefold() for item in prediction["masks"]}
    not_suggested = [name for name in task_label_names if name.casefold() not in suggested]
    if not_suggested:
        st.info(
            "Model chưa đề xuất mask cho các label sau trong ảnh này: "
            + ", ".join(not_suggested)
            + ". Hãy kiểm tra bằng mắt; nếu label xuất hiện, vẽ/điều chỉnh trong CVAT."
        )

    mask_items = prediction["masks"]
    overlap_pairs = find_mask_overlap_pairs(mask_items)
    show_mask_approval_ui(
        predictions,
        mask_items,
        overlap_pairs,
        image,
        image_width,
        int(prediction["frame_number"]),
    )

    with st.expander("Chỉnh sửa mask thủ công (thêm / xóa vùng)"):
        selected_index = st.selectbox(
            "Mask cần sửa",
            range(len(prediction["masks"])),
            format_func=lambda index: f"{index + 1}. {prediction['masks'][index]['class_name']}",
            key=f"brush_item_{ctx['task_id']}_{prediction['frame_number']}",
        )
        selected = prediction["masks"][selected_index]
        brush_key = f"brush:{ctx['task_id']}:{ctx['job_id']}:{prediction['frame_number']}:{selected_index}"
        if selected["approval_key"] in st.session_state.synced_mask_ids:
            st.info("Mask này đã đồng bộ. Hãy chỉnh sửa trong CVAT để giữ trạng thái nhất quán.")
        else:
            polarity = st.segmented_control("Thao tác", ["Thêm vào mask", "Xóa khỏi mask"], key=f"{brush_key}:mode")
            radius = st.slider("Bán kính cọ (pixel ảnh)", 2, 64, 12, key=f"{brush_key}:radius")
            selected_overlay = overlay_mask(image, selected["mask"], class_color(selected["class_name"]), alpha=0.55)
            try:
                from streamlit_image_coordinates import streamlit_image_coordinates

                brush_event = streamlit_image_coordinates(
                    Image.fromarray(selected_overlay),
                    width=image_width,
                    key=f"{brush_key}:canvas",
                    cursor="crosshair",
                )
                event_key = f"{brush_key}:last_event"
                if brush_event and brush_event.get("unix_time") is not None and brush_event["unix_time"] != st.session_state.get(event_key):
                    display_w = max(float(brush_event.get("width", image_width)), 1.0)
                    display_h = max(float(brush_event.get("height", image.shape[0] * image_width / image.shape[1])), 1.0)
                    x = int(round(float(brush_event.get("x", 0)) * image.shape[1] / display_w))
                    y = int(round(float(brush_event.get("y", 0)) * image.shape[0] / display_h))
                    y_grid, x_grid = np.ogrid[:image.shape[0], :image.shape[1]]
                    disk = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= int(radius) ** 2
                    if polarity == "Thêm vào mask":
                        selected["mask"][disk] = 1
                    else:
                        selected["mask"][disk] = 0
                    selected["bbox"] = mask_bbox(selected["mask"])
                    selected["qc"] = QualityChecker(
                        stuff_classes=review_stuff_classes,
                        thing_classes=review_thing_classes,
                    ).check_single_mask(
                        selected_index,
                        selected["mask"],
                        selected["class_name"],
                        selected.get("is_thing", True),
                        image.shape[:2],
                        selected.get("confidence", 0.0),
                    )
                    predictions.save_frame(current_name, prediction, replace=True)
                    st.session_state.approved_mask_ids.discard(selected["approval_key"])
                    st.session_state[event_key] = brush_event["unix_time"]
                    st.rerun()
            except ImportError:
                st.info("Cài streamlit-image-coordinates để chỉnh sửa mask trên ảnh.")
            st.caption("Nhấp lên ảnh để thêm hoặc xóa một vùng tròn. Thay đổi được lưu theo frame; hãy kiểm tra lại rồi duyệt.")

    with st.expander("Tinh chỉnh mask bằng SAM 3 (positive / negative points)"):
        st.caption("Nhấp điểm trên ảnh để bổ sung prompt. Kết quả SAM 3 thay mask đang chọn sau khi bạn chủ động bấm tinh chỉnh.")
        if not ctx.get("label_names"):
            st.warning("Task chưa có label để gán.")
            return
        refine_index = st.selectbox(
            "Đối tượng cần tinh chỉnh",
            range(len(prediction["masks"])),
            format_func=lambda index: f"{index + 1}. {prediction['masks'][index]['class_name']}",
            key=f"refine_item_{ctx['task_id']}_{prediction['frame_number']}",
        )
        polarity = st.segmented_control("Loại điểm", ["Positive", "Negative"], key=f"refine_polarity_{ctx['task_id']}_{prediction['frame_number']}")
        refine_key = f"refine:{ctx['task_id']}:{ctx['job_id']}:{prediction['frame_number']}:{refine_index}"
        points_key = f"{refine_key}:points"
        gen_key = f"{refine_key}:generation"
        last_key = f"{refine_key}:last_event"
        st.session_state.setdefault(points_key, [])
        st.session_state.setdefault(gen_key, 0)
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates

            click = streamlit_image_coordinates(
                Image.fromarray(overlay),
                width=image_width,
                key=f"{refine_key}:canvas:{st.session_state[gen_key]}",
                cursor="crosshair",
            )
            _record_point_click(click, points_key, last_key, image.shape[1], image.shape[0], 1 if polarity == "Positive" else 0)
        except ImportError:
            st.info("Cài streamlit-image-coordinates để đặt điểm trực tiếp trên ảnh.")
        points = st.session_state[points_key]
        st.caption(f"Prompt points: {points}")
        clear_col, refine_col = st.columns(2)
        refine_device = st.session_state.get(
            "inference_device_v2",
            "cuda:0" if device_status.get("cuda_available") else "cpu",
        )
        refine_available = bool(device_status.get("cuda_available")) and str(refine_device).startswith("cuda")
        if not refine_available:
            st.caption("SAM 3 refine cần chọn GPU CUDA trong bước Tạo nhãn.")
        with clear_col:
            if st.button("Xóa điểm", key=f"clear_{refine_key}"):
                st.session_state[points_key] = []
                st.session_state[gen_key] += 1
                st.session_state.pop(last_key, None)
                st.rerun()
        with refine_col:
            if st.button("Tinh chỉnh mask bằng SAM 3", key=f"apply_{refine_key}", disabled=(not points or not refine_available)):
                try:
                    engine = get_sam3_model(
                        st.session_state.get("sam3_weights", str(ROOT / "models" / "sam3.pt")),
                        str(refine_device),
                    )
                    engine.set_confidence_threshold(
                        float(st.session_state.get("inference_confidence_threshold_v2", 0.35))
                    )
                    selected = prediction["masks"][refine_index]
                    result = engine.predict(
                        image,
                        bbox=selected.get("bbox") or mask_bbox(selected["mask"]),
                        point_coords=[(x, y) for x, y, _ in points],
                        point_labels=[label for _, _, label in points],
                        mask=selected["mask"],
                    )
                    if not result:
                        st.warning("SAM 3 không trả về mask cho các prompt này.")
                    else:
                        selected["mask"] = result[0]["mask"]
                        selected["bbox"] = mask_bbox(selected["mask"])
                        selected["qc"] = QualityChecker(
                            stuff_classes=review_stuff_classes,
                            thing_classes=review_thing_classes,
                        ).check_single_mask(
                            refine_index,
                            selected["mask"],
                            selected["class_name"],
                            selected.get("is_thing", True),
                            image.shape[:2],
                            selected.get("confidence", 0.0),
                        )
                        predictions.save_frame(current_name, prediction, replace=True)
                        st.session_state.approved_mask_ids.discard(selected["approval_key"])
                        st.session_state[points_key] = []
                        st.session_state[gen_key] += 1
                        st.success("Đã cập nhật mask. Hãy kiểm tra lại rồi duyệt đồng bộ.")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Không tinh chỉnh được bằng SAM 3: {exc}")


def _record_point_click(
    click: Optional[Dict[str, Any]],
    points_key: str,
    last_event_key: str,
    image_width: int,
    image_height: int,
    label: int,
) -> None:
    if not click or click.get("unix_time") is None:
        return
    event_time = click["unix_time"]
    if st.session_state.get(last_event_key) == event_time:
        return
    shown_width = max(float(click.get("width", image_width)), 1.0)
    shown_height = max(float(click.get("height", image_height)), 1.0)
    x = int(round(float(click.get("x", 0)) * image_width / shown_width))
    y = int(round(float(click.get("y", 0)) * image_height / shown_height))
    x = max(0, min(image_width - 1, x))
    y = max(0, min(image_height - 1, y))
    st.session_state.setdefault(points_key, []).append((x, y, int(label)))
    st.session_state[last_event_key] = event_time


def show_sync_ui(ctx: Dict[str, Any], client: CVATClient, predictions: PredictionStore) -> None:
    """Preview and append only approved shapes, preserving existing CVAT annotations."""
    job_id = ctx.get("job_id")
    if not job_id:
        st.warning("Task chưa có Job để nhận annotation.")
        return

    sync_approved_mask_ids_from_widgets(predictions)
    job = ctx.get("job") or {}
    start_frame = int(job.get("start_frame", 0))
    stop_frame = int(job.get("stop_frame", 2**31 - 1))
    existing_annotations = ctx.get("annotations") or {}
    existing_count = len(existing_annotations.get("shapes", []))
    existing_track_count = len(existing_annotations.get("tracks", []))
    st.caption(
        f"Job {job_id} hiện có {existing_count} shape và {existing_track_count} track. "
        "Annotation mới được thêm vào Job; hãy kiểm tra mask semantic không chồng pixel giữa các class."
    )

    label_ids = {
        annotation_label_key(name): label_id
        for name, label_id in ctx["label_ids"].items()
    }
    label_types = {
        annotation_label_key(label["name"]): canonical_cvat_shape_type(label.get("type"))
        for label in ctx["labels"]
        if label.get("name")
    }
    label_attribute_specs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for label in ctx["labels"]:
        if not label.get("name"):
            continue
        label_key = annotation_label_key(label["name"])
        specs = label.get("attributes") or []
        label_attribute_specs[label_key] = {
            normalize_export_label(spec.get("name", "")): spec
            for spec in specs
            if isinstance(spec, dict) and spec.get("name")
        }

    def truncated_spec_for(label_key: str) -> Dict[str, Any]:
        spec = label_attribute_specs.get(label_key, {}).get(
            normalize_export_label("truncated"), {}
        )
        raw_values = spec.get("values") or []
        if isinstance(raw_values, str):
            raw_values = raw_values.splitlines()
        values = {
            str(value).strip().casefold()
            for value in raw_values
        }
        input_type = str(spec.get("input_type", "")).strip().casefold()
        is_boolean = input_type == "checkbox" or {"true", "false"}.issubset(values)
        mutable_value = spec.get("mutable", True)
        mutable = (
            mutable_value
            if isinstance(mutable_value, bool)
            else str(mutable_value).strip().casefold() not in ("0", "false", "no")
        )
        if spec.get("id") is not None and mutable and is_boolean:
            return spec
        return {}

    approved: List[Tuple[str, int, int, Dict[str, Any]]] = []
    for frame_meta in predictions.iter_metadata():
        task_frame = int(frame_meta["frame_number"])
        if not start_frame <= task_frame <= stop_frame:
            continue
        for mask_index, item in enumerate(frame_meta["masks"]):
            key = item["approval_key"]
            if key in st.session_state.approved_mask_ids and key not in st.session_state.synced_mask_ids:
                # CVAT annotation shape.frame uses the task frame number even
                # when the target Job starts partway through the task.
                approved.append((frame_meta["name"], task_frame, mask_index, item))
    if not approved:
        st.info("Chưa có annotation mới được duyệt để đồng bộ.")
        return

    pending_by_label: Dict[str, List[Dict[str, Any]]] = {}
    display_names: Dict[str, str] = {}
    approved_by_key: Dict[str, Dict[str, Any]] = {}
    for row in approved:
        item = row[3]
        label_key = annotation_label_key(item["class_name"])
        pending_by_label.setdefault(label_key, []).append(item)
        display_names.setdefault(label_key, item["class_name"])
        approved_by_key[item["approval_key"]] = item

    all_label_names = [label.get("name", "") for label in ctx["labels"] if label.get("name")]
    inferred_preset = detect_export_preset(all_label_names)
    preset_key = f"sync_shape_preset_{job_id}"
    applied_preset_key = f"sync_shape_preset_applied_{job_id}"
    shape_map_key = f"sync_shape_map_{job_id}"
    if preset_key not in st.session_state:
        st.session_state[preset_key] = inferred_preset
    if shape_map_key not in st.session_state:
        st.session_state[shape_map_key] = {}

    shape_map: Dict[str, str] = st.session_state[shape_map_key]
    st.subheader("Kiểu shape theo label")
    st.caption(
        "Mỗi label có kiểu hình học riêng. Một frame có thể đồng bộ đồng thời box, polygon và polyline. "
        "Các preset chỉ chọn cách chuyển mask; không thay đổi label đã cấu hình trong CVAT."
    )
    preset = st.selectbox(
        "Preset annotation",
        options=list(PRESET_LABELS),
        format_func=lambda value: PRESET_LABELS[value],
        key=preset_key,
        help="Preset đường phố theo guideline bounding box/polygon/polyline hoặc polygon cho semantic segmentation.",
    )
    reset_label_shapes = st.session_state.get(applied_preset_key) != preset
    if reset_label_shapes:
        if preset != "custom":
            shape_map = {
                annotation_label_key(label["name"]): recommend_export_shape(
                    label["name"],
                    declared_type=label.get("type", "any"),
                    preset=preset,
                )
                for label in ctx["labels"]
                if label.get("name")
            }
        st.session_state[shape_map_key] = shape_map
        st.session_state[applied_preset_key] = preset

    shape_display = {shape_type: label for label, shape_type in SHAPE_TYPES.items()}
    shape_order = ["mask", "polygon", "rectangle", "polyline", "skeleton"]
    available_by_label: Dict[str, List[str]] = {}
    unavailable_reasons: Dict[str, str] = {}
    for label_key, items in pending_by_label.items():
        declared_type = label_types.get(label_key, "any")
        segmentation_available = all(
            bool(item.get("segmentation_available", True)) for item in items
        )
        keypoint_available = bool(items) and all(bool(item.get("keypoints")) for item in items)
        label_id = label_ids.get(label_key)
        label = next(
            (entry for entry in ctx["labels"] if entry.get("id") == label_id),
            {},
        )
        has_sublabels = bool(get_cvat_sublabels(ctx["labels"], label))
        candidates = (
            [declared_type]
            if declared_type != "any"
            else list(shape_order[:-1])
        )
        if declared_type in ("any", "skeleton") and keypoint_available and has_sublabels:
            candidates.append("skeleton")
        expected_type = guideline_shape_for_label(
            display_names[label_key], preset
        ) if preset != "custom" else None
        if (
            expected_type
            and declared_type not in ("any", expected_type)
        ):
            unavailable_reasons[label_key] = (
                f"guideline yêu cầu {expected_type}, nhưng label CVAT khai báo {declared_type}"
            )
            candidates = []
        requires_instance_box = (
            shape_map.get(label_key) == "rectangle"
            and any(item.get("is_thing") is False for item in items)
        )
        if requires_instance_box:
            unavailable_reasons.setdefault(
                label_key,
                "prediction gộp nhiều pixel semantic, không tách được box riêng cho từng object; "
                "hãy chạy model instance rồi duyệt lại",
            )
            candidates = []
        available_by_label[label_key] = [
            shape_type
            for shape_type in candidates
            if shape_type in shape_display
            and (
                shape_type not in ("mask", "polygon", "polyline")
                or segmentation_available
            )
            and (shape_type != "skeleton" or (keypoint_available and has_sublabels))
            and label_id is not None
        ]

    unavailable_labels: List[str] = []
    with st.expander("Tùy chỉnh kiểu shape cho từng label", expanded=False):
        for label_key in sorted(pending_by_label, key=lambda key: display_names[key].casefold()):
            label_name = display_names[label_key]
            options = available_by_label[label_key]
            if not options:
                unavailable_labels.append(label_name)
                reason = unavailable_reasons.get(
                    label_key,
                    "không có shape tương thích với label/prediction trong CVAT",
                )
                st.caption(f"{label_name}: {reason}.")
                continue
            stored_type = shape_map.get(label_key)
            recommended_type = recommend_export_shape(
                label_name,
                declared_type=label_types.get(label_key, "any"),
                preset=preset,
                segmentation_available=all(
                    bool(item.get("segmentation_available", True))
                    for item in pending_by_label[label_key]
                ),
            )
            selected_type = stored_type if stored_type in options else recommended_type
            if selected_type not in options:
                selected_type = options[0]
            widget_key = f"sync_shape_{job_id}_{hashlib.sha1(label_key.encode('utf-8')).hexdigest()[:10]}"
            if reset_label_shapes or widget_key not in st.session_state or st.session_state.get(widget_key) not in options:
                st.session_state[widget_key] = selected_type
            shape_map[label_key] = st.selectbox(
                f"{label_name} · {len(pending_by_label[label_key])} annotation",
                options=options,
                format_func=lambda value: shape_display[value],
                key=widget_key,
            )
    st.session_state[shape_map_key] = shape_map
    if unavailable_labels:
        st.warning("Cần đổi kiểu prediction hoặc label CVAT cho: " + ", ".join(unavailable_labels))

    if any(shape_type == "polyline" for shape_type in shape_map.values()):
        min_polyline_length_px = st.slider(
            "Bỏ polyline ngắn hơn (pixel ảnh)",
            min_value=8,
            max_value=160,
            value=32,
            step=4,
            key=f"sync_min_polyline_length_{job_id}",
            help="Loại các nét vạch quá ngắn/nhiễu; giảm nếu cần giữ vạch đứt ngắn ở xa.",
        )
    else:
        min_polyline_length_px = 32
    max_shapes_per_frame = int(st.number_input(
        "Giới hạn shape mỗi frame khi đồng bộ",
        min_value=10,
        max_value=5000,
        value=250,
        step=10,
        key=f"sync_max_shapes_per_frame_{job_id}",
        help="Nếu số shape vượt giới hạn, tool dừng trước khi gửi để bạn kiểm tra hoặc tăng ngưỡng lọc.",
    ))

    all_prediction_items = [
        item
        for frame_meta in predictions.iter_metadata()
        for item in frame_meta["masks"]
    ]
    is_video_track_run = any(
        frame_meta.get("mode") == "SAM3 Video Track"
        for frame_meta in predictions.iter_metadata()
    )
    track_export_ready = bool(all_prediction_items) and all(
        item.get("track_id") is not None
        and label_types.get(annotation_label_key(item["class_name"]), "any") in ("any", "mask")
        and item["approval_key"] in st.session_state.approved_mask_ids
        and item["approval_key"] not in st.session_state.synced_mask_ids
        for item in all_prediction_items
    )
    if is_video_track_run and track_export_ready:
        export_mode = st.segmented_control(
            "Cách đồng bộ video",
            options=["Theo shape từng label", "Mask track"],
            key=f"sync_export_mode_{job_id}",
            help="Mask track giữ track ID qua các frame; chế độ theo label cho phép chọn hình học riêng cho từng class.",
        )
    elif is_video_track_run:
        export_mode = "Theo shape từng label"
        st.caption("Để tạo CVAT track, hãy duyệt mọi mask trong run video; partial approval vẫn có thể đồng bộ thành Mask per-frame.")
    else:
        export_mode = "Theo shape từng label"
    use_mask_tracks = export_mode == "Mask track"

    preview_rows: List[Dict[str, Any]] = []
    for frame_name, task_frame, mask_index, item in approved:
        label_key = annotation_label_key(item["class_name"])
        selected_type = "mask" if use_mask_tracks else shape_map.get(label_key)
        shape_name = "Mask track" if use_mask_tracks else shape_display.get(selected_type, "Không tương thích")
        preview_rows.append({
            "_approval_key": item["approval_key"],
            "Frame": task_frame,
            "Label": item["class_name"],
            "Confidence": round(float(item.get("confidence") or 0), 3),
            "QC": item["qc"]["level"],
            "Track ID": item.get("track_id"),
            "Shape": shape_name,
            "Occluded": False,
            "Truncated": False,
        })

    truncated_without_attribute = [
        row["Label"]
        for row in preview_rows
        if not truncated_spec_for(annotation_label_key(row["Label"]))
    ]
    st.write(f"Preview · {len(preview_rows)} mask được duyệt")
    st.caption(
        "Một mask có thể sinh nhiều shape nếu chứa nhiều vùng tách rời. Tool lọc polyline ngắn và chặn đồng bộ nếu một frame vượt giới hạn shape."
        " Có thể gắn cờ occluded cho từng object; Truncated được gửi nếu label CVAT có attribute tương ứng."
    )
    if truncated_without_attribute:
        st.info(
            "Muốn lưu Truncated, hãy khai báo checkbox attribute `truncated` cho label đó trong CVAT. "
            "Nếu đánh dấu mà thiếu attribute, tool sẽ dừng trước khi gửi để tránh mất thông tin."
        )
    fingerprint = hashlib.sha1(
        "|".join(row["_approval_key"] for row in preview_rows).encode("utf-8")
    ).hexdigest()[:10]
    edited_preview = st.data_editor(
        preview_rows,
        key=f"sync_review_{job_id}_{fingerprint}",
        column_config={"_approval_key": None},
        disabled=["_approval_key", "Frame", "Label", "Confidence", "QC", "Track ID", "Shape"],
        hide_index=True,
        num_rows="fixed",
        width="stretch",
        alt="Preview annotation và trạng thái occluded/truncated trước khi đồng bộ CVAT",
    )
    review_by_key = {str(row["_approval_key"]): row for row in edited_preview}
    confirmation_state = [
        (
            str(row["_approval_key"]),
            row["Shape"],
            bool(row.get("Occluded", False)),
            bool(row.get("Truncated", False)),
        )
        for row in edited_preview
    ]
    confirmation_signature = hashlib.sha1(
        json.dumps(
            {
                "rows": confirmation_state,
                "shape_map": shape_map,
                "min_polyline_length_px": min_polyline_length_px,
                "max_shapes_per_frame": max_shapes_per_frame,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:10]
    confirmed = st.checkbox(
        "Tôi đã kiểm tra preview và xác nhận đồng bộ",
        key=f"sync_preview_{job_id}_{confirmation_signature}",
    )
    if not st.button("Đồng bộ annotation lên CVAT", type="primary", disabled=not confirmed):
        return

    missing_truncated_attributes = sorted({
        str(row["Label"])
        for row in edited_preview
        if bool(row.get("Truncated"))
        and not truncated_spec_for(annotation_label_key(row["Label"]))
    })
    if missing_truncated_attributes:
        st.error(
            "CVAT chưa khai báo attribute `truncated` cho: "
            + ", ".join(missing_truncated_attributes)
            + ". Hãy thêm attribute đó rồi đồng bộ lại."
        )
        return
    invalid_occluded = sorted({
        str(row["Label"])
        for row in edited_preview
        if bool(row.get("Occluded"))
        and not approved_by_key.get(str(row["_approval_key"]), {}).get("is_thing", False)
    })
    if invalid_occluded:
        st.error(
            "Occluded chỉ áp dụng cho prediction instance. Hãy bỏ chọn ở các vùng semantic: "
            + ", ".join(invalid_occluded)
        )
        return

    shapes: List[Dict[str, Any]] = []
    tracks_by_id: Dict[Tuple[int, int], Dict[str, Any]] = {}
    synced_keys: List[str] = []
    skipped: List[str] = []
    grouped: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = {}
    for frame_name, job_frame, mask_index, metadata in approved:
        grouped.setdefault(frame_name, []).append((job_frame, mask_index, metadata))

    for frame_name, frame_items in grouped.items():
        prediction = predictions.load_frame(frame_name)
        for job_frame, mask_index, metadata in frame_items:
            item = prediction["masks"][mask_index]
            export_mask = np.asarray(item["mask"], dtype=np.uint8)
            normalized_name = annotation_label_key(item["class_name"])
            label_id = label_ids.get(normalized_name)
            declared_type = label_types.get(normalized_name, "any")
            shape_type = "mask" if use_mask_tracks else shape_map.get(normalized_name)
            review = review_by_key.get(item["approval_key"], {})
            is_occluded = bool(review.get("Occluded", False))
            attributes: List[Dict[str, Any]] = []
            truncated_spec = truncated_spec_for(normalized_name)
            if truncated_spec:
                attributes.append({
                    "spec_id": int(truncated_spec["id"]),
                    "value": "true" if bool(review.get("Truncated", False)) else "false",
                })
            if not np.asarray(export_mask).any():
                skipped.append(f"{item['class_name']} (mask rỗng)")
                continue
            if normalized_name in unavailable_reasons:
                skipped.append(f"{item['class_name']} ({unavailable_reasons[normalized_name]})")
                continue
            if use_mask_tracks:
                if label_id is None or declared_type not in ("any", "mask"):
                    skipped.append(f"{item['class_name']} (label type {declared_type})")
                    continue
                track_id = item.get("track_id")
                if track_id is None:
                    skipped.append(f"{item['class_name']} (không có track ID)")
                    continue
                track_key = (int(track_id), int(label_id))
                track = tracks_by_id.setdefault(track_key, {
                    "label_id": int(label_id),
                    "frame": job_frame,
                    "group": 0,
                    "source": "auto",
                    "attributes": [],
                    "shapes": [],
                })
                track["frame"] = min(int(track["frame"]), job_frame)
                track["shapes"].append({
                    "type": "mask",
                    "frame": job_frame,
                    "points": mask_to_cvat_points(export_mask),
                    "occluded": is_occluded,
                    "outside": False,
                    "keyframe": True,
                    "z_order": 0,
                    "rotation": 0,
                    "attributes": attributes,
                })
                synced_keys.append(item["approval_key"])
                continue
            if shape_type == "skeleton":
                if declared_type not in ("any", "skeleton"):
                    skipped.append(f"{item['class_name']} (label type {declared_type})")
                    continue
                label = next(
                    (row for row in ctx["labels"] if int(row.get("id", -1)) == int(label_id or -1)),
                    {},
                )
                sublabels = get_cvat_sublabels(ctx["labels"], label)
                skeleton = keypoints_to_cvat_skeleton(
                    item.get("keypoints") or [],
                    frame=job_frame,
                    label_id=int(label_id or -1),
                    sublabels=sublabels,
                )
                if skeleton is None:
                    skipped.append(f"{item['class_name']} (thiếu keypoint khớp sublabels CVAT)")
                    continue
                skeleton["occluded"] = is_occluded
                skeleton["attributes"] = attributes
                shapes.append(skeleton)
                synced_keys.append(item["approval_key"])
                continue
            if shape_type is None or label_id is None or declared_type not in ("any", shape_type):
                skipped.append(f"{item['class_name']} (label type {declared_type})")
                continue
            if shape_type == "mask":
                if not item.get("segmentation_available", True):
                    skipped.append(f"{item['class_name']} (không có mask, chỉ có bounding box)")
                    continue
                point_sets = [mask_to_cvat_points(export_mask)]
            elif shape_type == "rectangle":
                point_sets = [item.get("bbox") or mask_bbox(item["mask"])]
            elif shape_type == "polyline":
                if not item.get("segmentation_available", True):
                    skipped.append(f"{item['class_name']} (không có mask để tạo centerline)")
                    continue
                point_sets = mask_to_cvat_polylines(
                    export_mask,
                    min_path_length=min_polyline_length_px,
                )
            else:
                if not item.get("segmentation_available", True):
                    skipped.append(f"{item['class_name']} (không có mask để tạo polygon)")
                    continue
                point_sets = [
                    polygon for polygon in mask_to_polygons(
                        export_mask,
                        simplify_tolerance=1.5,
                        min_contour_area_px=32,
                    )
                    if len(polygon) >= 6
                ]
            if not point_sets or any(not points for points in point_sets):
                skipped.append(f"{item['class_name']} (shape rỗng)")
                continue
            for points in point_sets:
                shapes.append({
                    "type": shape_type,
                    "frame": job_frame,
                    "label_id": label_id,
                    "group": 0,
                    "source": "auto",
                    "occluded": is_occluded,
                    "outside": False,
                    "z_order": 0,
                    "rotation": 0,
                    "points": points,
                    "attributes": attributes,
                })
            synced_keys.append(item["approval_key"])

    tracks = list(tracks_by_id.values())
    if not shapes and not tracks:
        if skipped:
            st.warning("Đã bỏ qua: " + "; ".join(sorted(set(skipped))))
        st.error("Không có shape hợp lệ để gửi.")
        return

    shapes_per_frame: Dict[int, int] = {}
    for shape in shapes:
        frame = int(shape.get("frame", -1))
        shapes_per_frame[frame] = shapes_per_frame.get(frame, 0) + 1
    for track in tracks:
        for keyframe in track.get("shapes", []):
            frame = int(keyframe.get("frame", -1))
            shapes_per_frame[frame] = shapes_per_frame.get(frame, 0) + 1
    over_limit = [
        {"Frame": frame, "Shape": count, "Giới hạn": max_shapes_per_frame}
        for frame, count in sorted(shapes_per_frame.items())
        if count > max_shapes_per_frame
    ]
    if over_limit:
        st.error(
            f"Chưa đồng bộ: có {len(over_limit)} frame vượt giới hạn "
            f"{max_shapes_per_frame} shape/frame. Không có annotation nào được gửi."
        )
        st.dataframe(over_limit, hide_index=True, width="stretch")
        st.info(
            "Tăng ngưỡng confidence/diện tích tối thiểu khi chạy inference hoặc tăng "
            "độ dài polyline tối thiểu. Chỉ nâng giới hạn shape nếu đã kiểm tra các nét đó cần giữ."
        )
        return
    try:
        if use_mask_tracks:
            result = client.append_job_tracks(int(job_id), tracks)
            annotation_name = "track"
        else:
            result = client.append_job_shapes(int(job_id), shapes)
            annotation_name = "shape"
        st.session_state.synced_mask_ids.update(synced_keys)
        st.success(
            f"CVAT Job {job_id}: thêm {result['added']} {annotation_name}; "
            f"bỏ qua {result['duplicates']} trùng. Annotation cũ được giữ."
        )
        if skipped:
            st.warning("Đã bỏ qua: " + "; ".join(sorted(set(skipped))))
    except Exception as exc:
        logger.exception("CVAT annotation upload failed")
        st.error(f"Không đồng bộ được annotation: {exc}")


def show_frame_cleanup_ui(ctx: Dict[str, Any], client: CVATClient) -> None:
    """Review and delete standalone CVAT annotations from one selected frame."""
    job_id = ctx.get("job_id")
    if not job_id:
        st.warning("Task chưa có Job để thao tác annotation.")
        return

    st.header("5. Xóa nhãn của frame")
    st.caption(
        "Chọn frame theo số CVAT để xóa toàn bộ shape và tag đang gán ở frame đó. "
        "Ảnh và annotation ở các frame khác được giữ nguyên."
    )
    notice_key = f"cleanup_notice_{job_id}"
    notice = st.session_state.pop(notice_key, None)
    if notice:
        st.success(notice)
    bulk_notice_key = f"cleanup_bulk_notice_{job_id}"
    bulk_notice = st.session_state.pop(bulk_notice_key, None)
    if bulk_notice:
        st.success(bulk_notice)

    frames = ctx.get("frames") or []
    frame_by_number = {
        frame_number(frame, index): frame
        for index, frame in enumerate(frames)
    }
    frame_numbers = sorted(frame_by_number)
    if not frame_numbers:
        st.warning("Job không có frame để chọn.")
        return

    selected_frame = st.selectbox(
        "Frame cần dọn",
        options=frame_numbers,
        format_func=lambda number: (
            f"Frame {number} · {frame_by_number[number].get('name', 'không rõ tên')}"
        ),
        key=f"cleanup_frame_{ctx['task_id']}_{job_id}",
    )
    try:
        annotations = client.get_job_annotations(int(job_id))
    except Exception as exc:
        st.error(f"Không tải được annotation mới nhất của Job {job_id}: {exc}")
        return
    ctx["annotations"] = annotations

    bulk_counts = {
        "shapes": len(annotations.get("shapes", [])),
        "tags": len(annotations.get("tags", [])),
        "tracks": len(annotations.get("tracks", [])),
        "intervals": len(annotations.get("intervals", [])),
    }
    bulk_total = sum(bulk_counts.values())
    bulk_fingerprint = client.job_annotations_fingerprint(annotations)
    bulk_signature = bulk_fingerprint[:12]
    with st.container(border=True):
        st.subheader("Xóa nhãn trên tất cả ảnh của Job")
        st.caption(
            f"Job này có {len(frame_numbers)} ảnh và {bulk_total} mục annotation "
            f"({bulk_counts['shapes']} shape, {bulk_counts['tags']} tag, "
            f"{bulk_counts['tracks']} track, {bulk_counts['intervals']} interval). "
            "Thao tác xóa nhãn trên mọi frame, nhưng giữ nguyên ảnh và frame trong CVAT."
        )
        bulk_confirmed = st.checkbox(
            f"Tôi xác nhận xóa toàn bộ {bulk_total} mục annotation của Job {job_id}",
            key=f"cleanup_bulk_confirm_{job_id}_{bulk_signature}",
            disabled=bulk_total == 0,
        )
        if st.button(
            "Xóa toàn bộ nhãn của Job",
            type="primary",
            disabled=bulk_total == 0 or not bulk_confirmed,
            key=f"cleanup_bulk_submit_{job_id}_{bulk_signature}",
        ):
            try:
                result = client.delete_all_job_annotations(
                    int(job_id),
                    expected_fingerprint=bulk_fingerprint,
                )
                ctx["annotations"] = result["annotations"]
                st.session_state["task_context"]["annotations"] = result["annotations"]
                deleted_total = sum(
                    result[key]
                    for key in (
                        "deleted_shapes",
                        "deleted_tags",
                        "deleted_tracks",
                        "deleted_intervals",
                    )
                )
                st.session_state[bulk_notice_key] = (
                    f"Đã xóa {deleted_total} mục annotation trên tất cả {len(frame_numbers)} ảnh "
                    f"của CVAT Job {job_id}. Ảnh và frame vẫn được giữ nguyên."
                )
                st.rerun()
            except Exception as exc:
                logger.exception("CVAT job annotation deletion failed")
                st.error(f"Không xóa được toàn bộ annotation của Job {job_id}: {exc}")

    frame_shapes = [
        shape for shape in annotations.get("shapes", [])
        if int(shape.get("frame", -1)) == int(selected_frame)
    ]
    frame_tags = [
        tag for tag in annotations.get("tags", [])
        if int(tag.get("frame", -1)) == int(selected_frame)
    ]
    frame_tracks = [
        track for track in annotations.get("tracks", [])
        if client.track_covers_frame(track, int(selected_frame))
    ]
    labels_by_id = {
        int(label["id"]): label["name"]
        for label in ctx.get("labels", [])
        if label.get("id") is not None and label.get("name")
    }

    metric_shape, metric_tag, metric_track = st.columns(3)
    metric_shape.metric("Shape", len(frame_shapes))
    metric_tag.metric("Tag", len(frame_tags))
    metric_track.metric("Track đang chạy qua frame", len(frame_tracks))

    rows = [
        {
            "ID": item.get("id"),
            "Loại": item.get("type", "tag"),
            "Nhãn": labels_by_id.get(int(item.get("label_id", -1)), f"label:{item.get('label_id')}"),
            "Source": item.get("source", ""),
        }
        for item in [*frame_shapes, *frame_tags]
    ]
    if rows:
        st.dataframe(
            rows,
            hide_index=True,
            width="stretch",
            alt=f"Các shape và tag hiện có trên CVAT frame {selected_frame}",
        )
    elif not frame_tracks and not notice:
        st.success("Frame này hiện không có nhãn.")

    if frame_tracks:
        track_rows = [
            {
                "Track ID": track.get("id"),
                "Nhãn": labels_by_id.get(int(track.get("label_id", -1)), f"label:{track.get('label_id')}"),
                "Keyframe trong frame": any(
                    int(shape.get("frame", -1)) == int(selected_frame)
                    for shape in track.get("shapes", [])
                ),
            }
            for track in frame_tracks
        ]
        st.dataframe(
            track_rows,
            hide_index=True,
            width="stretch",
            alt=f"Các track có annotation trên CVAT frame {selected_frame}",
        )
        st.warning(
            "Frame này nằm trong ít nhất một track. Xóa cả track sẽ làm mất nhãn ở các frame khác, "
            "nên thao tác dọn frame đang bị khóa. Hãy chỉnh track trong CVAT."
        )

    selected_objects = [*frame_shapes, *frame_tags]
    selected_ids = [str(item.get("id")) for item in selected_objects if item.get("id") is not None]
    has_missing_ids = len(selected_ids) != len(selected_objects)
    if has_missing_ids:
        st.error("CVAT không trả ID cho mọi nhãn ở frame này; không thể xóa an toàn.")
    signature = hashlib.sha1(
        json.dumps(
            {
                "frame": int(selected_frame),
                "ids": sorted(selected_ids),
                "tracks": sorted(str(track.get("id")) for track in frame_tracks),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    can_delete = bool(selected_objects) and not frame_tracks and not has_missing_ids
    confirmed = st.checkbox(
        f"Tôi xác nhận xóa {len(frame_shapes)} shape và {len(frame_tags)} tag ở frame {selected_frame}",
        key=f"cleanup_confirm_{job_id}_{selected_frame}_{signature}",
        disabled=not can_delete,
    )
    if st.button(
        "Xóa sạch nhãn trên frame này",
        type="primary",
        disabled=not can_delete or not confirmed,
        key=f"cleanup_submit_{job_id}_{selected_frame}_{signature}",
    ):
        try:
            result = client.delete_job_frame_annotations(
                int(job_id),
                int(selected_frame),
                expected_ids=selected_ids,
            )
            ctx["annotations"] = result["annotations"]
            st.session_state["task_context"]["annotations"] = result["annotations"]
            st.session_state[notice_key] = (
                f"Frame {selected_frame}: đã xóa {result['deleted_shapes']} shape "
                f"và {result['deleted_tags']} tag khỏi CVAT Job {job_id}."
            )
            st.rerun()
        except Exception as exc:
            logger.exception("CVAT frame annotation deletion failed")
            st.error(f"Không xóa được annotation của frame {selected_frame}: {exc}")


# ---------------------------------------------------------------------------
# Khu vực: Helper functions
# ---------------------------------------------------------------------------
def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.5) -> np.ndarray:
    """Phủ mask lên ảnh với màu cho trước."""
    overlay = image.copy()
    colored = np.zeros_like(image)
    colored[mask > 0] = color
    overlay = (overlay * (1 - alpha) + colored * alpha).astype(np.uint8)
    return overlay


def show_pan_zoom_view(image: np.ndarray, title: str, initial_zoom: float) -> None:
    """Lossless aspect-ratio viewer with wheel/button zoom and drag-to-pan."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=True)
    source = base64.b64encode(buffer.getvalue()).decode("ascii")
    html = r"""<!doctype html><html><head><meta charset="utf-8"><style>
      *{box-sizing:border-box}body{margin:0;font:13px sans-serif;color:#ddd;background:#111}
      .bar{height:34px;display:flex;align-items:center;gap:8px;padding:4px 8px;background:#20242b}
      .bar span{margin-right:auto}.bar button{background:#343a46;color:white;border:0;border-radius:4px;padding:4px 9px;cursor:pointer}
      .view{height:440px;position:relative;overflow:hidden;background:#101318;cursor:grab;touch-action:none}
      .view.dragging{cursor:grabbing}.view img{position:absolute;max-width:none;max-height:none;transform-origin:0 0;user-select:none;-webkit-user-drag:none}
    </style></head><body><div class="bar"><span>__TITLE__</span><button id="minus">−</button><button id="plus">+</button><button id="reset">Fit</button></div>
    <div class="view" id="view"><img id="img" src="data:image/png;base64,__SOURCE__"></div>
    <script>
      const view=document.getElementById('view'),img=document.getElementById('img');let scale=1,fit=1,x=0,y=0,drag=null;
      function paint(){img.style.transform=`translate(${x}px,${y}px) scale(${scale})`}
      function fitImage(){if(!img.naturalWidth)return;fit=Math.min((view.clientWidth-24)/img.naturalWidth,(view.clientHeight-24)/img.naturalHeight);scale=fit*__ZOOM__;x=(view.clientWidth-img.naturalWidth*scale)/2;y=(view.clientHeight-img.naturalHeight*scale)/2;paint()}
      img.onload=fitImage;new ResizeObserver(fitImage).observe(view);
      function zoomAt(factor,cx=view.clientWidth/2,cy=view.clientHeight/2){const wx=(cx-x)/scale,wy=(cy-y)/scale;scale=Math.max(fit*.2,Math.min(fit*8,scale*factor));x=cx-wx*scale;y=cy-wy*scale;paint()}
      view.addEventListener('wheel',e=>{e.preventDefault();const r=view.getBoundingClientRect();zoomAt(e.deltaY<0?1.12:.89,e.clientX-r.left,e.clientY-r.top)},{passive:false});
      view.addEventListener('pointerdown',e=>{drag=[e.clientX-x,e.clientY-y];view.classList.add('dragging');view.setPointerCapture(e.pointerId)});
      view.addEventListener('pointermove',e=>{if(drag){x=e.clientX-drag[0];y=e.clientY-drag[1];paint()}});
      view.addEventListener('pointerup',()=>{drag=null;view.classList.remove('dragging')});
      document.getElementById('plus').onclick=()=>zoomAt(1.25);document.getElementById('minus').onclick=()=>zoomAt(.8);document.getElementById('reset').onclick=fitImage;
    </script></body></html>"""
    html = html.replace("__TITLE__", title).replace("__SOURCE__", source).replace("__ZOOM__", str(float(initial_zoom)))
    components.html(html, height=478, scrolling=False)


def class_color(class_name: str) -> Tuple[int, int, int]:
    """Màu theo class (dùng cho hiển thị)."""
    color_map = {
        "road": (128, 64, 128),
        "sidewalk": (244, 35, 232),
        "building": (70, 70, 70),
        "wall": (102, 102, 156),
        "fence": (190, 153, 153),
        "pole": (153, 153, 153),
        "traffic light": (250, 170, 30),
        "traffic sign": (220, 220, 0),
        "vegetation": (107, 142, 35),
        "terrain": (152, 251, 152),
        "sky": (70, 130, 180),
        "person": (220, 20, 60),
        "rider": (255, 0, 0),
        "bicycle": (119, 11, 32),
        "car": (0, 0, 142),
        "motorcycle": (0, 0, 230),
        "bus": (0, 60, 100),
        "truck": (0, 0, 70),
        "train": (0, 80, 100),
    }
    return color_map.get(class_name, (128, 128, 128))


# ---------------------------------------------------------------------------
# CVAT connection and task selection
# ---------------------------------------------------------------------------
_CANVAS_JS = r"""
export default function (component) {
  const { data, parentElement, setStateValue } = component;
  const canvas = parentElement.querySelector("#annotation-canvas");
  const finishButton = parentElement.querySelector("#finish-polygon");
  const undoButton = parentElement.querySelector("#undo-shape");
  const clearButton = parentElement.querySelector("#clear-shapes");
  const help = parentElement.querySelector("#canvas-help");
  if (!canvas || !finishButton || !undoButton || !clearButton) return;

  const ctx = canvas.getContext("2d");
  const image = new Image();
  let shapes = Array.isArray(data.shapes) ? JSON.parse(JSON.stringify(data.shapes)) : [];
  let draft = [];
  let dragStart = null;
  let dragEnd = null;
  let hoverPoint = null;
  const tool = data.tool || "polygon";
  const color = data.color || "#e45756";

  function toPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(canvas.width - 1, (event.clientX - rect.left) * canvas.width / rect.width)),
      y: Math.max(0, Math.min(canvas.height - 1, (event.clientY - rect.top) * canvas.height / rect.height))
    };
  }

  function paintPolygon(points, stroke, fill, width) {
    if (!points || points.length === 0) return;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i += 1) ctx.lineTo(points[i].x, points[i].y);
    if (points.length > 2) ctx.closePath();
    if (fill) {
      ctx.globalAlpha = 0.28;
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.stroke();
    for (const point of points) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, width * 1.5, 0, Math.PI * 2);
      ctx.fillStyle = stroke;
      ctx.fill();
    }
  }

  function render() {
    if (!image.complete || !image.naturalWidth) return;
    if (canvas.width !== image.naturalWidth || canvas.height !== image.naturalHeight) {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);
    const lineWidth = Math.max(3, canvas.width * 0.002);
    for (const shape of shapes) {
      paintPolygon(shape.points, shape.color || "#e45756", shape.color || "#e45756", lineWidth);
      const first = shape.points && shape.points[0];
      if (first && shape.label) {
        ctx.font = "bold " + Math.max(14, canvas.width * 0.018) + "px sans-serif";
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = "#111111";
        ctx.fillStyle = shape.color || "#e45756";
        ctx.strokeText(shape.label, first.x + 4, Math.max(20, first.y - 8));
        ctx.fillText(shape.label, first.x + 4, Math.max(20, first.y - 8));
      }
    }
    if (tool === "polygon" && draft.length) {
      const preview = draft.slice();
      if (hoverPoint) preview.push(hoverPoint);
      paintPolygon(preview, color, null, lineWidth);
    }
    if (tool === "rectangle" && dragStart && dragEnd) {
      const rectPoints = [
        dragStart,
        { x: dragEnd.x, y: dragStart.y },
        dragEnd,
        { x: dragStart.x, y: dragEnd.y }
      ];
      paintPolygon(rectPoints, color, color, lineWidth);
    }
    if (help) {
      help.textContent = tool === "polygon"
        ? "Polygon: nhấp để đặt điểm, bấm Hoàn tất polygon khi vẽ xong."
        : "Rectangle: kéo chuột để vẽ khung.";
    }
  }

  function commit(nextShapes) {
    shapes = nextShapes;
    setStateValue("shapes", shapes);
    render();
  }

  function finishPolygon() {
    if (tool !== "polygon" || draft.length < 3) return;
    commit(shapes.concat([{
      type: "polygon",
      label: data.label || "object",
      color: color,
      points: draft.map(function (point) { return { x: point.x, y: point.y }; })
    }]));
    draft = [];
    hoverPoint = null;
    render();
  }

  image.onload = render;
  image.src = data.image_src || "";
  if (image.complete) render();

  canvas.onpointerdown = function (event) {
    event.preventDefault();
    const point = toPoint(event);
    if (tool === "polygon") {
      draft.push(point);
      hoverPoint = point;
      render();
    } else {
      dragStart = point;
      dragEnd = point;
      canvas.setPointerCapture(event.pointerId);
      render();
    }
  };
  canvas.onpointermove = function (event) {
    const point = toPoint(event);
    hoverPoint = point;
    if (dragStart) dragEnd = point;
    render();
  };
  canvas.onpointerup = function (event) {
    if (tool !== "rectangle" || !dragStart) return;
    dragEnd = toPoint(event);
    const left = Math.min(dragStart.x, dragEnd.x);
    const right = Math.max(dragStart.x, dragEnd.x);
    const top = Math.min(dragStart.y, dragEnd.y);
    const bottom = Math.max(dragStart.y, dragEnd.y);
    if (right - left >= 2 && bottom - top >= 2) {
      commit(shapes.concat([{
        type: "rectangle",
        label: data.label || "object",
        color: color,
        points: [
          { x: left, y: top },
          { x: right, y: top },
          { x: right, y: bottom },
          { x: left, y: bottom }
        ]
      }]));
    }
    dragStart = null;
    dragEnd = null;
    render();
  };
  canvas.ondblclick = function (event) {
    event.preventDefault();
    finishPolygon();
  };
  finishButton.onclick = finishPolygon;
  undoButton.onclick = function () {
    if (draft.length) {
      draft.pop();
      render();
    } else if (shapes.length) {
      commit(shapes.slice(0, -1));
    }
  };
  clearButton.onclick = function () {
    draft = [];
    dragStart = null;
    dragEnd = null;
    hoverPoint = null;
    commit([]);
  };

  return function () {
    canvas.onpointerdown = null;
    canvas.onpointermove = null;
    canvas.onpointerup = null;
    canvas.ondblclick = null;
    finishButton.onclick = null;
    undoButton.onclick = null;
    clearButton.onclick = null;
  };
}
"""

if hasattr(st.components, "v2"):
    _CANVAS_COMPONENT = st.components.v2.component(
        "local_annotation_canvas",
        html="""
        <div class="canvas-shell">
          <canvas id="annotation-canvas" aria-label="Ảnh để vẽ annotation"></canvas>
          <div class="canvas-toolbar">
            <span id="canvas-help"></span>
            <button id="finish-polygon" type="button">Hoàn tất polygon</button>
            <button id="undo-shape" type="button">Undo</button>
            <button id="clear-shapes" type="button">Xóa hết</button>
          </div>
        </div>
        """,
        css="""
        .canvas-shell { width: 100%; }
        #annotation-canvas {
          display: block;
          width: 100%;
          height: auto;
          max-width: 100%;
          background: #15171a;
          touch-action: none;
          cursor: crosshair;
        }
        .canvas-toolbar {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 0.5rem;
          padding-top: 0.65rem;
        }
        #canvas-help { flex: 1 1 100%; color: var(--st-text-color); }
        .canvas-toolbar button {
          border: 1px solid var(--st-border-color);
          border-radius: var(--st-button-radius, 0.5rem);
          padding: 0.35rem 0.7rem;
          color: var(--st-text-color);
          background: var(--st-secondary-background-color);
          cursor: pointer;
        }
        .canvas-toolbar button:hover { border-color: var(--st-primary-color); }
        """,
        js=_CANVAS_JS,
    )
else:
    _CANVAS_COMPONENT = None


_GUIDELINE_CANVAS_JS = r"""
export default function (component) {
  const { data, parentElement, setStateValue } = component;
  const canvas = parentElement.querySelector("#guideline-canvas");
  const finish = parentElement.querySelector("#guideline-finish");
  const undo = parentElement.querySelector("#guideline-undo");
  const redo = parentElement.querySelector("#guideline-redo");
  const remove = parentElement.querySelector("#guideline-remove");
  const help = parentElement.querySelector("#guideline-help");
  if (!canvas || !finish || !undo || !redo || !remove) return;
  const ctx = canvas.getContext("2d");
  const image = new Image();
  let shapes = Array.isArray(data.shapes) ? JSON.parse(JSON.stringify(data.shapes)) : [];
  let draft = [], dragStart = null, dragEnd = null, hover = null;
  let history = Array.isArray(data.history) ? data.history.slice() : [];
  let future = Array.isArray(data.future) ? data.future.slice() : [];
  let dragTarget = null, dragBefore = null;
  const tool = data.tool || "polygon";
  const color = data.color || "#e45756";
  const visible = new Set(data.visible_labels || []);
  const spec = data.skeleton_spec || null;
  const pointGuidance = Array.isArray(data.point_guidance) ? data.point_guidance : [];
  if (finish) finish.style.display = data.face_mode ? "none" : "inline-block";
  function toPoint(event) {
    const r = canvas.getBoundingClientRect();
    return {x: Math.max(0, Math.min(canvas.width - 1, (event.clientX-r.left)*canvas.width/r.width)),
            y: Math.max(0, Math.min(canvas.height - 1, (event.clientY-r.top)*canvas.height/r.height))};
  }
  function line(points, stroke, close, fill) {
    if (!points || !points.length) return;
    ctx.beginPath(); ctx.moveTo(points[0].x, points[0].y);
    for (let i=1;i<points.length;i++) ctx.lineTo(points[i].x,points[i].y);
    if (close && points.length>2) ctx.closePath();
    if (fill) { ctx.globalAlpha=.18; ctx.fillStyle=fill; ctx.fill(); ctx.globalAlpha=1; }
    ctx.strokeStyle=stroke; ctx.lineWidth=Math.max(2,canvas.width*.002); ctx.lineJoin="round"; ctx.stroke();
  }
  function cross(a,b,c,d) {
    const orient=(p,q,r)=>(q.x-p.x)*(r.y-p.y)-(q.y-p.y)*(r.x-p.x);
    const a1=orient(a,b,c),a2=orient(a,b,d),b1=orient(c,d,a),b2=orient(c,d,b);
    return ((a1>0&&a2<0)||(a1<0&&a2>0))&&((b1>0&&b2<0)||(b1<0&&b2>0));
  }
  function polygonCrosses(points) {
    for(let i=0;i<points.length;i++) for(let j=i+1;j<points.length;j++) {
      if(j===i+1||(i===0&&j===points.length-1)) continue;
      if(cross(points[i],points[(i+1)%points.length],points[j],points[(j+1)%points.length])) return true;
    }
    return false;
  }
  function render() {
    if (!image.complete || !image.naturalWidth) return;
    if (canvas.width!==image.naturalWidth || canvas.height!==image.naturalHeight) {canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;}
    ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(image,0,0);
    for (const s of shapes) {
      if (!visible.has(s.label)) continue;
      const pts=s.points||[], c=s.color||"#e45756";
      if (s.type==="skeleton") {
        line(pts,c,s.contour==="closed",null);
        for (const p of pts) {
          ctx.beginPath(); ctx.arc(p.x,p.y,Math.max(4,canvas.width*.004),0,Math.PI*2);
          ctx.fillStyle=p.state==="Outside"?"#888":p.state==="Occluded"?"#ffb000":c; ctx.fill();
          ctx.font="bold "+Math.max(12,canvas.width*.014)+"px sans-serif"; ctx.lineWidth=3;ctx.strokeStyle="#111";ctx.fillStyle="#fff";
          ctx.strokeText(String(p.id),p.x+5,p.y-5);ctx.fillText(String(p.id),p.x+5,p.y-5);
        }
      } else {
        line(pts,c,s.type==="polygon",s.type==="polygon"?c:null);
        if (s.type==="rectangle" && pts.length>=3) line(pts,c,true,c);
        if (pts[0]) {ctx.font="bold "+Math.max(13,canvas.width*.016)+"px sans-serif";ctx.lineWidth=3;ctx.strokeStyle="#111";ctx.fillStyle=c;ctx.strokeText(s.label,pts[0].x+4,Math.max(18,pts[0].y-7));ctx.fillText(s.label,pts[0].x+4,Math.max(18,pts[0].y-7));}
      }
    }
    if ((tool==="polygon"||tool==="polyline") && draft.length) { const p=draft.slice();if(hover)p.push(hover);line(p,color,false,null); }
    if (tool==="rectangle" && dragStart && dragEnd) line([dragStart,{x:dragEnd.x,y:dragStart.y},dragEnd,{x:dragStart.x,y:dragEnd.y}],color,true,color);
    if (remove) remove.textContent = tool==="keypoint" ? "Xóa skeleton đang chọn" : "Xóa annotation cuối";
    if (help) {
      if (tool==="keypoint" && spec) {
        const activeShape=shapes.find(s=>s.type==="skeleton"&&s.label===data.label);
        const placed=(activeShape?.points||[]).length;
        const pointId=spec.start+placed;
        help.textContent=placed<spec.count
          ? `Điểm tiếp theo ${pointId}: ${pointGuidance[placed]||"đặt theo guideline"}. Nhấp để đặt; Shift+kéo điểm đã đặt để chỉnh, kể cả khi cần đặt các điểm trùng nhau.`
          : `${data.label}: đã đặt đủ ${spec.count} điểm. Kéo điểm trên canvas hoặc sửa trạng thái/tọa độ trong bảng.`;
      } else {
        help.textContent = tool==="rectangle" ? "Kéo để tạo box riêng cho object." : tool==="polyline" ? "Nhấp để đặt điểm; kết thúc đường mở bằng Hoàn tất." : "Nhấp để đặt vertex; hoàn tất polygon kín khi xong.";
      }
    }
  }
  function commit(next) { history.push(JSON.stringify(shapes)); if(history.length>60)history.shift();future=[];shapes=next;setStateValue("shapes",shapes);setStateValue("history",history);setStateValue("future",future);render(); }
  function finishGeometry() {
    if ((tool!=="polygon"&&tool!=="polyline") || draft.length<(tool==="polygon"?3:2)) return;
    if (tool==="polygon"&&polygonCrosses(draft)) { if(help)help.textContent="Polygon tự giao cắt; hãy chỉnh các vertex trước khi hoàn tất.";return; }
    commit(shapes.concat([{type:tool,label:data.label,color:color,points:draft.map(p=>({x:p.x,y:p.y})),...(tool==="polygon"?{}:{closed:false})}]));
    draft=[];hover=null;render();
  }
  image.onload=render;image.src=data.image_src||"";if(image.complete)render();
  canvas.onpointerdown=function(e){e.preventDefault();const p=toPoint(e);
    if(tool==="polygon"||tool==="polyline"){draft.push(p);hover=p;render();return;}
    if(tool==="keypoint"){
      const active=shapes.find(s=>s.type==="skeleton"&&s.label===data.label);
      const alreadyPlaced=(active?.points||[]).length;
      const canPlace=!!spec&&alreadyPlaced<spec.count;
      if(e.shiftKey||!canPlace){
        const radius=24*canvas.width/canvas.getBoundingClientRect().width;
        let nearest=null, distance=radius;
        for(const shape of shapes) if(shape.type==="skeleton"&&shape.label===data.label&&visible.has(shape.label)) for(const point of (shape.points||[])){
          const d=Math.hypot(point.x-p.x,point.y-p.y);if(d<distance){distance=d;nearest={label:shape.label,id:point.id};}
        }
        if(nearest){dragBefore=JSON.stringify(shapes);dragTarget=nearest;canvas.setPointerCapture(e.pointerId);return;}
      }
    }
    if(tool==="keypoint"&&spec){let next=shapes.slice();let idx=next.findIndex(s=>s.type==="skeleton"&&s.label===data.label);
      if(idx<0){next.push({...spec,type:"skeleton",label:data.label,color:color,points:[]});idx=next.length-1;}
      const s=next[idx], n=(s.points||[]).length;
      if(n<spec.count){s.points.push({id:spec.start+n,x:p.x,y:p.y,state:data.point_state||"Visible",source:data.point_source||"manual",review_status:"pending"});commit(next);} return;}
    dragStart=p;dragEnd=p;canvas.setPointerCapture(e.pointerId);render();
  };
  canvas.onpointermove=function(e){hover=toPoint(e);if(dragTarget){const pt=shapes.find(s=>s.label===dragTarget.label)?.points?.find(p=>p.id===dragTarget.id);if(pt){pt.x=hover.x;pt.y=hover.y;}}if(dragStart)dragEnd=hover;render();};
  canvas.onpointerup=function(e){if(dragTarget){if(dragBefore){history.push(dragBefore);if(history.length>60)history.shift();future=[];setStateValue("shapes",shapes);setStateValue("history",history);setStateValue("future",future);}dragTarget=null;dragBefore=null;render();return;}if(tool!=="rectangle"||!dragStart)return;dragEnd=toPoint(e);
    const l=Math.min(dragStart.x,dragEnd.x),r=Math.max(dragStart.x,dragEnd.x),t=Math.min(dragStart.y,dragEnd.y),b=Math.max(dragStart.y,dragEnd.y);
    if(r-l>=2&&b-t>=2)commit(shapes.concat([{type:"rectangle",label:data.label,color:color,occluded:!!data.occluded,truncated:!!data.truncated,points:[{x:l,y:t},{x:r,y:t},{x:r,y:b},{x:l,y:b}]}]));
    dragStart=null;dragEnd=null;render();};
  canvas.ondblclick=function(e){e.preventDefault();finishGeometry();};
  finish.onclick=finishGeometry;
  undo.onclick=function(){if(draft.length){draft.pop();render();return;}if(history.length){future.push(JSON.stringify(shapes));shapes=JSON.parse(history.pop());setStateValue("shapes",shapes);setStateValue("history",history);setStateValue("future",future);render();}};
  redo.onclick=function(){if(future.length){history.push(JSON.stringify(shapes));shapes=JSON.parse(future.pop());setStateValue("shapes",shapes);setStateValue("history",history);setStateValue("future",future);render();}};
  remove.onclick=function(){
    if(data.face_mode) {
      if(tool==="keypoint") commit(shapes.filter(s=>!(s.type==="skeleton"&&s.label===data.label)));
    }
    else if(shapes.length) commit(shapes.slice(0,-1));
  };
  return function(){canvas.onpointerdown=null;canvas.onpointermove=null;canvas.onpointerup=null;canvas.ondblclick=null;finish.onclick=null;undo.onclick=null;redo.onclick=null;remove.onclick=null;};
}
"""

if hasattr(st.components, "v2"):
    _GUIDELINE_CANVAS_COMPONENT = st.components.v2.component(
        "guideline_annotation_canvas",
        html="""
        <div class="canvas-shell">
          <canvas id="guideline-canvas" aria-label="Guideline annotation canvas"></canvas>
          <div class="canvas-toolbar">
            <span id="guideline-help"></span>
            <button id="guideline-finish" type="button">Hoàn tất polygon / polyline</button>
            <button id="guideline-undo" type="button">Undo</button>
            <button id="guideline-redo" type="button">Redo</button>
            <button id="guideline-remove" type="button">Xóa annotation cuối</button>
          </div>
        </div>
        """,
        css="""
        .canvas-shell { width:100%; }
        #guideline-canvas { display:block;width:100%;height:auto;max-width:100%;background:#15171a;touch-action:none;cursor:crosshair; }
        .canvas-toolbar { display:flex;flex-wrap:wrap;align-items:center;gap:.5rem;padding-top:.65rem; }
        #guideline-help { flex:1 1 100%;color:var(--st-text-color); }
        .canvas-toolbar button { border:1px solid var(--st-border-color);border-radius:var(--st-button-radius,.5rem);padding:.35rem .7rem;color:var(--st-text-color);background:var(--st-secondary-background-color);cursor:pointer; }
        """,
        js=_GUIDELINE_CANVAS_JS,
    )
else:
    _GUIDELINE_CANVAS_COMPONENT = None


def _canvas_label_color(label: str, labels: List[str]) -> str:
    palette = ["#e45756", "#4c78a8", "#54a24b", "#f58518", "#b279a2", "#72b7b2", "#ff9da6"]
    try:
        index = labels.index(label)
    except ValueError:
        index = 0
    return palette[index % len(palette)]


def _build_canvas_coco_zip(
    image_records: List[Dict[str, Any]],
    annotations_by_image: Dict[str, List[Dict[str, Any]]],
    labels: List[str],
) -> bytes:
    """Create a COCO instance-segmentation archive from local canvas polygons."""
    category_ids = {name: index + 1 for index, name in enumerate(labels)}
    coco_images: List[Dict[str, Any]] = []
    coco_annotations: List[Dict[str, Any]] = []
    archive_images: List[Tuple[str, bytes]] = []
    annotation_id = 1

    for image_id, record in enumerate(image_records, start=1):
        archive_name = f"images/{image_id:06d}_{Path(record['name']).stem}.png"
        coco_images.append({
            "id": image_id,
            "file_name": archive_name,
            "width": int(record["width"]),
            "height": int(record["height"]),
        })
        archive_images.append((archive_name, record["data"]))
        for shape in annotations_by_image.get(record["key"], []):
            points = shape.get("points") or []
            if len(points) < 3 or shape.get("label") not in category_ids:
                continue
            coords = [
                (
                    max(0.0, min(float(record["width"] - 1), float(point["x"]))),
                    max(0.0, min(float(record["height"] - 1), float(point["y"]))),
                )
                for point in points
            ]
            flat = [round(value, 2) for point in coords for value in point]
            xs = [point[0] for point in coords]
            ys = [point[1] for point in coords]
            polygon_area = abs(sum(
                coords[i][0] * coords[(i + 1) % len(coords)][1]
                - coords[(i + 1) % len(coords)][0] * coords[i][1]
                for i in range(len(coords))
            )) / 2.0
            if polygon_area <= 0:
                continue
            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_ids[shape["label"]],
                "segmentation": [flat],
                "area": round(polygon_area, 2),
                "bbox": [
                    round(min(xs), 2),
                    round(min(ys), 2),
                    round(max(xs) - min(xs), 2),
                    round(max(ys) - min(ys), 2),
                ],
                "iscrowd": 0,
            })
            annotation_id += 1

    coco = {
        "info": {"description": "Local canvas annotations exported by CVAT Auto Labeler"},
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": category_ids[name], "name": name, "supercategory": "object"}
            for name in labels
        ],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("annotations/instances_default.json", json.dumps(coco, ensure_ascii=False, indent=2))
        for archive_name, image_bytes in archive_images:
            archive.writestr(archive_name, image_bytes)
    return buffer.getvalue()


def render_guideline_canvas_workspace(mode: str) -> None:
    """Street Scene and Facial Landmark-50 workspaces with strict taxonomies."""
    if _GUIDELINE_CANVAS_COMPONENT is None:
        st.error("Guideline canvas cần Streamlit 1.57 trở lên.")
        return
    face_mode = mode == "Facial Landmark-50"
    st.subheader(mode)
    st.caption("Tọa độ được lưu theo kích thước ảnh gốc. Annotation guideline được lưu riêng theo ảnh trong outputs/local_annotations.")
    if face_mode:
        st.info("Chỉ annotate người lái; trái/phải theo phía trên ảnh. Mắt nhắm vẫn đủ 8 điểm, miệng đóng vẫn giữ moitrong, mặt nghiêng chỉnh từng điểm. Pre-label phải kiểm tra đủ 50 trạng thái; ảnh không có mặt thì không tạo landmark.")
    else:
        st.info("Mỗi object có box riêng và box được overlap. area/alternative chỉ gồm hướng đi cùng chiều. Lane marking luôn là đường Polyline mở; không nối qua đoạn thiếu bằng chứng.")
    uploads = st.file_uploader(
        "Chọn ảnh cần gán nhãn", type=["png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=True, key=f"guideline_uploads_{mode}",
    )
    if not uploads:
        st.info("Tải ảnh lên để bắt đầu. Các annotation đã lưu trong outputs/local_annotations sẽ được khôi phục khi chọn lại cùng ảnh.")
        return
    records = []
    for upload in uploads:
        raw = upload.getvalue()
        key = hashlib.sha256(upload.name.encode("utf-8") + raw).hexdigest()[:24]
        try:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            buf = io.BytesIO(); image.save(buf, format="PNG", optimize=True)
            records.append({"key": key, "name": upload.name, "data": buf.getvalue(), "width": image.width, "height": image.height})
        except Exception:
            st.warning(f"Không đọc được file ảnh: {upload.name}")
    if not records:
        return
    record_map = {r["key"]: r for r in records}
    active_key = st.selectbox("Ảnh đang gán nhãn", list(record_map), format_func=lambda k: f"{record_map[k]['name']} ({record_map[k]['width']}×{record_map[k]['height']})", key=f"guideline_active_{mode}")
    record = record_map[active_key]
    image_state_key = f"guideline_{'face' if face_mode else 'street'}_{active_key}"
    storage = ROOT / "outputs" / "local_annotations" / ("facial" if face_mode else "street") / f"{active_key}.json"
    if image_state_key not in st.session_state:
        loaded = []
        if storage.is_file():
            try:
                payload = json.loads(storage.read_text(encoding="utf-8"))
                if payload.get("image", {}).get("key") == active_key and payload.get("mode") == mode:
                    loaded = payload.get("annotations", [])
                    st.session_state[f"{image_state_key}_issues"] = payload.get("issues", [])
            except (OSError, json.JSONDecodeError):
                st.warning("Không đọc được annotation lưu trước đó; tệp vẫn được giữ nguyên.")
        st.session_state[image_state_key] = loaded
    annotations = st.session_state[image_state_key]
    issue_key = f"{image_state_key}_issues"
    issues = st.session_state.setdefault(issue_key, [])

    if face_mode:
        labels = [spec["name"] for spec in SKELETONS]
        active_label = st.selectbox("Skeleton", labels, format_func=lambda name: f"{name} · ID {SKELETON_BY_NAME[name]['start']}–{SKELETON_BY_NAME[name]['start'] + SKELETON_BY_NAME[name]['count'] - 1}", key=f"{image_state_key}_skeleton")
        tool = "keypoint"
        spec = SKELETON_BY_NAME[active_label]
        point_state = st.selectbox("Trạng thái điểm mới", POINT_STATES, key=f"{image_state_key}_new_state")
        no_face = st.checkbox("Ảnh này không có mặt người lái", key=f"{image_state_key}_no_face")
        if no_face and annotations:
            st.warning("Ảnh đã có landmark; bỏ chọn tùy chọn này trước khi tiếp tục chỉnh sửa.")
        hidden_labels = st.multiselect("Ẩn skeleton trên canvas", labels, key=f"{image_state_key}_hidden")
        if active_label in hidden_labels:
            st.caption("Skeleton đang chọn luôn hiển thị để đặt và chỉnh đúng điểm; bỏ chọn nó khỏi danh sách ẩn nếu muốn giữ trạng thái hiển thị sau khi đổi skeleton.")
        visible_labels = [name for name in labels if name not in hidden_labels or name == active_label]
        selected_idx = next((i for i, shape in enumerate(annotations) if shape.get("label") == active_label), None)
    else:
        labels = [label for group in STREET_LABELS.values() for label in group]
        active_label = st.selectbox("Nhãn", labels, key=f"{image_state_key}_label")
        tool = next(shape for shape, group in STREET_LABELS.items() if active_label in group)
        spec = None
        point_state = "Visible"
        hidden_labels = st.multiselect("Ẩn nhãn trên canvas", labels, key=f"{image_state_key}_hidden")
        visible_labels = [name for name in labels if name not in hidden_labels]
        selected_idx = st.selectbox("Annotation đang chọn để sửa/xóa", [None, *range(len(annotations))], format_func=lambda i: "Chưa chọn" if i is None else f"{i + 1}. {annotations[i].get('label')} · {annotations[i].get('type')}", key=f"{image_state_key}_selected")
        if tool == "rectangle":
            attr_col1, attr_col2 = st.columns(2)
            with attr_col1:
                next_occluded = st.checkbox("occluded cho box mới", key=f"{image_state_key}_occluded")
            with attr_col2:
                next_truncated = st.checkbox("truncated cho box mới", key=f"{image_state_key}_truncated")
        else:
            next_occluded = next_truncated = False

    if selected_idx is not None and 0 <= selected_idx < len(annotations):
        selected = annotations[selected_idx]
        if not face_mode and selected.get("type") == "rectangle":
            ac1, ac2 = st.columns(2)
            with ac1:
                selected["occluded"] = st.checkbox("occluded", value=bool(selected.get("occluded")), key=f"{image_state_key}_edit_occ_{selected_idx}")
            with ac2:
                selected["truncated"] = st.checkbox("truncated", value=bool(selected.get("truncated")), key=f"{image_state_key}_edit_trunc_{selected_idx}")
        points = selected.get("points", [])
        if points:
            editor_rows = [{"id": int(p.get("id", j)), "x": float(p["x"]), "y": float(p["y"]), **({"state": p.get("state", "Visible")} if face_mode else {})} for j, p in enumerate(points)]
            column_config = {"id": st.column_config.NumberColumn("Point ID", disabled=True), "x": st.column_config.NumberColumn("X", min_value=0.0, max_value=float(record["width"] - 1)), "y": st.column_config.NumberColumn("Y", min_value=0.0, max_value=float(record["height"] - 1))}
            if face_mode:
                column_config["state"] = st.column_config.SelectboxColumn("State", options=list(POINT_STATES), required=True)
            edited = st.data_editor(editor_rows, key=f"{image_state_key}_points_{selected_idx}", num_rows="fixed", hide_index=True, column_config=column_config)
            edited_rows = edited.to_dict("records") if hasattr(edited, "to_dict") else edited
            for p, row in zip(points, edited_rows):
                p["x"], p["y"] = float(row["x"]), float(row["y"])
                if face_mode:
                    p["state"] = row["state"]
        if st.button("Xóa annotation đang chọn", key=f"{image_state_key}_delete_{selected_idx}"):
            annotations.pop(selected_idx)
            st.rerun()

    if face_mode:
        counts = {s["name"]: len(next((a.get("points", []) for a in annotations if a.get("label") == s["name"]), [])) for s in SKELETONS}
        visible_count = sum(p.get("state") == "Visible" for a in annotations for p in a.get("points", []))
        occluded_count = sum(p.get("state") == "Occluded" for a in annotations for p in a.get("points", []))
        outside_count = sum(p.get("state") == "Outside" for a in annotations for p in a.get("points", []))
        complete = sum(counts.values())
        st.caption(f"Tiến độ {complete}/50 điểm · Visible {visible_count} · Occluded {occluded_count} · Outside {outside_count} · thiếu {50-complete}")
        selected_visible = sum(point.get("state") == "Visible" for shape in annotations if shape.get("label") == active_label for point in shape.get("points", []))
        minimum = int(spec["minimum_visible"])
        st.caption(f"{active_label}: đang thấy {selected_visible}/{spec['count']} điểm; ngưỡng guideline {minimum}/{spec['count']}. Dưới ngưỡng, chỉ hoàn tất điểm nếu còn bằng chứng; nếu không, đặt cả skeleton Outside.")
        active_shape = next((shape for shape in annotations if shape.get("label") == active_label), None)
        placed_count = len(active_shape.get("points", [])) if active_shape else 0
        if placed_count < spec["count"]:
            next_id = spec["start"] + placed_count
            st.info(f"Điểm tiếp theo · ID {next_id}: {SKELETON_POINT_GUIDANCE[active_label][placed_count]}. Đặt theo đúng thứ tự Point ID; trái/phải tính theo phía trên ảnh.")
    else:
        st.caption(f"{len(annotations)} annotation · {len(records)} ảnh trong batch")

    canvas_key = f"guideline_canvas_state_{mode}_{active_key}"
    saved_component_state = st.session_state.get(canvas_key, {})
    image_src = "data:image/png;base64," + base64.b64encode(record["data"]).decode("ascii")
    guideline_shapes = []
    for i, shape in enumerate(annotations):
        copy = dict(shape)
        copy["color"] = _canvas_label_color(str(shape.get("label", "")), labels)
        guideline_shapes.append(copy)
    if isinstance(saved_component_state, dict):
        # Python-side edits (coordinates, states, or attributes) are the source
        # of truth when the component is mounted again.
        st.session_state[canvas_key] = {**saved_component_state, "shapes": guideline_shapes}
    result = _GUIDELINE_CANVAS_COMPONENT(
        key=canvas_key,
        data={"image_src": image_src, "shapes": guideline_shapes, "tool": ("inspect" if face_mode and no_face else tool), "label": active_label,
              "color": _canvas_label_color(active_label, labels), "visible_labels": visible_labels,
              "face_mode": face_mode,
              "skeleton_spec": ({"start": spec["start"], "count": spec["count"], "contour": spec["contour"]} if spec else None),
              "point_guidance": SKELETON_POINT_GUIDANCE.get(active_label, []) if face_mode else [],
              "point_state": point_state,
              "occluded": bool(next_occluded) if not face_mode and tool == "rectangle" else False,
              "truncated": bool(next_truncated) if not face_mode and tool == "rectangle" else False,
              "history": saved_component_state.get("history", []) if isinstance(saved_component_state, dict) else [],
              "future": saved_component_state.get("future", []) if isinstance(saved_component_state, dict) else []},
        width="stretch", on_shapes_change=lambda: None,
        on_history_change=lambda: None, on_future_change=lambda: None,
    )
    updated = getattr(result, "shapes", None)
    if isinstance(updated, list):
        annotations[:] = updated
    if not face_mode and tool == "rectangle":
        for shape in annotations:
            if shape.get("type") == "rectangle" and shape.get("label") == active_label:
                shape.setdefault("occluded", bool(next_occluded)); shape.setdefault("truncated", bool(next_truncated))

    if face_mode and not no_face:
        validation = validate_facial(annotations, record["width"], record["height"])
    elif face_mode:
        validation = [] if not annotations else ["Remove landmarks or mark this image as containing a face."]
    else:
        validation = validate_street(annotations, record["width"], record["height"])
    if validation:
        with st.expander(f"Kiểm tra guideline · {len(validation)} mục cần xem", expanded=False):
            for message in validation:
                st.warning(message)
    else:
        st.success("Các annotation hiện có hợp lệ theo cấu trúc guideline.")

    with st.expander("Ghi nhận issue cho reviewer"):
        issue_type = st.selectbox("Issue type", ISSUE_TYPES, key=f"{image_state_key}_issue_type")
        issue_note = st.text_input("Ghi chú", key=f"{image_state_key}_issue_note")
        if st.button("Thêm issue", key=f"{image_state_key}_add_issue") and issue_note.strip():
            issues.append({"type": issue_type, "note": issue_note.strip()})
        for n, item in enumerate(issues):
            st.write(f"{item.get('type')}: {item.get('note')}")
            if st.button("Xóa", key=f"{image_state_key}_issue_delete_{n}"):
                issues.pop(n); st.rerun()

    save_annotation_file(storage, record, mode, annotations, issues)
    archive = {"schema_version": 1, "mode": mode, "image": {"key": active_key, "name": record["name"], "width": record["width"], "height": record["height"]}, "annotations": export_annotations(annotations), "issues": issues}
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Xuất annotation JSON", data=json.dumps(archive, ensure_ascii=False, indent=2).encode("utf-8"), file_name=f"{Path(record['name']).stem}_annotations.json", mime="application/json", key=f"{image_state_key}_download")
    with c2:
        imported = st.file_uploader("Nạp annotation JSON", type=["json"], key=f"{image_state_key}_import")
        if imported and st.button("Nạp tệp vào ảnh này", key=f"{image_state_key}_import_button"):
            try:
                imported_bytes = imported.getvalue()
                annotations[:] = load_annotation_file(imported_bytes, record, mode)
                imported_payload = json.loads(imported_bytes.decode("utf-8"))
                imported_issues = imported_payload.get("issues", [])
                if isinstance(imported_issues, list):
                    issues[:] = imported_issues
                st.rerun()
            except (ValueError, json.JSONDecodeError) as exc:
                st.error(f"Không thể nạp tệp: {exc}")


def render_cvat_facial_workspace(ctx: Dict[str, Any], client: CVATClient) -> None:
    """Manually annotate one driver face and append/update its group in CVAT."""
    st.header("Facial Landmark-50 · CVAT Job")
    st.caption("Ảnh được tải từ Job hiện tại. Tọa độ là pixel ảnh gốc; lưu chỉ cập nhật nhóm skeleton của app ở frame đang chọn.")
    try:
        taxonomy = resolve_facial_taxonomy(ctx.get("labels", []))
    except FacialTaxonomyError as exc:
        st.error(str(exc))
        st.info("Mỗi label phải là Skeleton, có đúng số point sublabel theo guideline, tên point phải mã hóa ID (ví dụ `point_14`) và SVG edges phải nối đúng thứ tự. CVAT không cho đổi skeleton config của Task đã tạo; nếu cấu hình sai, tạo Task mới với taxonomy đúng.")
        return
    topology_warnings = [
        f"{name}: {entry['topology_warning']}"
        for name, entry in taxonomy.items()
        if entry.get("topology_warning")
    ]
    if topology_warnings:
        st.warning(
            "CVAT không cung cấp topology SVG có thể xác minh qua API. App vẫn cho vẽ/lưu "
            "vì Point ID đã khớp tường minh với sublabel; canvas dùng contour theo guideline. "
            "CVAT có thể không nối các điểm thành đường trong giao diện nếu SVG của Task bị thiếu/sai.\n\n"
            + "\n".join(f"• {warning}" for warning in topology_warnings)
        )

    frame_options = {}
    for index, frame in enumerate(ctx.get("frames", [])):
        number = frame_number(frame, index)
        frame_options[f"{number}: {frame.get('name') or f'frame_{number}'}"] = number
    if not frame_options:
        st.warning("Job không có frame.")
        return
    selected_frame = st.selectbox("Frame CVAT", list(frame_options), key=f"face_frame_{ctx['task_id']}_{ctx['job_id']}")
    frame_number_value = frame_options[selected_frame]
    image_key = f"cvat_face_{ctx['task_id']}_{ctx['job_id']}_{frame_number_value}"
    try:
        image_array = load_image_from_cvat(client, int(ctx["task_id"]), frame_number_value, int(ctx["job_id"]))
    except Exception as exc:
        st.error(f"Không tải được ảnh từ Job: {exc}")
        return
    image = Image.fromarray(image_array)
    png = io.BytesIO()
    image.save(png, format="PNG", optimize=True)
    image_src = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode("ascii")
    record = {
        "key": image_key, "name": f"task_{ctx['task_id']}_frame_{frame_number_value}.png",
        "width": image.width, "height": image.height,
    }
    storage = ROOT / "outputs" / "local_annotations" / "cvat_facial" / str(ctx["job_id"]) / f"frame_{frame_number_value}.json"
    state_key = f"{image_key}_annotations"
    if state_key not in st.session_state:
        if storage.is_file():
            try:
                st.session_state[state_key] = load_annotation_file(storage.read_bytes(), record, "Facial Landmark-50")
            except (OSError, ValueError, json.JSONDecodeError):
                st.warning("Không khôi phục được bản nháp cục bộ; tệp gốc được giữ nguyên.")
                st.session_state[state_key] = []
        else:
            st.session_state[state_key] = []
    annotations = st.session_state[state_key]
    for shape in annotations:
        for point in shape.get("points", []):
            point.setdefault("source", "manual")
            point.setdefault("review_status", "pending")

    ai_meta_key = f"{state_key}_ai_metadata"
    st.info("Bấm chạy AI sẽ tạo lại đề xuất và thay bản nháp hiện tại của frame này nếu phát hiện được mặt.")
    if st.button("AI tự gán 50 điểm cho frame", type="primary", key=f"{image_key}_auto_label"):
        try:
            with st.spinner("MediaPipe đang phát hiện mặt và đề xuất 50 landmark…"):
                proposed, ai_meta = facial_mp_module.propose_facial_landmark50(
                    np.asarray(image.convert("RGB")),
                )
            if not proposed:
                st.warning("Không phát hiện được mặt trong frame này; không ghi annotation.")
            else:
                annotations[:] = proposed
                st.session_state[state_key] = annotations
                st.session_state[ai_meta_key] = ai_meta
                save_annotation_file(storage, record, "Facial Landmark-50", annotations)
                st.rerun()
        except Exception as exc:
            logger.exception(
                "Facial landmark auto-label failed task=%s job=%s frame=%s",
                ctx.get("task_id"), ctx.get("job_id"), frame_number_value,
            )
            st.error(f"AI auto-label thất bại: {type(exc).__name__}: {exc}")
    ai_meta = st.session_state.get(ai_meta_key)
    if ai_meta:
        st.caption(
            f"Nguồn: {ai_meta.get('model')} · {ai_meta.get('native_landmarks', 0)} mesh points · "
            f"face detected: {ai_meta.get('face_count', 0)} · mọi điểm ở trạng thái review pending."
        )
        if ai_meta.get("warning"):
            st.warning(ai_meta["warning"])
        st.info(
            "Đây là pre-label từ mesh 478 điểm theo contour/ID của guideline; MediaPipe không trả "
            "confidence hoặc trạng thái che khuất cho từng điểm. Kiểm tra trạng thái trước khi đồng bộ CVAT."
        )

    svg_verified_count = sum(bool(entry.get("svg_verified")) for entry in taxonomy.values())
    if svg_verified_count == len(SKELETONS):
        st.success("CVAT taxonomy đạt: 7 skeleton parent, point sublabel, Point ID và SVG topology đã được kiểm tra.")
    else:
        st.info(f"CVAT taxonomy đạt cho 7 nhóm và 50 Point ID; SVG topology xác minh được {svg_verified_count}/7 nhóm.")
    group_col, state_col = st.columns(2)
    with group_col:
        active_label = st.selectbox(
            "Skeleton đang vẽ", [spec["name"] for spec in SKELETONS],
            format_func=lambda name: f"{name} · ID {SKELETON_BY_NAME[name]['start']}–{SKELETON_BY_NAME[name]['start'] + SKELETON_BY_NAME[name]['count'] - 1}",
            key=f"{image_key}_skeleton",
        )
    with state_col:
        point_state = st.selectbox("Trạng thái điểm mới", POINT_STATES, key=f"{image_key}_point_state")
    spec = SKELETON_BY_NAME[active_label]
    hidden = st.multiselect("Ẩn skeleton khác", [item["name"] for item in SKELETONS if item["name"] != active_label], key=f"{image_key}_hidden")
    visible_labels = [item["name"] for item in SKELETONS if item["name"] not in hidden]
    selected_idx = next((index for index, shape in enumerate(annotations) if shape.get("label") == active_label), None)
    if selected_idx is not None:
        points = annotations[selected_idx].get("points", [])
        editor_rows = [{"id": int(point["id"]), "x": float(point["x"]), "y": float(point["y"]), "state": point.get("state", "Visible"), "source": point.get("source", "manual"), "review_status": point.get("review_status", "pending")} for point in points]
        config = {
            "id": st.column_config.NumberColumn("Point ID", disabled=True),
            "x": st.column_config.NumberColumn("X", min_value=0.0, max_value=float(image.width - 1)),
            "y": st.column_config.NumberColumn("Y", min_value=0.0, max_value=float(image.height - 1)),
            "state": st.column_config.SelectboxColumn("Visibility", options=list(POINT_STATES), required=True),
            "source": st.column_config.TextColumn("Source", disabled=True),
            "review_status": st.column_config.SelectboxColumn("Review", options=["pending", "reviewed", "accepted"], required=True),
        }
        edited = st.data_editor(editor_rows, key=f"{image_key}_points_{selected_idx}", num_rows="fixed", hide_index=True, column_config=config)
        rows = edited.to_dict("records") if hasattr(edited, "to_dict") else edited
        for point, row in zip(points, rows):
            point["x"], point["y"], point["state"], point["review_status"] = float(row["x"]), float(row["y"]), row["state"], row["review_status"]
    if selected_idx is not None and st.button("Xóa skeleton đang chọn", key=f"{image_key}_delete_{selected_idx}"):
        annotations.pop(selected_idx)
        st.rerun()

    complete = sum(len(shape.get("points", [])) for shape in annotations)
    states = {state: sum(point.get("state") == state for shape in annotations for point in shape.get("points", [])) for state in POINT_STATES}
    st.caption(f"Tiến độ {complete}/50 · Visible {states['Visible']} · Occluded {states['Occluded']} · Outside {states['Outside']}")
    active_shape = next((shape for shape in annotations if shape.get("label") == active_label), None)
    point_index = len(active_shape.get("points", [])) if active_shape else 0
    if point_index < spec["count"]:
        next_id = spec["start"] + point_index
        st.info(f"Điểm kế tiếp · ID {next_id}: {SKELETON_POINT_GUIDANCE[active_label][point_index]}")

    canvas_key = f"cvat_face_canvas_{ctx['job_id']}_{frame_number_value}"
    saved_component_state = st.session_state.get(canvas_key, {})
    canvas_shapes = []
    for shape in annotations:
        rendered = dict(shape)
        rendered["color"] = _canvas_label_color(str(shape.get("label", "")), [item["name"] for item in SKELETONS])
        canvas_shapes.append(rendered)
    if isinstance(saved_component_state, dict):
        st.session_state[canvas_key] = {**saved_component_state, "shapes": canvas_shapes}
    result = _GUIDELINE_CANVAS_COMPONENT(
        key=canvas_key,
        data={
            "image_src": image_src, "shapes": canvas_shapes, "tool": "keypoint",
            "label": active_label, "color": _canvas_label_color(active_label, [item["name"] for item in SKELETONS]),
            "visible_labels": visible_labels,
            "skeleton_spec": {"start": spec["start"], "count": spec["count"], "contour": spec["contour"]},
            "point_guidance": SKELETON_POINT_GUIDANCE[active_label], "point_state": point_state,
            "face_mode": True,
            "history": saved_component_state.get("history", []),
            "future": saved_component_state.get("future", []),
        },
        width="stretch", on_shapes_change=lambda: None,
        on_history_change=lambda: None, on_future_change=lambda: None,
    )
    updated = getattr(result, "shapes", None)
    if isinstance(updated, list):
        annotations[:] = updated
    draft_validation = validate_facial(annotations, image.width, image.height, allow_incomplete=True)
    final_validation = validate_facial(annotations, image.width, image.height)
    if draft_validation:
        with st.expander(f"Kiểm tra guideline · {len(draft_validation)} mục cần xem", expanded=False):
            for message in draft_validation:
                st.warning(message)
    elif not final_validation:
        st.success("50 điểm hợp lệ theo guideline. Có thể lưu draft hoặc đồng bộ nhóm này lên CVAT.")
    else:
        st.info(f"Draft đang có {complete}/50 điểm. Validator sẽ kiểm tra đủ 7 skeleton khi hoàn tất.")

    if not final_validation and st.button("Accept All · duyệt 50 điểm", key=f"{image_key}_accept_all"):
        for shape in annotations:
            for point in shape.get("points", []):
                point["review_status"] = "accepted"
        st.rerun()

    save_annotation_file(storage, record, "Facial Landmark-50", annotations)
    st.download_button(
        "Tải bản nháp JSON", data=storage.read_bytes(), file_name=f"frame_{frame_number_value}_facial_landmark50.json",
        mime="application/json", key=f"{image_key}_download",
    )
    all_accepted = not final_validation and all(point.get("review_status") == "accepted" for shape in annotations for point in shape.get("points", []))
    if not final_validation and not all_accepted:
        st.info("Duyệt trạng thái Review của từng điểm hoặc dùng **Accept All** trước khi lưu lên CVAT.")
    confirmed = st.checkbox(
        f"Xác nhận ghi/cập nhật đúng 7 skeleton của frame {frame_number_value} trong Job {ctx['job_id']}",
        key=f"{image_key}_confirm_save", disabled=not all_accepted,
    )
    if st.button("Save to CVAT", type="primary", disabled=bool(final_validation) or not all_accepted or not confirmed, key=f"{image_key}_save_cvat"):
        try:
            saved_group = next((int(shape["cvat_group"]) for shape in annotations if shape.get("cvat_group") is not None), None)
            current = client.get_job_annotations(int(ctx["job_id"]))
            group = saved_group if saved_group is not None else next_cvat_group(current)
            cvat_shapes = facial_annotations_to_cvat_shapes(annotations, taxonomy, frame=frame_number_value, group=group)
            result_info = save_facial_shapes_to_cvat(
                client, int(ctx["job_id"]), cvat_shapes, frame=frame_number_value,
                taxonomy=taxonomy, group=group,
            )
            for shape in annotations:
                shape["cvat_group"] = result_info["group"]
            save_annotation_file(storage, record, "Facial Landmark-50", annotations)
            st.success(f"Đã {('thêm' if result_info['added'] else 'cập nhật')} 7 skeleton và đọc lại xác nhận từ CVAT (group {result_info['group']}).")
            st.rerun()
        except Exception as exc:
            logger.exception("Facial Landmark CVAT save failed")
            st.error(f"Không lưu được lên CVAT: {exc}")


def render_canvas_workspace() -> None:
    """Standalone upload-and-annotate workflow that does not require CVAT task access."""
    st.title("Gán nhãn ảnh bằng canvas")
    guideline_mode = st.segmented_control(
        "Chế độ annotation",
        ["Custom canvas", "Street Scene", "Facial Landmark-50"],
        default="Custom canvas",
        key="canvas_guideline_mode",
    )
    if guideline_mode in ("Street Scene", "Facial Landmark-50"):
        render_guideline_canvas_workspace(guideline_mode)
        return
    st.caption("Tải ảnh trực tiếp, vẽ polygon hoặc rectangle, rồi xuất bộ dữ liệu COCO để nhập vào CVAT.")
    if _CANVAS_COMPONENT is None:
        st.error("Canvas cần Streamlit 1.57 trở lên. Hãy cập nhật Streamlit theo requirements.txt.")
        return

    label_text = st.text_input("Danh sách nhãn (ngăn cách bằng dấu phẩy)", value="object", key="canvas_label_text")
    labels = list(dict.fromkeys(part.strip() for part in label_text.split(",") if part.strip()))
    if not labels:
        st.warning("Nhập ít nhất một nhãn trước khi vẽ.")
        return

    uploads = st.file_uploader(
        "Chọn ảnh cần gán nhãn",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=True,
        key="canvas_image_uploads",
    )
    if not uploads:
        st.session_state.pop("canvas_image_cache", None)
        st.session_state.pop("canvas_annotations", None)
        st.session_state.pop("canvas_generations", None)
        for key in list(st.session_state):
            if str(key).startswith("canvas_annotation_"):
                st.session_state.pop(key, None)
        st.info("Tải một hoặc nhiều ảnh lên để bắt đầu.")
        return

    cache = st.session_state.setdefault("canvas_image_cache", {})
    image_records: List[Dict[str, Any]] = []
    for index, upload in enumerate(uploads):
        source = upload.getvalue()
        cache_key = hashlib.sha256(
            str(index).encode("utf-8") + upload.name.encode("utf-8") + source
        ).hexdigest()[:20]
        record = cache.get(cache_key)
        if record is None:
            try:
                image = ImageOps.exif_transpose(Image.open(io.BytesIO(source))).convert("RGB")
                normalized = io.BytesIO()
                image.save(normalized, format="PNG", optimize=True)
                record = {
                    "key": cache_key,
                    "name": upload.name,
                    "data": normalized.getvalue(),
                    "width": image.width,
                    "height": image.height,
                }
                cache[cache_key] = record
            except Exception:
                st.warning(f"Không đọc được file ảnh: {upload.name}")
                continue
        image_records.append(record)
    active_keys = {record["key"] for record in image_records}
    for old_key in list(cache):
        if old_key not in active_keys:
            del cache[old_key]
    if not image_records:
        st.error("Không có ảnh hợp lệ để hiển thị.")
        return

    annotations = st.session_state.setdefault("canvas_annotations", {})
    generations = st.session_state.setdefault("canvas_generations", {})
    for old_key in list(annotations):
        if old_key not in active_keys:
            del annotations[old_key]
    for old_key in list(generations):
        if old_key not in active_keys:
            del generations[old_key]

    record_by_key = {record["key"]: record for record in image_records}
    active_key = st.selectbox(
        "Ảnh đang gán nhãn",
        options=[record["key"] for record in image_records],
        format_func=lambda key: (
            f"{image_records.index(record_by_key[key]) + 1}. {record_by_key[key]['name']} "
            f"({record_by_key[key]['width']}×{record_by_key[key]['height']})"
        ),
        key="canvas_active_image",
    )
    active_record = record_by_key[active_key]
    shape_col, label_col = st.columns(2)
    with shape_col:
        tool = st.selectbox("Công cụ", ["Polygon", "Rectangle"], key="canvas_tool")
    with label_col:
        active_label = st.selectbox("Nhãn cho shape mới", labels, key="canvas_active_label")

    annotations.setdefault(active_key, [])
    generations.setdefault(active_key, 0)
    clear_col, count_col = st.columns([1, 3])
    with clear_col:
        if st.button("Xóa annotation ảnh này", key=f"canvas_reset_{active_key}"):
            annotations[active_key] = []
            generations[active_key] += 1
            st.rerun()
    with count_col:
        st.caption(f"{len(annotations[active_key])} shape · {len(image_records)} ảnh trong phiên")

    image_src = "data:image/png;base64," + base64.b64encode(active_record["data"]).decode("ascii")
    component_key = f"canvas_annotation_{active_key}_{generations[active_key]}"
    saved_component_state = st.session_state.get(component_key, {})
    if isinstance(saved_component_state, dict) and isinstance(saved_component_state.get("shapes"), list):
        annotations[active_key] = saved_component_state["shapes"]
    component_state = _CANVAS_COMPONENT(
        key=component_key,
        data={
            "image_src": image_src,
            "shapes": annotations[active_key],
            "tool": tool.casefold(),
            "label": active_label,
            "color": _canvas_label_color(active_label, labels),
        },
        width="stretch",
        on_shapes_change=lambda: None,
    )
    shapes = getattr(component_state, "shapes", None)
    if isinstance(shapes, list):
        annotations[active_key] = shapes

    st.caption("Canvas lưu tọa độ theo kích thước ảnh gốc; polygon kết thúc bằng nút Hoàn tất hoặc nhấp đúp.")
    export_zip = _build_canvas_coco_zip(image_records, annotations, labels)
    st.download_button(
        "Tải bộ dữ liệu COCO (.zip)",
        data=export_zip,
        file_name="canvas_annotations_coco.zip",
        mime="application/zip",
        type="primary",
        key="canvas_download_coco",
    )
    st.caption("Sau khi tải, nhập file ZIP vào một Task CVAT có các nhãn tương ứng.")


def _set_default_environment_url() -> None:
    environment = st.session_state.get("cvat_environment", "CVAT Local")
    if environment == "CVAT Local":
        st.session_state.cvat_url = CVATClient.DEFAULT_URL
    elif environment == "CVAT Online":
        st.session_state.cvat_url = CVATClient.ONLINE_URL


st.session_state.setdefault("app_mode", "CVAT")
app_mode = st.sidebar.selectbox("Chế độ làm việc", ["CVAT", "Canvas local"], key="app_mode")
if app_mode == "Canvas local":
    render_canvas_workspace()
    st.stop()


for key, value in {
    "cvat_environment": "CVAT Local",
    "cvat_url": os.getenv("CVAT_SERVER_URL", CVATClient.DEFAULT_URL),
    "cvat_task_job_url": "",
    "cvat_auth_choice": "Personal Access Token (khuyến nghị)",
    "cvat_user": os.getenv("CVAT_USERNAME", ""),
    "cvat_pass": os.getenv("CVAT_PASSWORD", ""),
    "cvat_access_token": os.getenv("CVAT_ACCESS_TOKEN", ""),
    "cvat_organization": os.getenv("CVAT_ORGANIZATION", "cohort-5a"),
}.items():
    st.session_state.setdefault(key, value)

_apply_task_job_url()
direct_target = parse_cvat_task_job_url(st.session_state.cvat_task_job_url)
with st.sidebar.expander("Kết nối CVAT", expanded=not st.session_state.connected):
    st.text_input(
        "Liên kết Task / Job",
        key="cvat_task_job_url",
        placeholder="https://cvat.edufun.ai/tasks/226/jobs/982",
        help="Dán URL dạng /tasks/<id>/jobs/<id> để mở đúng Task và Job.",
        on_change=_apply_task_job_url,
    )
    if st.session_state.cvat_task_job_url.strip() and direct_target is None:
        st.warning("URL chưa đúng định dạng. Dùng link CVAT dạng /tasks/<id>/jobs/<id>.")
    st.selectbox("Môi trường", ["CVAT Local", "CVAT Online", "Custom"], key="cvat_environment", on_change=_set_default_environment_url, disabled=direct_target is not None)
    st.text_input("CVAT Server URL", key="cvat_url", help="Local mặc định http://localhost:8080; Online mặc định https://cvat.edufun.ai.", disabled=direct_target is not None)
    st.text_input(
        "Organization slug",
        key="cvat_organization",
        help="Workspace organization của CVAT, ví dụ cohort-5a. Để trống nếu làm trong Personal workspace.",
    )
    auth_choice = st.segmented_control(
        "Phương thức đăng nhập",
        ["Username / password (session)", "Personal Access Token (khuyến nghị)"],
        key="cvat_auth_choice",
    )
    if auth_choice.startswith("Username"):
        st.text_input("Username", key="cvat_user")
        st.text_input("Password", type="password", key="cvat_pass")
        auth_mode = "session"
    else:
        st.text_input("CVAT Personal Access Token", type="password", key="cvat_access_token")
        auth_mode = "pat"

    connect_col, test_col = st.columns(2)
    connect_clicked = connect_col.button("Connect", type="primary", width="stretch")
    test_clicked = test_col.button("Test Connection", width="stretch")
    if connect_clicked or test_clicked:
        previous_client = st.session_state.cvat_client
        if previous_client:
            previous_client.logout()
        st.session_state.connected = False
        st.session_state.task_context = None
        st.session_state.predictions = None
        st.session_state.approved_mask_ids = set()
        st.session_state.synced_mask_ids = set()
        st.session_state.cvat_auth_mode = auth_mode
        st.session_state.cvat_client = CVATClient(
            base_url=st.session_state.cvat_url,
            username=st.session_state.cvat_user if auth_mode == "session" else "",
            password=st.session_state.cvat_pass if auth_mode == "session" else "",
            access_token=st.session_state.cvat_access_token if auth_mode == "pat" else "",
            auth_mode=auth_mode,
            organization_slug=st.session_state.cvat_organization,
        )
        try:
            st.session_state.cvat_client.login()
            server_info = st.session_state.cvat_client.test_connection()
            st.session_state.connected = True
            st.success(f"CVAT kết nối được ({server_info.get('version', 'connected')}).")
        except Exception as exc:
            st.session_state.connected = False
            logger.exception("CVAT connection test failed")
            st.error(str(exc))

active_client = st.session_state.cvat_client
if st.session_state.connected and active_client and not (connect_clicked or test_clicked):
    credentials_match = (
        active_client.base_url == st.session_state.cvat_url.rstrip("/")
        and active_client.auth_mode == auth_mode
        and getattr(active_client, "organization_slug", "") == st.session_state.cvat_organization.strip()
        and (auth_mode != "session" or (
            active_client.username == st.session_state.cvat_user
            and active_client.password == st.session_state.cvat_pass
        ))
        and (auth_mode != "pat" or active_client.access_token == st.session_state.cvat_access_token)
    )
    if not credentials_match:
        active_client.logout()
        st.session_state.cvat_client = None
        st.session_state.connected = False
        st.session_state.task_context = None
        st.session_state.predictions = None
        st.session_state.approved_mask_ids = set()
        st.session_state.synced_mask_ids = set()
        st.sidebar.info("URL hoặc thông tin đăng nhập đã đổi. Hãy kết nối lại CVAT.")

if st.session_state.connected:
    st.sidebar.success(f"Đã kết nối: {st.session_state.get('cvat_url', CVATClient.DEFAULT_URL)}")
    if st.sidebar.button("Đăng xuất", width="stretch"):
        if st.session_state.cvat_client:
            st.session_state.cvat_client.logout()
        st.session_state.cvat_client = None
        st.session_state.connected = False
        st.session_state.task_context = None
        st.session_state.predictions = None
        st.rerun()
else:
    st.sidebar.warning("Chưa kết nối CVAT")

st.title("🤖 AI Auto Labeler · CVAT")
st.caption("Gán nhãn → duyệt và sửa → đánh giá → đồng bộ. Mọi tính năng hiện có vẫn nằm trong từng bước.")
device_status = get_cached_device_status()
if device_status.get("cuda_available"):
    gpu_description = " · ".join(
        f"{name} ({device_status['gpu_memory_gib'][index]:.1f} GB)"
        if index < len(device_status.get("gpu_memory_gib", [])) else name
        for index, name in enumerate(device_status.get("gpu_names", []))
    ) or "CUDA"
    st.badge(f"GPU sẵn sàng · {gpu_description}", icon=":material/developer_board:", color="green")
elif device_status.get("driver_gpus"):
    driver_description = ", ".join(
        f"{gpu['name']} ({gpu['memory_gib']:.1f} GB)" if gpu.get("memory_gib") else gpu["name"]
        for gpu in device_status["driver_gpus"]
    )
    st.warning(
        f"Đã thấy {driver_description}, nhưng PyTorch của môi trường này là bản CPU-only "
        f"({device_status.get('torch_version') or 'chưa có PyTorch'}). Chạy `setup_gpu.ps1` rồi mở app bằng `.venv-sam3` để bật CUDA.",
        title="GPU chưa được PyTorch sử dụng",
    )
else:
    st.badge("Đang chạy CPU", icon=":material/desktop_windows:", color="gray")
if not st.session_state.connected:
    st.info("Mở sidebar để chọn Canvas local hoặc kết nối CVAT Local/Online.")
    st.stop()

client = get_client()
if direct_target is not None:
    _, task_id, job_id = direct_target
    st.sidebar.info(f"Đích từ URL: Task {task_id} · Job {job_id}")
else:
    try:
        projects = client.list_projects()
        project_options = {"Tất cả Projects": None}
        project_options.update({f"{project['id']}: {project['name']}": int(project["id"]) for project in projects})
        if not projects:
            st.sidebar.warning("Đăng nhập thành công nhưng tài khoản này không có Project nào được CVAT trả về.")
        project_label = st.sidebar.selectbox("Project", list(project_options), key="selected_project")
        tasks = client.list_tasks(project_id=project_options[project_label])
    except Exception as exc:
        st.error(f"Không đọc được Projects/Tasks từ CVAT: {exc}")
        if "401" in str(exc):
            st.session_state.connected = False
        st.stop()
    if not tasks:
        st.sidebar.warning(
            "Không có Task nào tài khoản này được phép xem. Kiểm tra quyền trong Organization/Project "
            "hoặc nhờ quản trị viên phân công Task cho tài khoản này."
        )
        st.stop()

    task_options = {f"{task['id']}: {task['name']}": int(task["id"]) for task in tasks}
    task_label = st.sidebar.selectbox("Task", list(task_options), key="selected_task")
    task_id = task_options[task_label]
    try:
        jobs = client.list_jobs(task_id)
    except Exception as exc:
        st.sidebar.error(f"Không đọc được Job của Task {task_id}: {exc}")
        jobs = []
    job_options = {f"{job['id']} · frames {job.get('start_frame', '?')}–{job.get('stop_frame', '?')} · {job.get('state', 'unknown')}": int(job["id"]) for job in jobs}
    job_label = st.sidebar.selectbox("Job", list(job_options) or ["Task chưa có Job"], key="selected_job")
    job_id = job_options.get(job_label)
if st.sidebar.button("Tải Task / Job", type="primary", disabled=job_id is None, width="stretch"):
    try:
        ctx_loaded = client.load_task_context(task_id, job_id)
        st.session_state.task_context = ctx_loaded
        st.session_state.loaded_context_url = client.base_url
        st.session_state.predictions = None
        st.session_state.qc_results = None
        st.session_state.approved_mask_ids = set()
        st.session_state.synced_mask_ids = set()
        st.success(f"Đã tải Task {task_id}, Job {job_id} với {ctx_loaded['image_count']} frame.")
    except Exception as exc:
        st.sidebar.error(f"Không tải được Task / Job: {exc}")

ctx = st.session_state.task_context
if not ctx:
    st.info("Chọn Project, Task và Job trong sidebar, sau đó bấm **Tải Task / Job**.")
    st.stop()
if (
    int(ctx["task_id"]) != int(task_id)
    or int(ctx.get("job_id") or 0) != int(job_id or 0)
    or st.session_state.get("loaded_context_url") != client.base_url
):
    st.info("Đã chọn Task hoặc Job khác. Bấm **Tải Task / Job** trước khi dự đoán hay đồng bộ.")
    st.stop()
with st.sidebar.expander("Labels của Task"):
    for label in ctx["labels"]:
        st.write(f"• {label['name']} · {label.get('type', 'any')} · id {label['id']}")

summary = st.columns(4)
summary[0].metric("Task", ctx["task"]["name"])
summary[1].metric("Job", ctx["job_id"] or "—")
summary[2].metric("Frames", ctx["image_count"])
summary[3].metric("Labels", len(ctx["labels"]))
facial_task = is_facial_landmark_task(ctx["label_names"])
if facial_task:
    st.caption(f"Annotation: Facial Landmark-50 · Job state: {(ctx.get('job') or {}).get('state', 'unknown')}")
    workspace_step = st.segmented_control(
        "Quy trình", ["Facial Landmark-50", "Xóa nhãn frame"],
        default="Facial Landmark-50", key=f"workspace_step_{task_id}_{job_id}_facial",
        label_visibility="collapsed",
    )
    if workspace_step == "Facial Landmark-50":
        render_cvat_facial_workspace(ctx, client)
    elif workspace_step == "Xóa nhãn frame":
        show_frame_cleanup_ui(ctx, client)
    st.stop()

st.caption(f"Segmentation: {ctx['segmentation_type']} · Job state: {(ctx.get('job') or {}).get('state', 'unknown')}")
workspace_step = st.segmented_control(
    "Quy trình",
    ["Tạo nhãn", "Duyệt & chỉnh sửa", "Đánh giá", "Đồng bộ CVAT", "Xóa nhãn frame"],
    default="Tạo nhãn",
    key=f"workspace_step_{task_id}_{job_id}",
    label_visibility="collapsed",
)
if workspace_step == "Tạo nhãn":
    st.header("1. Inference")
    if not ctx.get("frames"):
        st.warning("Job không có frame để xử lý.")
        st.stop()
    mode = st.segmented_control(
        "Chế độ gán nhãn",
        ["Manual + AI", "YOLO Only", "SAM3 Only", "YOLO + SAM3", "SAM3 Video Track"],
        default="Manual + AI",
        key=f"inference_mode_{task_id}_{job_id}",
    )
    seg_type = ctx["segmentation_type"]
    fallback_labels = sam3_fallback_labels(mode, seg_type, ctx["label_names"])
    if mode == "Manual + AI":
        cfg = get_model_config(SegmentationType(seg_type)) if seg_type in ("semantic", "instance", "panoptic") else None
        if cfg:
            st.info(f"Mô hình theo Task: {describe_model(SegmentationType(seg_type))}")
        elif fallback_labels:
            st.info("Task type chưa nhận diện; SAM 3 sẽ tạo mask cho các label tùy biến của Task.")
        else:
            st.warning("Task type chưa nhận diện; hãy chọn YOLO Only hoặc SAM3 Only.")
        if ctx["task"]["name"].strip().casefold() == "medium_instance":
            self_annotation_complete = st.checkbox("Tôi đã tự vẽ object đầu tiên trong CVAT và ghi report", key=f"medium_first_object_done_{ctx['task_id']}")
        else:
            self_annotation_complete = True
    else:
        self_annotation_complete = True

    cuda_available = bool(device_status.get("cuda_available"))
    sam_mode = mode in ("SAM3 Only", "YOLO + SAM3", "SAM3 Video Track") or bool(fallback_labels)
    sam3_checkpoint_ready = False
    with st.expander("Cấu hình model", expanded=False):
        weight_files = sorted({str(path.resolve()) for folder in (ROOT / "models", ROOT) for path in folder.glob("*.pt") if path.is_file()})
        if not weight_files:
            weight_files = ["yolov8n-seg.pt"]
        default_weight = next((path for path in weight_files if Path(path).name == "yolov8n-seg.pt"), weight_files[0])
        weight_options = weight_files + ["Custom path…"]
        yolo_weights = st.selectbox("YOLO weights", weight_options, index=weight_options.index(default_weight))
        if yolo_weights == "Custom path…":
            yolo_weights = st.text_input("Đường dẫn weights YOLO", value="").strip()
        sam3_weights = st.text_input(
            "Checkpoint SAM 3",
            value=os.getenv("SAM3_WEIGHTS", str(ROOT / "models" / "sam3.pt")),
            key="sam3_weights",
            help="Checkpoint Meta cấp quyền. Có thể dùng sam3.pt hoặc sam3.1_multiplex.pt cho tác vụ ảnh.",
        )
        sam3_checkpoint_ready = bool(sam3_weights) and Path(sam3_weights).expanduser().is_file()
        if sam_mode and not sam3_checkpoint_ready:
            st.caption("Chưa tìm thấy checkpoint ở đường dẫn này. Đặt file SAM 3 đã được cấp quyền vào `models/` hoặc nhập đúng đường dẫn.")
        confidence = st.slider(
            "Ngưỡng confidence",
            0.0,
            1.0,
            0.35,
            0.01,
            help="Mặc định cao hơn để giảm dự đoán yếu. Hạ ngưỡng nếu bị thiếu đối tượng; dùng 0% khi calibrate trên Ground Truth.",
            key="inference_confidence_threshold_v2",
        )
        min_component_area_px = st.slider(
            "Bỏ vùng mask nhỏ hơn (pixel)",
            min_value=16,
            max_value=2000,
            value=96,
            step=16,
            help="Bỏ mảnh nhiễu và vật thể rất nhỏ trước khi duyệt. Giảm giá trị nếu cần giữ đối tượng xa hoặc nhỏ.",
            key="inference_min_component_area_px_v1",
        )
        devices = available_devices(device_status)
        device_names = {
            device_name: (
                f"GPU {device_name.removeprefix('cuda:')} · {device_status['gpu_names'][int(device_name.split(':')[1])] }"
                if device_name.startswith("cuda:") and int(device_name.split(":")[1]) < len(device_status.get("gpu_names", []))
                else "CPU"
            )
            for device_name in devices
        }
        device = st.selectbox(
            "Thiết bị inference",
            devices,
            index=0,
            format_func=lambda value: device_names[value],
            key="inference_device_v2",
        )
        st.caption("SAM 3 dùng FP16 và kích thước 640 để tiết kiệm VRAM. Chạy inference ở 0% trên Ground Truth để tìm ngưỡng confidence.")

    yolo_label_mapping: Optional[Dict[str, str]] = None
    if mode in ("YOLO Only", "YOLO + SAM3"):
        weight_signature = hashlib.sha1(str(yolo_weights).strip().encode("utf-8")).hexdigest()[:12]
        class_names_state_key = f"yolo_class_names_{weight_signature}"
        if st.button(
            "Đọc danh sách class từ weights YOLO",
            key=f"load_yolo_classes_{task_id}_{job_id}_{weight_signature}",
            disabled=not str(yolo_weights).strip(),
        ):
            try:
                with st.spinner("Đang đọc class names từ weights YOLO..."):
                    yolo_model_for_mapping = get_yolo_model(yolo_weights, device)
                    loaded_class_names = list(yolo_model_for_mapping.class_names)
                if not loaded_class_names:
                    st.warning("Weights này không cung cấp class names.")
                else:
                    st.session_state[class_names_state_key] = loaded_class_names
            except Exception as exc:
                st.error(f"Không đọc được class names từ weights YOLO: {exc}")

        yolo_model_class_names = st.session_state.get(class_names_state_key)
        if yolo_model_class_names:
            st.caption(
                "Tự khớp tên class với label CVAT; chỉnh cột CVAT label để đổi tên hoặc chọn bỏ qua. "
                "Ánh xạ chỉ đổi tên dự đoán, không thêm khả năng nhận diện mới cho model."
            )
            skip_label = "— Bỏ qua —"
            task_label_by_normalized_name = {
                normalize_label_name(name): name for name in ctx["label_names"]
            }
            mapping_rows = [
                {
                    "YOLO class": class_name,
                    "CVAT label": task_label_by_normalized_name.get(
                        normalize_label_name(class_name), skip_label
                    ),
                }
                for class_name in yolo_model_class_names
            ]
            edited_mapping_rows = st.data_editor(
                mapping_rows,
                key=f"yolo_label_map_{task_id}_{job_id}_{weight_signature}",
                column_config={
                    "CVAT label": st.column_config.SelectboxColumn(
                        "CVAT label",
                        options=[skip_label, *ctx["label_names"]],
                    ),
                },
                disabled=["YOLO class"],
                hide_index=True,
                num_rows="fixed",
                height=420,
                width="stretch",
            )
            yolo_label_mapping = {
                str(row["YOLO class"]): str(row["CVAT label"])
                for row in edited_mapping_rows
                if str(row["CVAT label"]) != skip_label
            }
            covered_labels = {name.casefold() for name in yolo_label_mapping.values()}
            missing_labels = [
                name for name in ctx["label_names"] if name.casefold() not in covered_labels
            ]
            st.caption(
                f"Đã ánh xạ {len(yolo_label_mapping)} class YOLO sang "
                f"{len(covered_labels)} / {len(ctx['label_names'])} label của Task."
            )
            if missing_labels:
                if mode == "YOLO + SAM3":
                    st.warning(
                        f"YOLO weights chỉ ánh xạ được {len(covered_labels)} / "
                        f"{len(ctx['label_names'])} label. Các label còn lại sẽ được thử bằng "
                        "SAM 3 text prompt bên dưới; để YOLO tự nhận diện chúng, cần weights "
                        "đã huấn luyện với các class tương ứng."
                    )
                else:
                    st.warning(
                        "Weights hiện chưa có class được ánh xạ cho: "
                        + ", ".join(missing_labels)
                        + ". Muốn YOLO nhận diện các lớp này, hãy dùng weights đã huấn luyện với chúng "
                        "hoặc chọn YOLO + SAM3 / Manual + AI."
                    )
        elif mode == "YOLO Only":
            st.info(
                "YOLO Only tạo annotation cho class có trong weights. Đọc danh sách class để xem độ phủ "
                "và ánh xạ sang label của Task; label ngoài dữ liệu huấn luyện cần weights riêng."
            )

    supported_labels = model_supported_labels(mode, seg_type, yolo_label_mapping)
    unsupported_labels = (
        [name for name in ctx["label_names"] if name.casefold() not in supported_labels]
        if supported_labels is not None
        else []
    )
    fallback_labels = sam3_fallback_labels(
        mode,
        seg_type,
        ctx["label_names"],
        yolo_label_mapping,
    )
    if fallback_labels:
        st.info(
            "Nhãn ngoài taxonomy sẽ được bổ sung bằng SAM 3 text prompt: "
            + ", ".join(fallback_labels)
            + ". Hãy duyệt kỹ mask area/lane trước khi đồng bộ."
        )
        with st.expander("Xem prompt SAM 3 tự động", expanded=False):
            st.dataframe(
                [{"CVAT label": name, "SAM 3 text prompt": sam3_prompt_for_label(name)} for name in fallback_labels],
                hide_index=True,
                width="stretch",
            )
    elif unsupported_labels and yolo_label_mapping is not None:
        st.warning(
            "Weights YOLO chưa ánh xạ class cho: "
            + ", ".join(unsupported_labels)
            + ". Hãy chọn weights huấn luyện các lớp này hoặc dùng chế độ có SAM 3."
        )

    if sam_mode and not cuda_available:
        st.warning("SAM 3 cần CUDA. Chạy setup_gpu.ps1 để cài PyTorch CUDA; YOLO vẫn có thể chạy CPU.")
    if fallback_labels and cuda_available and not sam3_checkpoint_ready:
        st.warning("SAM 3 fallback cần checkpoint được cấp quyền. Đăng nhập Hugging Face bằng `.venv-sam3\\Scripts\\hf.exe auth login`, rồi chạy lại `run_gpu.ps1` để tool tự tải vào models/.")
    if fallback_labels and device.startswith("cuda"):
        gpu_index = int(device.split(":", 1)[1]) if ":" in device else 0
        gpu_memory = device_status.get("gpu_memory_gib", [])
        if gpu_index < len(gpu_memory) and gpu_memory[gpu_index] <= 8:
            st.caption("Để dành VRAM cho SAM 3, model Cityscapes/YOLO hỗ trợ nhãn còn lại sẽ chạy trên CPU khi fallback bật.")

    frame_options = {}
    for index, frame in enumerate(ctx["frames"]):
        number = frame_number(frame, index)
        frame_options[f"{number}: {frame.get('name') or f'frame_{number}'}"] = number
    selected_frame_label = st.selectbox("Frame", list(frame_options))
    selected_frame_number = frame_options[selected_frame_label]
    sam_prompts: Dict[str, Any] = {}
    has_prompt = True

    if mode in ("SAM3 Only", "SAM3 Video Track"):
        prompt_kind = st.segmented_control(
            "Prompt SAM 3",
            ["Box", "Điểm", "Text", "Exemplar", "Text + exemplar"],
            default="Text",
            key=f"sam_prompt_kind_{task_id}_{job_id}",
        )
        if not ctx["label_names"]:
            st.error("Task phải có ít nhất một CVAT label.")
            has_prompt = False
        else:
            class_name = st.selectbox("CVAT label", ctx["label_names"])
            image_key = f"sam_prompt_image:{ctx['task_id']}:{ctx['job_id']}:{selected_frame_number}"
            if st.session_state.get("prompt_image_key") != image_key:
                try:
                    st.session_state.prompt_image = load_image_from_cvat(
                        client,
                        ctx["task_id"],
                        selected_frame_number,
                        ctx.get("job_id"),
                    )
                    st.session_state.prompt_image_key = image_key
                except Exception as exc:
                    st.error(f"Không tải được frame để đặt prompt: {exc}")
            prompt_image = st.session_state.get("prompt_image")
            if prompt_image is None:
                has_prompt = False
            else:
                try:
                    from streamlit_image_coordinates import streamlit_image_coordinates
                    canvas_width = min(640, prompt_image.shape[1])
                    if prompt_kind in ("Box", "Exemplar", "Text + exemplar"):
                        st.caption("Kéo chuột từ một góc sang góc đối diện.")
                        box_key = f"sam_box:{ctx['task_id']}:{ctx['job_id']}:{selected_frame_number}"
                        gen_key, last_key = f"{box_key}:generation", f"{box_key}:last"
                        st.session_state.setdefault(gen_key, 0)
                        drag = streamlit_image_coordinates(Image.fromarray(prompt_image), width=canvas_width, click_and_drag=True, cursor="crosshair", key=f"{box_key}:canvas:{st.session_state[gen_key]}")
                        if drag and drag.get("unix_time") != st.session_state.get(last_key):
                            shown_w = max(float(drag.get("width", canvas_width)), 1.0)
                            shown_h = max(float(drag.get("height", prompt_image.shape[0])), 1.0)
                            sx, sy = prompt_image.shape[1] / shown_w, prompt_image.shape[0] / shown_h
                            x1, x2 = sorted((int(drag.get("x1", 0) * sx), int(drag.get("x2", 0) * sx)))
                            y1, y2 = sorted((int(drag.get("y1", 0) * sy), int(drag.get("y2", 0) * sy)))
                            st.session_state[f"{box_key}:value"] = [x1, y1, x2, y2]
                            st.session_state[last_key] = drag.get("unix_time")
                        drawn_box = st.session_state.get(f"{box_key}:value")
                        if prompt_kind == "Box":
                            sam_prompts["bbox"] = drawn_box
                            st.caption(f"Bounding box: {drawn_box}")
                        elif drawn_box:
                            sam_prompts["exemplar_boxes"] = [drawn_box]
                            st.caption(f"Exemplar box: {drawn_box} · SAM 3 sẽ tìm các object tương tự.")
                        if prompt_kind in ("Text", "Text + exemplar"):
                            sam_prompts["text_prompt"] = st.text_input(
                                "Mô tả ngắn đối tượng",
                                value=sam3_prompt_for_label(class_name),
                                key=f"sam_text_prompt_{task_id}_{job_id}",
                            ).strip()
                        if st.button("Xóa box", key=f"{box_key}:clear"):
                            st.session_state.pop(f"{box_key}:value", None)
                            st.session_state[gen_key] += 1
                            st.session_state.pop(last_key, None)
                            st.rerun()
                    elif prompt_kind == "Điểm":
                        polarity = st.segmented_control(
                            "Điểm tiếp theo",
                            ["Positive", "Negative"],
                            key=f"sam_point_polarity_{task_id}_{job_id}_{selected_frame_number}",
                        )
                        points_key = f"sam_points:{ctx['task_id']}:{ctx['job_id']}:{selected_frame_number}"
                        gen_key = f"{points_key}:generation"
                        st.session_state.setdefault(points_key, [])
                        st.session_state.setdefault(gen_key, 0)
                        click = streamlit_image_coordinates(Image.fromarray(prompt_image), width=canvas_width, cursor="crosshair", key=f"{points_key}:canvas:{st.session_state[gen_key]}")
                        _record_point_click(click, points_key, f"{points_key}:last_event", prompt_image.shape[1], prompt_image.shape[0], 1 if polarity == "Positive" else 0)
                        sam_prompts["point_coords"] = [(x, y) for x, y, _ in st.session_state[points_key]]
                        sam_prompts["point_labels"] = [label for _, _, label in st.session_state[points_key]]
                        st.caption(f"Points: {st.session_state[points_key]}")
                        if st.button("Xóa points", key=f"{points_key}:clear"):
                            st.session_state[points_key] = []
                            st.session_state[gen_key] += 1
                            st.session_state.pop(f"{points_key}:last_event", None)
                            st.rerun()
                    else:
                        sam_prompts["text_prompt"] = st.text_input(
                            "Mô tả ngắn đối tượng",
                            value=sam3_prompt_for_label(class_name),
                            key=f"sam_text_prompt_{task_id}_{job_id}",
                        ).strip()
                except ImportError:
                    st.error("Thiếu streamlit-image-coordinates. Cài requirements.txt.")
                    has_prompt = False
            sam_prompts["class_name"] = class_name
            has_prompt = has_prompt and bool(
                sam_prompts.get("bbox")
                or sam_prompts.get("point_coords")
                or sam_prompts.get("text_prompt")
                or sam_prompts.get("exemplar_boxes")
            )

    col_all, col_one = st.columns(2)
    disable_sam = sam_mode and (
        not cuda_available or not device.startswith("cuda") or not sam3_checkpoint_ready
    )
    with col_all:
        text_batch = mode == "SAM3 Only" and bool(sam_prompts.get("text_prompt")) and not sam_prompts.get("exemplar_boxes")
        batch_label = "▶ Theo dõi từ frame đã chọn" if mode == "SAM3 Video Track" else "🚀 Chạy toàn bộ frame"
        batch_disabled = (
            not self_annotation_complete
            or (mode == "SAM3 Only" and not text_batch)
            or (mode == "SAM3 Video Track" and not has_prompt)
            or disable_sam
            or (
                mode == "Manual + AI"
                and seg_type not in ("semantic", "instance", "panoptic")
                and not fallback_labels
            )
        )
        if st.button(batch_label, type="primary", disabled=batch_disabled):
            run_prediction(
                ctx,
                client,
                confidence,
                device,
                mode=mode,
                yolo_weights=yolo_weights,
                sam3_weights=sam3_weights,
                selected_frame_number=selected_frame_number,
                sam_prompts=sam_prompts,
                yolo_label_mapping=yolo_label_mapping,
                min_component_area_px=min_component_area_px,
            )
    with col_one:
        frame_disabled = not self_annotation_complete or disable_sam or not has_prompt or mode == "SAM3 Video Track"
        if st.button("🖼️ Chạy frame đã chọn", disabled=frame_disabled):
            run_prediction(
                ctx,
                client,
                confidence,
                device,
                mode=mode,
                yolo_weights=yolo_weights,
                sam3_weights=sam3_weights,
                current_only=True,
                selected_frame_number=selected_frame_number,
                sam_prompts=sam_prompts,
                yolo_label_mapping=yolo_label_mapping,
                min_component_area_px=min_component_area_px,
            )

predictions = st.session_state.predictions
if not isinstance(predictions, PredictionStore) or not predictions.frame_names:
    if workspace_step == "Xóa nhãn frame":
        show_frame_cleanup_ui(ctx, client)
    elif workspace_step != "Tạo nhãn":
        st.info("Chạy inference trước, sau đó các bước duyệt, đánh giá và đồng bộ sẽ sẵn sàng.")
else:
    if workspace_step == "Duyệt & chỉnh sửa":
        st.header("Duyệt và chỉnh sửa")
        processed, total = len(predictions.frame_names), max(int(ctx["image_count"]), 1)
        review_count = sum(
            1 for frame_meta in predictions.iter_metadata() for item in frame_meta["masks"]
            if item["approval_key"] not in st.session_state.approved_mask_ids
            and item["approval_key"] not in st.session_state.synced_mask_ids
        )
        cols = st.columns(4)
        cols[0].metric("Đã xử lý", processed)
        cols[1].metric("Chưa xử lý", max(total - processed, 0))
        cols[2].metric("Cần duyệt", review_count)
        cols[3].metric("Đã đồng bộ", len(st.session_state.synced_mask_ids))
        st.progress(min(processed / total, 1.0), text=f"{processed}/{total} frame")
        with st.expander("Labels của Task", expanded=False):
            st.write(" · ".join(ctx["label_names"]))
        with st.expander("Tóm tắt mask theo frame", expanded=False):
            frame_summary = [
                {
                    "Frame": frame_meta["name"],
                    "Masks": len(frame_meta["masks"]),
                    "Detections": "; ".join(
                        f"{item['class_name']}"
                        + (f" #track {item['track_id']}" if item.get("track_id") is not None else "")
                        + f" ({item['confidence']:.2f}, QC {item['qc']['level']})"
                        for item in frame_meta["masks"]
                    ),
                }
                for frame_meta in predictions.iter_metadata()
            ]
            st.dataframe(frame_summary, hide_index=True)
        show_prediction_review(predictions, ctx["label_names"], ctx)

    elif workspace_step == "Đánh giá":
        st.header("Đánh giá Ground Truth")
        st.caption(
            "Metrics tự tính khi mở bước này và chỉ dùng các frame đã có prediction. "
            "Việc xác nhận annotation là Ground Truth vẫn cần người kiểm tra."
        )
        annotation_snapshot = ctx.get("annotations") or {}
        gt_revision = (
            annotation_snapshot.get("version", "unknown"),
            len(annotation_snapshot.get("shapes", [])),
            len(annotation_snapshot.get("tracks", [])),
        )
        gt_confirmed = st.checkbox(
            "Tôi đã kiểm tra annotation Job này và xác nhận là Ground Truth",
            key=f"gt_confirm_{ctx['task_id']}_{ctx['job_id']}_{gt_revision[0]}_{gt_revision[1]}_{gt_revision[2]}",
            help="Ô này chỉ ghi nhận xác nhận của bạn; không ảnh hưởng việc app tự tính metrics.",
        )
        if gt_confirmed:
            st.success("Annotation đã được bạn xác nhận là Ground Truth.")
        else:
            st.info("Metrics bên dưới chỉ mang tính tham khảo cho đến khi bạn xác nhận Ground Truth.")
        target_precision = st.slider(
            "Mục tiêu Precision trên Job này",
            min_value=0.50,
            max_value=1.00,
            value=0.90,
            step=0.01,
            key=f"target_precision_{ctx['task_id']}_{ctx['job_id']}",
        )
        evaluation_cache = st.session_state.setdefault("ground_truth_eval_cache", {})
        evaluation_key = ground_truth_evaluation_cache_key(ctx, predictions, target_precision)
        evaluation = evaluation_cache.get(evaluation_key)
        if evaluation is None:
            with st.spinner("Đang tự đánh giá prediction với annotation CVAT..."):
                evaluation = compute_ground_truth_evaluation(ctx, predictions, target_precision)
            if len(evaluation_cache) >= 32:
                evaluation_cache.clear()
            evaluation_cache[evaluation_key] = evaluation

        if not evaluation["ground_truth_count"]:
            annotations = ctx.get("annotations") or {}
            shape_count = len(annotations.get("shapes", []))
            track_count = len(annotations.get("tracks", []))
            st.warning("Chưa tìm thấy Ground Truth dạng polygon, rectangle hoặc mask trong các frame đã chạy.")
            if track_count:
                st.caption(
                    f"CVAT Job có {shape_count} shape và {track_count} track. "
                    "Bộ đánh giá hiện chỉ đọc shape polygon, rectangle và mask; track CVAT chưa được chuyển thành Ground Truth."
                )
            else:
                st.caption(f"CVAT Job hiện có {shape_count} shape annotation trong snapshot đã tải.")
            st.info(
                "Dấu tích ở bước Duyệt & chỉnh sửa chỉ xác nhận prediction AI để đồng bộ, "
                "không biến prediction thành Ground Truth. Hãy tạo/kiểm tra annotation chuẩn trên CVAT "
                "rồi tải lại Task / Job trước khi đánh giá."
            )
        else:
            scores = evaluation["scores"]
            cols = st.columns(5)
            cols[0].metric("Precision", f"{scores['precision']:.1%}")
            cols[1].metric("Recall", f"{scores['recall']:.1%}")
            cols[2].metric("Mask IoU", f"{scores['iou']:.1%}")
            cols[3].metric("Dice", f"{scores['dice']:.1%}")
            cols[4].metric("mAP@50", f"{scores['map50']:.1%}")
            st.caption(
                f"TP {scores['tp']} · FP {scores['fp']} · FN {scores['fn']} · "
                f"GT {scores['ground_truth_count']} · "
                f"Frame {evaluation['evaluated_frame_count']}/{evaluation['job_frame_count']} đã chạy."
            )
            if evaluation["evaluated_frame_count"] < evaluation["job_frame_count"]:
                st.info("Run hiện tại chưa bao phủ toàn bộ Job; metrics chỉ đại diện cho các frame đã chạy.")
            st.json(scores["per_class"])

            if evaluation["missing_threshold_metadata"]:
                st.warning("Lần inference này chưa lưu ngưỡng sinh candidate. Hãy chạy lại inference ở 0% rồi đánh giá.")
            elif evaluation["mixed_thresholds"]:
                st.warning("Các frame được sinh ở ngưỡng khác nhau; hãy tạo một run mới ở 0% để calibrate.")
            else:
                calibration = evaluation["calibration"]
                if calibration["status"] == "missing_confidence":
                    st.warning("Không thể calibrate: một số prediction thiếu confidence score, như SAM 3 text prompt.")
                elif calibration["status"] == "no_predictions":
                    st.warning("Không có prediction đủ ngưỡng để calibrate.")
                elif calibration["status"] == "target_met":
                    st.success(
                        f"Trên Job này, confidence ≥ {calibration['threshold']:.2f} cho "
                        f"Precision {calibration['precision']:.1%}, Recall {calibration['recall']:.1%} "
                        f"(TP {calibration['tp']}, FP {calibration['fp']})."
                    )
                    st.caption("Đặt slider confidence ở giá trị gợi ý, chạy inference lại rồi đo lại. Kết quả chỉ mô tả Job đã chọn.")
                else:
                    st.warning(
                        f"Không tìm được ngưỡng đạt Precision {target_precision:.0%}. "
                        f"Tốt nhất trên Job này là {calibration['precision']:.1%} "
                        f"ở ngưỡng {calibration['threshold']:.2f}; có thể cần weights hoặc dữ liệu phù hợp hơn."
                    )

    elif workspace_step == "Đồng bộ CVAT":
        st.header("Đồng bộ annotation lên CVAT")
        show_sync_ui(ctx, client, predictions)
    elif workspace_step == "Xóa nhãn frame":
        show_frame_cleanup_ui(ctx, client)
