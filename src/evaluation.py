"""Ground-truth evaluation for object masks and CVAT job annotations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from src.annotation_processor import draw_cvat_mask


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Mask dimensions do not match: {left.shape} vs {right.shape}")
    a = left.astype(bool)
    b = right.astype(bool)
    union = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def mask_dice(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Mask dimensions do not match: {left.shape} vs {right.shape}")
    a = left.astype(bool)
    b = right.astype(bool)
    denominator = int(a.sum() + b.sum())
    return float(2 * np.logical_and(a, b).sum() / denominator) if denominator else 1.0


def evaluate_instances(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute precision, recall, mean matched IoU/Dice, and mask mAP@50.

    Every record has ``image_id``, ``class_name`` and a binary ``mask``;
    predictions can include a numeric ``confidence``. No model confidence is
    converted into an accuracy claim.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1].")

    gt_by_key: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    pred_by_class: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in ground_truth:
        gt_by_key[(str(item["image_id"]), str(item["class_name"]))].append(item)
    for item in predictions:
        pred_by_class[str(item["class_name"])].append(item)

    classes = sorted(
        {class_name for _, class_name in gt_by_key}
        | set(pred_by_class)
    )
    per_class: Dict[str, Dict[str, float]] = {}
    total_tp = total_fp = total_gt = 0
    all_ious: List[float] = []
    all_dices: List[float] = []
    average_precisions: List[float] = []

    for class_name in classes:
        class_gt = {
            image_id: list(items)
            for (image_id, name), items in gt_by_key.items()
            if name == class_name
        }
        class_gt_count = sum(len(items) for items in class_gt.values())
        matched: Dict[str, set[int]] = {image_id: set() for image_id in class_gt}
        ordered = sorted(
            pred_by_class.get(class_name, []),
            key=lambda item: float(item.get("confidence") or 0.0),
            reverse=True,
        )
        tp_flags: List[int] = []
        fp_flags: List[int] = []
        class_ious: List[float] = []
        class_dices: List[float] = []
        for prediction in ordered:
            image_id = str(prediction["image_id"])
            candidates = class_gt.get(image_id, [])
            available = [
                (idx, item) for idx, item in enumerate(candidates)
                if idx not in matched.get(image_id, set())
            ]
            scores = [mask_iou(prediction["mask"], item["mask"]) for _, item in available]
            best_pos = int(np.argmax(scores)) if scores else -1
            best_iou = scores[best_pos] if best_pos >= 0 else 0.0
            if best_iou >= iou_threshold:
                gt_index = available[best_pos][0]
                matched.setdefault(image_id, set()).add(gt_index)
                tp_flags.append(1)
                fp_flags.append(0)
                class_ious.append(best_iou)
                class_dices.append(mask_dice(prediction["mask"], available[best_pos][1]["mask"]))
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        tp = sum(tp_flags)
        fp = sum(fp_flags)
        fn = class_gt_count - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / class_gt_count if class_gt_count else 0.0
        ap = _average_precision(tp_flags, fp_flags, class_gt_count)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "iou": float(np.mean(class_ious)) if class_ious else 0.0,
            "dice": float(np.mean(class_dices)) if class_dices else 0.0,
            "ap50": ap,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        total_tp += tp
        total_fp += fp
        total_gt += class_gt_count
        all_ious.extend(class_ious)
        all_dices.extend(class_dices)
        if class_gt_count:
            average_precisions.append(ap)

    return {
        "iou_threshold": float(iou_threshold),
        "precision": total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
        "recall": total_tp / total_gt if total_gt else 0.0,
        "iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "dice": float(np.mean(all_dices)) if all_dices else 0.0,
        "map50": float(np.mean(average_precisions)) if average_precisions else 0.0,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_gt - total_tp,
        "ground_truth_count": total_gt,
        "per_class": per_class,
    }


def recommend_precision_threshold(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    target_precision: float = 0.9,
    iou_threshold: float = 0.5,
    minimum_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Find the cutoff with highest measured recall that meets target precision.

    The result is empirical for the supplied ground truth and the candidate
    predictions that were retained during inference. It is not a guarantee on
    unseen images.
    """
    if not 0.0 <= target_precision <= 1.0:
        raise ValueError("target_precision must be in [0, 1].")
    if not 0.0 <= minimum_threshold <= 1.0:
        raise ValueError("minimum_threshold must be in [0, 1].")

    scored_predictions: List[Mapping[str, Any]] = []
    for item in predictions:
        confidence = item.get("confidence")
        if confidence is None:
            return {
                "status": "missing_confidence",
                "target_precision": float(target_precision),
            }
        value = float(confidence)
        if not np.isfinite(value):
            return {
                "status": "missing_confidence",
                "target_precision": float(target_precision),
            }
        scored_predictions.append(item)

    minimum_percent = int(np.ceil(minimum_threshold * 100.0 - 1e-9))
    gt_by_key: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    predictions_by_class: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in ground_truth:
        gt_by_key[(str(item["image_id"]), str(item["class_name"]))].append(item)
    for item in scored_predictions:
        predictions_by_class[str(item["class_name"])].append(item)

    outcomes: List[Tuple[float, int, int]] = []
    for class_name, class_predictions in predictions_by_class.items():
        matched: Dict[str, set[int]] = defaultdict(set)
        ordered = sorted(
            class_predictions,
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )
        for prediction in ordered:
            image_id = str(prediction["image_id"])
            candidates_for_image = gt_by_key.get((image_id, class_name), [])
            available = [
                (idx, item) for idx, item in enumerate(candidates_for_image)
                if idx not in matched[image_id]
            ]
            ious = [mask_iou(prediction["mask"], item["mask"]) for _, item in available]
            best_pos = int(np.argmax(ious)) if ious else -1
            best_iou = ious[best_pos] if best_pos >= 0 else 0.0
            if best_iou >= iou_threshold:
                matched[image_id].add(available[best_pos][0])
                outcomes.append((float(prediction["confidence"]), 1, 0))
            else:
                outcomes.append((float(prediction["confidence"]), 0, 1))

    total_gt = len(ground_truth)
    ordered_outcomes = sorted(outcomes, key=lambda item: item[0], reverse=True)
    candidates: List[Dict[str, Any]] = []
    cursor = total_tp = total_fp = 0
    for percent in range(100, minimum_percent - 1, -1):
        threshold = percent / 100.0
        while cursor < len(ordered_outcomes) and ordered_outcomes[cursor][0] >= threshold:
            _, tp, fp = ordered_outcomes[cursor]
            total_tp += tp
            total_fp += fp
            cursor += 1
        prediction_count = total_tp + total_fp
        if not prediction_count:
            continue
        candidates.append({
            "threshold": threshold,
            "precision": total_tp / prediction_count,
            "recall": total_tp / total_gt if total_gt else 0.0,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_gt - total_tp,
            "prediction_count": prediction_count,
        })

    if not candidates:
        return {
            "status": "no_predictions",
            "target_precision": float(target_precision),
            "minimum_threshold": float(minimum_threshold),
        }

    eligible = [item for item in candidates if item["precision"] >= target_precision]
    if eligible:
        best = max(
            eligible,
            key=lambda item: (item["recall"], item["precision"], -item["threshold"]),
        )
        status = "target_met"
    else:
        best = max(
            candidates,
            key=lambda item: (item["precision"], item["recall"], -item["threshold"]),
        )
        status = "target_unmet"

    return {
        "status": status,
        "target_precision": float(target_precision),
        **best,
    }


def _average_precision(tp_flags: Sequence[int], fp_flags: Sequence[int], gt_count: int) -> float:
    if gt_count == 0:
        return 0.0
    if not tp_flags:
        return 0.0
    tp = np.cumsum(np.asarray(tp_flags, dtype=np.float64))
    fp = np.cumsum(np.asarray(fp_flags, dtype=np.float64))
    recall = tp / gt_count
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def cvat_shapes_to_records(
    annotations: Mapping[str, Any],
    labels: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Rasterize rectangle, polygon and this tool's CVAT mask shapes as GT.

    CVAT Job annotation frame IDs remain Task frame IDs, including for Jobs
    whose start frame is greater than zero.
    """
    label_names = {int(label["id"]): str(label["name"]) for label in labels}
    frame_lookup = {
        int(frame.get("number", index)): str(frame.get("name", frame.get("number", index)))
        for index, frame in enumerate(frames)
    }
    size_lookup = {
        int(frame.get("number", index)): (int(frame["height"]), int(frame["width"]))
        for index, frame in enumerate(frames)
        if frame.get("height") and frame.get("width")
    }
    records: List[Dict[str, Any]] = []
    for shape in annotations.get("shapes", []):
        label_name = label_names.get(int(shape.get("label_id", -1)))
        if not label_name:
            continue
        task_frame = int(shape.get("frame", 0))
        if task_frame not in frame_lookup or task_frame not in size_lookup:
            continue
        image_id = frame_lookup[task_frame]
        height, width = size_lookup[task_frame]
        canvas = np.zeros((height, width), dtype=np.uint8)
        points = shape.get("points", [])
        shape_type = str(shape.get("type", "")).casefold()
        if shape_type == "rectangle" and len(points) >= 4:
            x1, y1, x2, y2 = [int(round(value)) for value in points[:4]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), 1, thickness=-1)
        elif shape_type == "polygon" and len(points) >= 6:
            polygon = np.asarray(points, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
            cv2.fillPoly(canvas, [polygon], 1)
        elif shape_type == "mask" and len(points) > 4:
            draw_cvat_mask(canvas, points)
        if canvas.any():
            records.append({"image_id": image_id, "class_name": label_name, "mask": canvas, "confidence": 1.0})

    return records
