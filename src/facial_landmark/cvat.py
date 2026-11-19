"""Strict CVAT taxonomy mapping and safe Job-frame writes for face skeletons."""
from __future__ import annotations

import math
import html
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Mapping, Sequence

from src.annotation_processor import get_cvat_sublabels
from src.cvat_client import CVATAPIError, CVATClient
from src.guideline_annotation import POINT_STATES, SKELETONS, point_state_flags

logger = logging.getLogger(__name__)


class FacialTaxonomyError(ValueError):
    """The connected CVAT Task cannot represent the 50-point guideline safely."""


def _numeric_point_id(name: str) -> int | None:
    match = re.search(r"(?:^|[^a-z0-9])(?:point|pt|id|p)?[_\s-]*(\d+)\s*$", name.strip(), re.I)
    if not match:
        return None
    return int(match.group(1))


def _parse_skeleton_svg(svg: Any, *, label_id: Any, label_name: str) -> ET.Element:
    """Parse CVAT SVG, accepting raw XML and HTML-escaped API payloads."""
    if isinstance(svg, dict):
        svg = svg.get("svg") or svg.get("content") or ""
    raw = str(svg or "").lstrip("\ufeff \t\r\n")
    candidates = [raw]
    decoded = raw
    for _ in range(2):
        decoded = html.unescape(decoded)
        if decoded not in candidates:
            candidates.append(decoded)
    last_error: ET.ParseError | None = None
    for candidate in candidates:
        try:
            return ET.fromstring(candidate)
        except ET.ParseError as exc:
            last_error = exc
        # Some CVAT versions serialize a skeleton SVG as sibling fragments
        # instead of one document root. Preserve the fragments inside a
        # synthetic root so node and edge records remain inspectable.
        fragment = re.sub(r"<\?xml[^>]*\?>", "", candidate, flags=re.IGNORECASE).strip()
        try:
            return ET.fromstring(f"<cvat-svg-fragments>{fragment}</cvat-svg-fragments>")
        except ET.ParseError:
            pass
    logger.error(
        "CVAT skeleton SVG parse failed: label_id=%r label=%r payload_type=%s "
        "payload_length=%d parse_error=%s svg_prefix=%r",
        label_id, label_name, type(svg).__name__, len(raw), last_error, raw[:400],
    )
    raise FacialTaxonomyError(
        f"`{label_name}` có SVG nhưng XML không hợp lệ (label_id={label_id}, "
        f"{last_error}; payload_length={len(raw)}). Xem log `src.facial_landmark.cvat` "
        "để biết 400 ký tự đầu của SVG."
    ) from last_error


def resolve_facial_taxonomy(labels: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Require seven skeleton parents and explicit child names encoding point IDs.

    Point order is never inferred from a list position. Child labels must be
    named with global IDs (e.g. ``point_14``) or local IDs (e.g. ``point_0``)
    within each parent group.
    """
    by_name = {str(label.get("name", "")).strip().casefold(): label for label in labels}
    resolved: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for spec in SKELETONS:
        name, start, count = spec["name"], int(spec["start"]), int(spec["count"])
        parent = by_name.get(name)
        if not parent:
            errors.append(f"Thiếu skeleton label `{name}`.")
            continue
        if str(parent.get("type", "any")).strip().casefold() != "skeleton":
            errors.append(f"`{name}` phải có kiểu Skeleton trong CVAT.")
            continue
        children = get_cvat_sublabels(labels, dict(parent))
        if len(children) != count:
            errors.append(f"`{name}` cần {count} point sublabel, CVAT trả về {len(children)}.")
            continue
        parsed: list[int | None] = [_numeric_point_id(str(child.get("name", ""))) for child in children]
        if all(value is not None for value in parsed):
            values = [int(value) for value in parsed if value is not None]
            if set(values) == set(range(start, start + count)):
                global_ids = values
            elif set(values) == set(range(count)):
                global_ids = [start + value for value in values]
            else:
                errors.append(f"`{name}`: tên point sublabel phải mã hóa ID {start}–{start + count - 1} hoặc ID cục bộ 0–{count - 1}.")
                continue
        else:
            errors.append(f"`{name}`: tên mỗi point sublabel phải mã hóa Point ID; không suy luận thứ tự từ danh sách CVAT.")
            continue
        if len(set(global_ids)) != count:
            errors.append(f"`{name}` có Point ID trùng.")
            continue
        child_by_point = {point_id: child for point_id, child in zip(global_ids, children)}
        if any(child.get("id") is None for child in children):
            errors.append(f"`{name}` có point sublabel thiếu CVAT ID.")
            continue
        svg = parent.get("svg")
        topology_warning = None
        svg_verified = False
        if not str(svg or "").strip():
            detail_error = parent.get("_svg_fetch_error")
            detail_result = parent.get("_svg_fetch_result")
            if detail_error:
                topology_warning = f"Không lấy được SVG qua GET /api/labels/{parent.get('id')}: {detail_error}"
            elif detail_result:
                topology_warning = str(detail_result)
            else:
                topology_warning = "Task/label detail API không cung cấp SVG skeleton."
        else:
            try:
                root = _parse_skeleton_svg(
                    svg,
                    label_id=parent.get("id"),
                    label_name=name,
                )
            except FacialTaxonomyError as exc:
                topology_warning = str(exc)
            else:
                node_to_point: dict[str, int] = {}
                for node in root.iter():
                    if node.attrib.get("data-type") != "element node":
                        continue
                    point_id = _numeric_point_id(node.attrib.get("data-label-name", ""))
                    node_id = node.attrib.get("data-node-id")
                    if point_id is not None and node_id is not None:
                        if point_id in range(count):
                            point_id += start
                        node_to_point[node_id] = point_id
                expected_point_ids = set(range(start, start + count))
                actual_edges = set()
                nodes_valid = set(node_to_point.values()) == expected_point_ids
                if nodes_valid:
                    for node in root.iter():
                        if node.attrib.get("data-type") != "edge":
                            continue
                        first = node_to_point.get(node.attrib.get("data-node-from", ""))
                        second = node_to_point.get(node.attrib.get("data-node-to", ""))
                        if first is not None and second is not None:
                            actual_edges.add(tuple(sorted((first, second))))
                    expected_edges = {
                        tuple(sorted((start + index, start + index + 1)))
                        for index in range(count - 1)
                    }
                    if spec["contour"] == "closed":
                        expected_edges.add(tuple(sorted((start, start + count - 1))))
                    svg_verified = actual_edges == expected_edges
                if not svg_verified:
                    topology_warning = (
                        f"SVG có nhưng topology không khớp {spec['contour']} Point ID order "
                        f"(nodes_valid={nodes_valid}, actual_edges={sorted(actual_edges)})."
                    )
        resolved[name] = {
            "parent": parent,
            "children": child_by_point,
            "spec": spec,
            "svg_verified": svg_verified,
            "topology_warning": topology_warning,
        }
    if errors:
        raise FacialTaxonomyError("CVAT skeleton taxonomy chưa khớp Facial Landmark-50:\n• " + "\n• ".join(errors))
    return resolved


def facial_annotations_to_cvat_shapes(
    annotations: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, Mapping[str, Any]],
    *,
    frame: int,
    group: int,
) -> list[dict[str, Any]]:
    """Serialize a validated single face instance into seven CVAT skeletons."""
    expected_names = [str(spec["name"]) for spec in SKELETONS]
    supplied_names = [str(shape.get("label", "")) for shape in annotations]
    if len(annotations) != len(SKELETONS) or set(supplied_names) != set(expected_names):
        raise FacialTaxonomyError("Annotation phải có đúng một skeleton cho từng label trong bảy nhóm.")
    if len(set(supplied_names)) != len(supplied_names):
        raise FacialTaxonomyError("Annotation có skeleton label bị trùng.")
    shapes: list[dict[str, Any]] = []
    by_name = {str(shape.get("label", "")): shape for shape in annotations}
    for spec in SKELETONS:
        name = spec["name"]
        shape = by_name.get(name)
        if not shape or shape.get("type") != "skeleton":
            raise FacialTaxonomyError(f"Thiếu skeleton annotation `{name}`.")
        points = list(shape.get("points") or [])
        expected = list(range(int(spec["start"]), int(spec["start"]) + int(spec["count"])))
        if [point.get("id") for point in points] != expected:
            raise FacialTaxonomyError(f"`{name}` phải có đúng Point ID {expected[0]}–{expected[-1]} theo thứ tự.")
        elements = []
        for point in points:
            state = point.get("state")
            if state not in POINT_STATES:
                raise FacialTaxonomyError(f"Point {point.get('id')} thiếu trạng thái hợp lệ.")
            try:
                x, y = float(point["x"]), float(point["y"])
            except (KeyError, TypeError, ValueError) as exc:
                raise FacialTaxonomyError(f"Point {point.get('id')} thiếu tọa độ hợp lệ.") from exc
            if not math.isfinite(x) or not math.isfinite(y):
                raise FacialTaxonomyError(f"Point {point.get('id')} có tọa độ không hữu hạn.")
            child = taxonomy[name]["children"][int(point["id"])]
            flags = point_state_flags(str(state))
            elements.append({
                "type": "points", "frame": int(frame), "label_id": int(child["id"]),
                "group": 0, "source": "manual", "occluded": flags["occluded"],
                "outside": flags["outside"], "z_order": 0, "rotation": 0,
                "points": [x, y],
                "attributes": [], "elements": [],
            })
        shapes.append({
            "type": "skeleton", "frame": int(frame),
            "label_id": int(taxonomy[name]["parent"]["id"]),
            "group": int(group), "source": "manual", "occluded": False,
            "outside": False, "z_order": 0, "rotation": 0, "points": [],
            "attributes": [], "elements": elements,
        })
    return shapes


def next_cvat_group(annotations: Mapping[str, Any]) -> int:
    """Allocate a positive object-group ID not currently used by the Job."""
    used = [int(shape.get("group", 0) or 0) for shape in annotations.get("shapes", [])]
    return max([0, *used]) + 1


def _same_shape_payload(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for field in ("type", "frame", "label_id", "group", "points", "occluded", "outside", "rotation"):
        if actual.get(field) != expected.get(field):
            return False

    def element_signature(shape: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        return sorted(
            (
                int(element.get("label_id", -1)),
                tuple(round(float(value), 5) for value in element.get("points", [])),
                bool(element.get("outside", False)),
                bool(element.get("occluded", False)),
            )
            for element in shape.get("elements", [])
        )

    return element_signature(actual) == element_signature(expected)


def save_facial_shapes_to_cvat(
    client: CVATClient,
    job_id: int,
    shapes: Sequence[Mapping[str, Any]],
    *,
    frame: int,
    taxonomy: Mapping[str, Mapping[str, Any]],
    group: int | None = None,
) -> dict[str, int]:
    """Create or update exactly one seven-skeleton face group and verify read-back.

    A group with no matching CVAT shapes is created. A complete matching group
    is updated in place. A partial group is treated as a conflict; unrelated
    shapes, tags, tracks and frames are never replaced.
    """
    job_id, frame = int(job_id), int(frame)
    current = client.get_job_annotations(job_id)
    if group is None:
        group = next_cvat_group(current)
    group = int(group)
    expected = [dict(shape) for shape in shapes]
    if len(expected) != len(SKELETONS) or any(
        shape.get("frame") != frame or shape.get("group") != group
        for shape in expected
    ):
        raise FacialTaxonomyError("Payload CVAT phải có đúng 7 skeleton cho frame và group đang chọn.")
    expected_labels = {int(taxonomy[spec["name"]]["parent"]["id"]) for spec in SKELETONS}
    matching = [
        shape for shape in current.get("shapes", [])
        if int(shape.get("frame", -1)) == frame
        and int(shape.get("group", 0) or 0) == group
        and int(shape.get("label_id", -1)) in expected_labels
        and shape.get("type") == "skeleton"
    ]
    by_label = {int(shape.get("label_id", -1)): shape for shape in matching}
    if len(matching) not in (0, len(expected_labels)) or len(by_label) != len(matching):
        raise CVATAPIError("Nhóm face trên frame có annotation CVAT chưa đầy đủ hoặc bị trùng; tải lại để xử lý conflict.")
    if not matching:
        if any(
            int(shape.get("frame", -1)) == frame and int(shape.get("group", 0) or 0) == group
            for shape in current.get("shapes", [])
        ):
            raise CVATAPIError("CVAT group ID đã được dùng trên frame này. Tải lại frame và thử lại để tránh gộp nhãn.")
        client.append_job_shapes(job_id, [dict(shape) for shape in expected])
    else:
        refreshed_before_update = client.get_job_annotations(job_id)
        current_ids = {int(shape.get("id", -1)): shape for shape in current.get("shapes", []) if shape.get("id") is not None}
        latest_ids = {int(shape.get("id", -1)): shape for shape in refreshed_before_update.get("shapes", []) if shape.get("id") is not None}
        if any(
            int(shape.get("id", -1)) not in latest_ids
            or int(shape.get("id", -1)) not in current_ids
            or CVATClient._shape_fingerprint(shape) != CVATClient._shape_fingerprint(latest_ids[int(shape["id"])])
            for shape in matching
        ):
            raise CVATAPIError("Skeleton CVAT đã di chuyển trong lúc chuẩn bị lưu. Tải lại Job và kiểm tra conflict.")
        updates = []
        for shape in expected:
            label_id = int(shape["label_id"])
            existing = by_label.get(label_id)
            if existing is None or existing.get("id") is None:
                raise CVATAPIError("Không tìm thấy ID của đủ 7 skeleton CVAT để cập nhật an toàn.")
            updates.append({**dict(shape), "id": int(existing["id"])})
        client.patch_job_annotations(job_id, {
            "version": 0, "tags": [], "shapes": updates, "tracks": [], "intervals": [],
        }, action="update")
    refreshed = client.get_job_annotations(job_id)
    actual_group = [
        shape for shape in refreshed.get("shapes", [])
        if int(shape.get("frame", -1)) == frame
        and int(shape.get("group", 0) or 0) == group
        and int(shape.get("label_id", -1)) in expected_labels
        and shape.get("type") == "skeleton"
    ]
    actual_by_label = {int(shape["label_id"]): shape for shape in actual_group}
    if len(actual_by_label) != len(expected_labels) or any(
        label_id not in actual_by_label or not _same_shape_payload(actual_by_label[label_id], shape)
        for shape in expected for label_id in [int(shape["label_id"])]
    ):
        raise CVATAPIError("Đã gửi annotation nhưng đọc lại từ CVAT không khớp; hãy tải lại Job trước khi thử lại.")
    return {"group": group, "added": len(expected) if not matching else 0, "updated": len(expected) if matching else 0}
