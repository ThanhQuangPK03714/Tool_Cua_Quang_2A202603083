from __future__ import annotations

from copy import deepcopy

import pytest

from src.facial_landmark.cvat import (
    FacialTaxonomyError,
    facial_annotations_to_cvat_shapes,
    next_cvat_group,
    resolve_facial_taxonomy,
    save_facial_shapes_to_cvat,
)
from src.guideline_annotation import SKELETONS


def _labels():
    labels = []
    next_id = 100
    for spec in SKELETONS:
        start, count, name = spec["start"], spec["count"], spec["name"]
        children = []
        nodes = []
        edges = []
        for local_id in range(count):
            point_id = start + local_id
            child_id = next_id
            next_id += 1
            children.append({"id": child_id, "name": f"point_{point_id}", "type": "points"})
            nodes.append(f'<circle data-type="element node" data-node-id="n{point_id}" data-label-name="point_{point_id}"/>')
        pairs = [(start + i, start + i + 1) for i in range(count - 1)]
        if spec["contour"] == "closed":
            pairs.append((start, start + count - 1))
        edges = [f'<line data-type="edge" data-node-from="n{a}" data-node-to="n{b}"/>' for a, b in pairs]
        labels.append({
            "id": next_id, "name": name, "type": "skeleton", "sublabels": children,
            "svg": "<svg xmlns='http://www.w3.org/2000/svg'>" + "".join(nodes + edges) + "</svg>",
        })
        next_id += 1
    return labels


def _annotations():
    result = []
    for spec in SKELETONS:
        points = [
            {"id": point_id, "x": float(point_id + 3), "y": 12.0, "state": "Visible"}
            for point_id in range(spec["start"], spec["start"] + spec["count"])
        ]
        result.append({"type": "skeleton", "label": spec["name"], "contour": spec["contour"], "points": points})
    result[0]["points"][0]["state"] = "Outside"
    result[0]["points"][1]["state"] = "Occluded"
    return result


def test_resolve_taxonomy_and_serialize_states_and_ids():
    taxonomy = resolve_facial_taxonomy(_labels())
    shapes = facial_annotations_to_cvat_shapes(_annotations(), taxonomy, frame=8, group=3)
    assert len(shapes) == 7
    assert shapes[0]["frame"] == 8
    assert shapes[0]["group"] == 3
    assert [point["label_id"] for point in shapes[0]["elements"]] == [100, 101, 102, 103, 104]
    assert (shapes[0]["elements"][0]["outside"], shapes[0]["elements"][0]["occluded"]) == (True, False)
    assert (shapes[0]["elements"][1]["outside"], shapes[0]["elements"][1]["occluded"]) == (False, True)


def test_taxonomy_rejects_bad_child_id_or_skeleton_edges():
    labels = _labels()
    labels[0]["sublabels"][0]["name"] = "unknown"
    with pytest.raises(FacialTaxonomyError, match="Point ID"):
        resolve_facial_taxonomy(labels)

    labels = _labels()
    labels[3]["svg"] = labels[3]["svg"].replace('data-node-to="n21"', 'data-node-to="n20"', 1)
    taxonomy = resolve_facial_taxonomy(labels)
    assert taxonomy["mattrai"]["svg_verified"] is False
    assert "topology không khớp" in taxonomy["mattrai"]["topology_warning"]

    labels = _labels()
    labels[0]["svg"] = ""
    labels[0]["_svg_fetch_error"] = "CVAT trả về HTTP 403."
    taxonomy = resolve_facial_taxonomy(labels)
    assert taxonomy["longmaytrai"]["svg_verified"] is False
    assert "GET /api/labels/105" in taxonomy["longmaytrai"]["topology_warning"]


def test_taxonomy_decodes_escaped_svg_and_reports_parse_detail(caplog):
    labels = _labels()
    labels[0]["svg"] = labels[0]["svg"].replace("<", "&lt;").replace(">", "&gt;")
    assert resolve_facial_taxonomy(labels)["longmaytrai"]

    labels[0]["svg"] = "<svg><g></svg>"
    taxonomy = resolve_facial_taxonomy(labels)
    assert "XML không hợp lệ" in taxonomy["longmaytrai"]["topology_warning"]
    assert "CVAT skeleton SVG parse failed" in caplog.text


def test_taxonomy_accepts_cvat_svg_sibling_roots():
    labels = _labels()
    labels[0]["svg"] += "<svg><metadata /></svg>"
    taxonomy = resolve_facial_taxonomy(labels)
    assert taxonomy["longmaytrai"]["svg_verified"] is True


def test_taxonomy_allows_missing_svg_when_explicit_point_mapping_is_valid():
    labels = _labels()
    for label in labels:
        label.pop("svg")
    taxonomy = resolve_facial_taxonomy(labels)
    assert len(taxonomy) == 7
    assert all(not entry["svg_verified"] for entry in taxonomy.values())
    assert all(entry["topology_warning"] for entry in taxonomy.values())


class _MemoryClient:
    def __init__(self, initial=None):
        self.shapes = deepcopy(initial or [])
        self.next_id = 1
        self.calls = []

    def get_job_annotations(self, _job_id):
        return {"version": 0, "tags": [], "shapes": deepcopy(self.shapes), "tracks": [], "intervals": []}

    def append_job_shapes(self, _job_id, shapes):
        self.calls.append(("create", deepcopy(list(shapes))))
        for shape in shapes:
            saved = deepcopy(shape)
            saved["id"] = self.next_id
            self.next_id += 1
            self.shapes.append(saved)
        return {"added": len(shapes), "duplicates": 0}

    def patch_job_annotations(self, _job_id, payload, action):
        self.calls.append((action, deepcopy(payload)))
        by_id = {shape["id"]: shape for shape in self.shapes}
        for shape in payload["shapes"]:
            by_id[shape["id"]] = deepcopy(shape)
        self.shapes = list(by_id.values())


def test_save_adds_then_updates_same_group_without_touching_other_shapes():
    taxonomy = resolve_facial_taxonomy(_labels())
    client = _MemoryClient([{"id": 900, "type": "rectangle", "frame": 8, "label_id": 999, "points": [0, 0, 3, 3]}])
    existing = client.get_job_annotations(1)
    group = next_cvat_group(existing)
    shapes = facial_annotations_to_cvat_shapes(_annotations(), taxonomy, frame=8, group=group)
    created = save_facial_shapes_to_cvat(client, 1, shapes, frame=8, taxonomy=taxonomy, group=group)
    assert created == {"group": 1, "added": 7, "updated": 0}
    assert len(client.shapes) == 8

    changed = facial_annotations_to_cvat_shapes(_annotations(), taxonomy, frame=8, group=group)
    changed[0]["elements"][0]["points"] = [77.0, 88.0]
    updated = save_facial_shapes_to_cvat(client, 1, changed, frame=8, taxonomy=taxonomy, group=group)
    assert updated == {"group": 1, "added": 0, "updated": 7}
    assert len(client.shapes) == 8
    assert client.calls[-1][0] == "update"
    assert client.shapes[1]["elements"][0]["points"] == [77.0, 88.0]
