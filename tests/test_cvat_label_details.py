from __future__ import annotations

from types import SimpleNamespace

from src.cvat_client import CVATClient


def test_get_labels_fetches_skeleton_detail_when_task_summary_omits_svg():
    client = object.__new__(CVATClient)
    calls = []

    def request(method, path):
        calls.append((method, path))
        return SimpleNamespace(json=lambda: {
            "id": 7,
            "name": "longmaytrai",
            "type": "skeleton",
            "svg": "<svg>configured graph</svg>",
            "sublabels": [{"id": 70, "name": "point_0"}],
        })

    client._request = request
    labels = client.get_labels(12, task={"labels": [
        {"id": 7, "name": "longmaytrai", "type": "skeleton"},
        {"id": 8, "name": "person", "type": "polygon"},
    ]})

    assert labels[0]["svg"] == "<svg>configured graph</svg>"
    assert labels[0]["sublabels"] == [{"id": 70, "name": "point_0"}]
    assert labels[1]["name"] == "person"
    assert calls == [("GET", "/api/labels/7")]
