"""Safe edits for AI-assisted facial annotations."""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence


def merge_prediction_preserving_reviewed(
    existing: Sequence[Mapping[str, Any]],
    proposed: Sequence[Mapping[str, Any]],
    face_instance_id: str,
) -> list[dict[str, Any]]:
    """Fill missing points and refresh pending AI points without replacing reviewed work."""
    result = copy.deepcopy(list(existing))
    for suggestion in proposed:
        label = str(suggestion.get("label", ""))
        target = next(
            (
                shape for shape in result
                if shape.get("label") == label
                and str(shape.get("face_instance_id") or "face-01") == str(face_instance_id)
            ),
            None,
        )
        if target is None:
            target = copy.deepcopy(dict(suggestion))
            target["face_instance_id"] = str(face_instance_id)
            result.append(target)
            continue
        current_by_id = {int(point["id"]): point for point in target.get("points", []) if "id" in point}
        for point in suggestion.get("points", []):
            point_id = int(point["id"])
            current = current_by_id.get(point_id)
            protected = current is not None and (
                str(current.get("source", "manual")).casefold() in {"manual", "imported"}
                or str(current.get("review_status", "pending")).casefold() in {"accepted", "saved"}
            )
            if current is None or not protected:
                current_by_id[point_id] = copy.deepcopy(point)
        target["points"] = [current_by_id[key] for key in sorted(current_by_id)]
        target["face_instance_id"] = str(face_instance_id)
    return result
