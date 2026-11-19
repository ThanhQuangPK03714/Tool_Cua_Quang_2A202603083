"""Road-scene text prompts for SAM 3's open-vocabulary fallback."""

from __future__ import annotations


BDD100K_SAM3_PROMPTS = {
    "area/alternative": "alternative drivable road area reachable by changing lanes from the vehicle's current lane",
    "area/direct": "drivable road surface directly ahead in the vehicle's current lane",
    "area/drivable": "drivable road surface directly ahead of the vehicle",
    "lane/crosswalk": "painted pedestrian crosswalk stripes across the road",
    "lane/double white": "two parallel white lane marking lines painted on the road",
    "lane/double yellow": "two parallel yellow lane marking lines painted on the road",
    "lane/road curb": "the road-edge curb line separating the roadway from the roadside",
    "lane/single other": "a single lane marking line that is neither white nor yellow",
    "lane/single white": "a single white lane marking line painted on the road",
    "lane/single yellow": "a single yellow lane marking line painted on the road",
    "lane/double other": "two parallel lane marking lines that are neither white nor yellow",
}


def sam3_prompt_for_label(label: str) -> str:
    """Return a natural language prompt for a CVAT label."""
    normalized = " ".join(label.strip().casefold().replace("_", " ").split())
    if normalized in BDD100K_SAM3_PROMPTS:
        return BDD100K_SAM3_PROMPTS[normalized]
    concept = normalized.replace("/", " ").replace("-", " ")
    return f"{concept} in a road scene"


def is_region_label(label: str) -> bool:
    """BDD drivable-area and lane-marking labels are semantic regions."""
    normalized = label.strip().casefold()
    return normalized.startswith(("area/", "lane/"))
