"""
Run pipeline: Connect CVAT -> Read Task 4 -> Run AI models -> Show results.
"""
import sys
import io
import os
sys.path.insert(0, ".")

from src.env_loader import load_dotenv

load_dotenv()

from src.cvat_client import CVATClient
from src.semantic_segmenter import CityscapesSegformer
from src.instance_segmenter import YOLOv8Seg
from src.panoptic_segmenter import PanopticSegmenter
from src.quality_checker import QualityChecker
from src.task_detector import get_stuff_classes, get_thing_classes
from src.device_utils import get_device_status, resolve_device
import numpy as np
from PIL import Image

CVA_USERNAME = os.getenv("CVAT_USERNAME", "")
CVA_PASSWORD = os.getenv("CVAT_PASSWORD", "")
if not CVA_USERNAME or not CVA_PASSWORD:
    raise RuntimeError("Set CVAT_USERNAME and CVAT_PASSWORD in .env before running this script.")

print("=" * 50)
print("Pipeline: CVAT Task 4 (hard_panoptic)")
print("=" * 50)
device = resolve_device(os.getenv("MODEL_DEVICE", "auto"))
device_status = get_device_status()
if device.startswith("cuda"):
    print(f"Inference device: {device} · {device_status['gpu_names'][0]}")
else:
    print("Inference device: CPU")

# 1. Connect CVAT
print("\n[1/6] Connecting CVAT...")
client = CVATClient(
    base_url=os.getenv("CVAT_SERVER_URL", "http://localhost:8080"),
    username=CVA_USERNAME,
    password=CVA_PASSWORD,
)
client.login()
print("   OK - Connected")
# 2. Load Task 4
print("\n[2/6] Loading Task 4...")
ctx = client.load_task_context(4)
print(f"   Task: {ctx['task']['name']}")
print(f"   Type: {ctx['segmentation_type']}")
print(f"   Job ID: {ctx['job_id']}")
print(f"   Frames: {ctx['image_count']}")
print(f"   Labels ({len(ctx['labels'])}): {ctx['label_names']}")

frames = ctx['frames']

# 3. Download images
print("\n[3/6] Downloading images...")
images = {}
for idx, f in enumerate(frames):
    content = client.get_frame_image(4, idx, quality="original")
    img = np.array(Image.open(io.BytesIO(content)).convert("RGB"))
    name = f['name']
    images[name] = img
    print(f"   OK - {name} ({img.shape[1]}×{img.shape[0]})")


# 4. Run AI models
print("\n[4/6] Running AI models...")

# Semantic (SegFormer)
print("\n  -- Semantic (SegFormer Cityscapes) --")
semantic_model = CityscapesSegformer(device=device, input_size=(512, 512), confidence_threshold=0.25)
semantic_results = {}
for stem, img in images.items():
    label_map, conf = semantic_model.predict(img)
    semantic_results[stem] = {"label_map": label_map, "confidence": conf}
    print(f"   OK - {stem}: semantic mask ({label_map.shape[1]}×{label_map.shape[0]})")

# Instance (YOLOv8-seg)
print("\n  -- Instance (YOLOv8-seg) --")
instance_model = YOLOv8Seg(device=device, input_size=(640, 640), confidence_threshold=0.25)
instance_results = {}
for stem, img in images.items():
    instances = instance_model.predict(img)
    instance_results[stem] = {"instances": instances}
    print(f"   OK - {stem}: {len(instances)} instance(s)")
    for i, inst in enumerate(instances):
        print(f"      • {inst['class_name']} (conf={inst['confidence']:.3f}, area={inst['area']:,} px)")

# Panoptic (combined)
print("\n  -- Panoptic (combined) --")
panoptic_model = PanopticSegmenter(
    device=device,
    semantic_input_size=(512, 512),
    instance_input_size=(640, 640),
    semantic_conf=0.25,
    instance_conf=0.25,
    instance_iou=0.45,
)
panoptic_results = {}
for stem, img in images.items():
    result = panoptic_model.predict(img)
    panoptic_results[stem] = result
    print(f"   OK - {stem}: {len(result['segments'])} segments")
    for seg in result['segments']:
        t = "thing" if seg['is_thing'] else "stuff"
        print(f"      • [{t}] {seg['class_name']} (id={seg['id']}, area={seg['area']:,} px)")



# 5. Show results
print("\n[5/6] Result summary:")
for stem, pred in panoptic_results.items():
    img = images[stem]
    label_map = pred['label_map']
    print(f"\n   {stem}: {img.shape[1]}×{img.shape[0]}")
    print(f"   Labels: {ctx['label_names']}")

    unique, counts = np.unique(label_map, return_counts=True)
    for class_id, count in zip(unique, counts):
        if class_id == 255:
            nm = "Unknown"
        else:
            nm = ctx['label_names'][class_id] if class_id < len(ctx['label_names']) else f"id_{class_id}"
        print(f"      • {nm}: {count:,} px ({count / label_map.size * 100:.1f}%)")

# 6. QC
print("\n[6/6] QC check...")
stuff = get_stuff_classes(ctx['label_names'])
things = get_thing_classes(ctx['label_names'])
qc = QualityChecker(stuff_classes=stuff, thing_classes=things)

all_records = []
for stem, pred in panoptic_results.items():
    img_h, img_w = pred['image'].shape[:2]
    if ctx['segmentation_type'] == "semantic":
        label_map = pred['label_map']
        conf_map = pred['confidence']
        for class_name in ctx['label_names']:
            class_id = ctx['label_names'].index(class_name)
            mask = (label_map == class_id).astype(np.uint8)
            if mask.sum() == 0:
                continue
            avg_conf = float(conf_map[mask > 0].mean()) if mask.sum() > 0 else 0.0
            all_records.append({
                "image_stem": stem,
                "mask_id": f"{stem}_{class_name}",
                "class_name": class_name,
                "is_thing": False,
                "mask": mask,
                "confidence": avg_conf,
            })
    elif ctx['segmentation_type'] == "instance":
        for i, inst in enumerate(pred['instances']):
            all_records.append({
                "image_stem": stem,
                "mask_id": f"{stem}_{inst['class_name']}_{i}",
                "class_name": inst['class_name'],
                "is_thing": True,
                "mask": inst['mask'],
                "confidence": inst['confidence'],
            })
    elif ctx['segmentation_type'] == "panoptic":
        for seg in pred['segments']:
            all_records.append({
                "image_stem": stem,
                "mask_id": f"{stem}_{seg['class_name']}_{seg['id']}",
                "class_name": seg['class_name'],
                "is_thing": seg['is_thing'],
                "mask": (pred['panoptic_map'] == seg['id']).astype(np.uint8),
                "confidence": seg['confidence'],
            })

masks_for_qc = [
    {"id": q["mask_id"], "class_name": q["class_name"], "is_thing": q["is_thing"], "mask": q["mask"], "confidence": q["confidence"]}
    for q in all_records
]

qc_results = qc.check_multiple_masks(masks_for_qc, (img_h, img_w))

for q, r in zip(all_records, qc_results):
    mark = "✅" if r.passed else "⚠️"
    print(f"   {mark} {q['class_name']:12s} conf={q['confidence']:.3f} level={r.level:6s} {r.issues[0] if r.issues else ''}")

print("\n" + "=" * 50)
print("Pipeline COMPLETE ✓")
print("=" * 50)
print("\nResults saved to outputs/")
print("Open Streamlit at http://localhost:8501 for interactive view.")
