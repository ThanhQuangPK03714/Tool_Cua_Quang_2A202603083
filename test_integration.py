"""
Integration test: đọc Task 4 từ CVAT local (hard_panoptic, 2 frames).
"""
import sys
import os
sys.path.insert(0, ".")

from src.env_loader import load_dotenv
from src.cvat_client import CVATClient

load_dotenv()
username = os.getenv("CVAT_USERNAME", "")
password = os.getenv("CVAT_PASSWORD", "")
if not username or not password:
    raise RuntimeError("Set CVAT_USERNAME and CVAT_PASSWORD in .env before running this script.")

client = CVATClient(
    base_url=os.getenv("CVAT_SERVER_URL", "http://localhost:8080"),
    username=username,
    password=password,
)
client.login()

# Load Task 4
ctx = client.load_task_context(4)

print("=== TASK 4 INTEGRATION TEST ===")
print(f"Task name:        {ctx['task']['name']}")
print(f"Segmentation:     {ctx['segmentation_type']}")
print(f"Task ID:          {ctx['task_id']}")
print(f"Job ID:           {ctx['job_id']}")
print(f"Image count:      {ctx['image_count']}")
print(f"Frame size:       {ctx['frame_width']}x{ctx['frame_height']}")
print(f"Job state:        {ctx['job']['state'] if ctx['job'] else 'N/A'}")
print(f"Labels:           {len(ctx['labels'])} labels")
print(f"  Names:          {ctx['label_names']}")
print(f"  IDs:            {ctx['label_ids']}")

# Verify image data via API
frames = ctx['frames']
print(f"\n=== FRAMES ({len(frames)}) ===")
for f in frames:
    print(f"  {f['name']} - {f['width']}x{f['height']} (frame #{f['number']})")

# Verify empty annotations
print(f"\n=== ANNOTATIONS ===")
print(f"Job: {ctx['job_id']}")
ann = ctx['annotations']
print(f"Shapes (annotation count): {len(ann.get('shapes', []))}")
print(f"State: {ann.get('state', 'N/A')}")

print("\n=== INTEGRATION TEST PASSED ===")
