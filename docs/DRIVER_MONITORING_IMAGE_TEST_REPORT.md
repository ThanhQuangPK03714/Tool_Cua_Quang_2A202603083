# Driver Monitoring Image Test Report

Report date: 2026-11-19

## Result

**Real-image evaluation is blocked.** The user supplied inline previews in chat, but the original files named below are not present under `/mnt/data`, the repository, or the local Codex attachments. Inline image previews cannot be loaded by this app/test process. No substitute images were generated.

Four preview images were visible in the latest message (a color cabin view, dark grayscale/IR-like view, another color cabin view, and a very dark rotated grayscale/IR-like view). They were not available as files for pixel-level processing. The two remaining named samples were not visible as files or previews in this workspace.

## Per-sample status

Every cell below is **Not run — original file unavailable**. Descriptions are from the request or inline previews only; they are not quality-assessment or model results.

| Sample file | Described condition | Load/quality/preprocess | Face detection/landmarks/mapping | Manual ROI/canvas/50 points | Draft/CVAT |
|---|---|---|---|---|---|
| `image(20261119-031146).png` | Inline preview 1: color cabin view, driver visible at an angle | Not run | Not run | Not run on this image | Not run |
| `image(20261119-031200).png` | Inline preview 2: very dark grayscale/IR-like view, driver handling glasses | Not run | Not run | Not run on this image | Not run |
| `image(20261119-031209).png` | Inline preview 3: color cabin view, tilted camera/driver pose | Not run | Not run | Not run on this image | Not run |
| `image(20261119-031216).png` | Inline preview 4: dark rotated grayscale/IR-like view with reflective glasses | Not run | Not run | Not run on this image | Not run |
| `image(20261119-031223).png` | Description in the request: color, oblique face | Not run | Not run | Not run on this image | Not run |
| `image(20261119-031232).png` | Description in the request: dark IR-like image, hand/glasses occlusion | Not run | Not run | Not run on this image | Not run |

## What was tested without the sample images

- `python -m pytest tests -q`: 40 passed, 5 skipped.
- Preprocessing tests use small in-memory arrays for deterministic checks of channel handling, CLAHE/gamma/denoising output dimensions, ROI crop/clamping, and rotation/inverse transforms. These are unit tests, not replacements for the six real images.
- Guideline tests cover 50 IDs, multiple local Face Instances, incomplete drafts, deleting/re-adding a point ID, persistence and merge behavior.
- CVAT unit tests use a mock client; no live CVAT write/read-back was performed.
- Streamlit Agent API reported a healthy app and rendered the Canvas local Facial Landmark-50 route. The uploader is not supported by that API, so no actual sample was loaded into the canvas.

## How to unblock

Copy the six unchanged originals to `tests/fixtures/facial_landmark/driver_monitoring/`. That directory contains a README with the exact filenames. After they are available, rerun each image through original/quality/preprocessing, MediaPipe variants and manual ROI/canvas. Record detector, mesh/mapping, manual editing, draft persistence and live CVAT outcomes separately. Do not infer NME/PCK without reviewed 50-point labels.
