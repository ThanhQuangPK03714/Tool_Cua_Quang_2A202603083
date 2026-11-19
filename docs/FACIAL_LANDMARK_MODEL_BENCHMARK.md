# Facial Landmark Model Benchmark

Report date: 2026-11-19

Status: **Preprocessing and retry pipeline implemented; no benchmark was run on the six requested originals.** The inline previews are visible in chat, but the image files are not present in this workspace. No accuracy or inference success is claimed for them.

## Current model availability

| Candidate | Native output | Runtime/checkpoint | Mapping | In-cabin benchmark |
|---|---|---|---|---|
| MediaPipe Face Landmarker | 478-point mesh with face detection | MediaPipe 1.2.3 and `models/face_landmarker.task` are available in the Python 3.12 runtime | Existing guideline contour mapping; unverified against reviewed Facial-50 ground truth | Not run on the supplied images |
| RTMPose/MMPose face model | Model-specific face keypoints | MMPose/MMCV/checkpoint unavailable | Not implemented | Unavailable |
| SCRFD / RetinaFace | Face boxes and limited detector keypoints | Not installed | Detector only; no 50-point output mapping | Unavailable |
| FAN / face-alignment | 68-point schema | Package/checkpoint unavailable | Not implemented | Unavailable |

The available MediaPipe runner now tries the original image first, then modest image upscales for small inputs, quality-selected CLAHE/gamma/denoise variants, and overlapping tiles as a last resort. A manual ROI can be cropped from the original and run through the same path. All points are mapped back to source-image coordinates. The first successful candidate is used; a brighter preview is not treated as proof of better landmarks. Suggestions remain review-pending and no per-point confidence is fabricated.

## Evidence and limits

- `python -c "import mediapipe"` succeeds in the Python 3.12 runtime; the FaceLandmarker API is present.
- Unit tests cover preprocessing, grayscale appearance handling, ROI clamp/crop, coordinate rotation inverses, and annotation merge behavior.
- No inference was run on the six requested image files because those originals are not on disk.
- The mapping still creates the project's 50-point proposal from selected mesh contours. It has no reviewed-ground-truth benchmark, so do not treat its output as verified labels.
- MediaPipe does not provide the guideline's per-point Visible/Occluded states or per-landmark confidence. Human review remains required.

## Metrics required before model selection

After the originals and reviewed 50-point labels are available, report detection recall by condition, face-size and pose bins, per-region NME with a declared normalization distance, PCK with declared thresholds, mapping-failure rate, partial-visibility behavior, CPU/GPU latency, and failure examples. Split by subject/session where identifiers exist. Do not report accuracy without reviewed ground truth.

## Blocking inputs

- The six original sample images under `tests/fixtures/facial_landmark/driver_monitoring/`.
- Reviewer-approved Facial Landmark-50 ground truth for accuracy metrics.
- The live CVAT Task taxonomy and an authenticated Job for save/read-back validation.
