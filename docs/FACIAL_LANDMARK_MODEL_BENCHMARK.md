# Facial Landmark Model Benchmark

Status: **Inference smoke-tested; benchmark not run because no reviewed Facial Landmark-50 ground truth was supplied.**

MediaPipe Face Landmarker is installed in `.venv-sam3` with Google's official `face_landmarker.task` asset. A smoke inference on a cached Task 472 cabin frame detected one face and produced 50 guideline-ordered points that passed the geometry validator. This is a working pre-label path, not an accuracy benchmark. MMPose, MMCV, MMDetection, face-alignment and ONNX Runtime are not installed.

| Candidate | Native output | Face detection | 50-point mapping | Local runtime | Benchmark |
|---|---|---|---|---|---|
| RTMPose/MMPose face model | Face keypoints (model-specific schema) | Model/pipeline dependent | Not verified | Dependencies absent | Not run |
| MediaPipe Face Landmarker | Dense 478-point 3D face mesh | Included in task model | Deterministic contour mapping v1; not benchmark-verified | Local inference passed on one cached frame | No GT benchmark |
| FAN / face-alignment | 68-point schema | Pipeline dependent | Not verified | Package/checkpoint absent | Not run |

## Metrics required before model selection

The app writes MediaPipe suggestions as `review_status=pending`. MediaPipe provides the mesh but no per-point confidence or CVAT Visible/Occluded state; the adapter marks in-frame points Visible and out-of-frame points Outside, so state review remains necessary. Mapping v1 uses the official eyebrow, eye, nose and lip mesh contours and samples points at the guideline's named fractions. Its target accuracy is unverified. Use a human-reviewed benchmark split by subject where subject IDs exist. Report NME with a declared normalization distance, PCK with declared thresholds, per-region NME, face-detection recall, mapping-failure rate, CPU/GPU latency, and failure cases.

## Next blocking inputs

- Representative face images with accepted 50-point ground truth.
- The exact CVAT skeleton child labels/order for the target Task.
- Representative cabin images and reviewed 50-point ground truth to validate mapping v1.
- Verification of CVAT visibility states for occluded landmarks.
