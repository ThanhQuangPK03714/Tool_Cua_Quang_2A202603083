# Facial Landmark Implementation Report

Status: **PARTIALLY COMPLETED**

Report date: 2026-11-18

## Completed

- Audited the current Streamlit, Canvas v2, guideline, CVAT client and segmentation inference paths; see `FACIAL_LANDMARK_AUDIT.md`.
- Confirmed the seven parent group names, global ID ranges, counts and open/closed contours against the pasted guideline.
- Added per-ID anatomical placement cues to the manual canvas and prevented segmentation inference from being used on the exact Facial Landmark-50 label set.
- Added user-facing explanation that mask segmentation does not output facial skeleton/keypoints.
- Added tests for the seven-label task detector and placement cue coverage.
- Added a connected CVAT Job workspace with strict parent/child/Point-ID validation, SVG topology verification when available, per-point visibility, append-or-update by group and server read-back verification.
- Added tests for CVAT skeleton serialization, point states/IDs, topology rejection, create/update behavior and preservation of unrelated shapes.
- Added a connected CVAT Job workspace that loads the frame from CVAT, supports per-point state and coordinate review, requires acceptance of all 50 points, and writes only after a separate save confirmation.
- Added label-detail hydration for skeleton labels when the compact Task payload omits the SVG; preflight then validates the fetched skeleton graph.
- Changed missing/invalid SVG from a hard stop to a visible warning when all child Point IDs map explicitly. The app uses guideline contours and logs malformed SVG details so annotation is not blocked by the CVAT label-list response.
- Added parsing for CVAT SVG payloads containing multiple XML roots, draft validation from 0/50, and a one-click MediaPipe Face Landmarker proposal for all 50 points on the selected frame.

## Partially completed

- **Manual annotation**: the existing standalone Canvas local workflow remains separate. A connected CVAT Job workspace now loads CVAT images and saves a one-face seven-skeleton group after review. Multiple face instances, point-level deletion, complete keyboard workflows and full zoom/pan controls are not implemented.
- **CVAT**: append/update and server read-back are implemented for a preflight-compatible one-face group. Unit tests use a mock client; the user's actual Task taxonomy and server behavior remain unverified because no authenticated live CVAT Task/Job was available.

## Not implemented

- RTMPose and FAN inference.
- Benchmark-verified accuracy for the MediaPipe-to-50 mapping; only a smoke inference and guideline geometry check were performed.
- Model comparison, confidence-based review, multi-face AI, model registry, ONNX deployment and active learning.
- Multiple Face Instance editing and benchmark metrics.

## Blocked by environment/data

- No authenticated live CVAT Task/Job is available to inspect or write/read back.
- No reviewed Facial Landmark-50 ground truth is present, so mapping accuracy, NME/PCK and speed cannot be certified.
- No representative reviewed ground truth is available to certify the mapping accuracy. MediaPipe does not return per-point confidence or the guideline's Visible/Occluded state.
- The source repository contains no DOCX/PDF guideline; this implementation uses the pasted guideline text and cannot verify unseen diagrams.

## Tests observed

- `pytest tests -q`: 28 passed, 5 skipped (2026-11-18).
- `py_compile` for the changed app and guideline module passed.
- `node --check` on the embedded Canvas JavaScript passed.
- Streamlit app startup and navigation into Facial Landmark-50 passed using the local agent API.
- The full root-level pytest collection is blocked by `test_integration.py`, which raises unless CVAT credentials are configured. No credentials were read or added.

## References

- CVAT skeleton manual annotation: https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/skeletons/
- CVAT Job API: https://docs.cvat.ai/docs/api_sdk/sdk/reference/apis/jobs-api/
- MediaPipe Face Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- MMPose: https://github.com/open-mmlab/mmpose
