# Facial Landmark Implementation Report

Report date: 2026-11-19

Status: **Partially completed.** Priority 1 manual fallback and core image-quality/preprocessing support are implemented. The requested six-file real-image evaluation and several model integrations are blocked or not implemented; see the sample report and benchmark.

## Completed

- Kept the seven guideline groups and Point IDs unchanged.
- Added `src/facial_landmark/preprocessing/` with image-quality metrics, grayscale-appearance tags, CLAHE, gamma correction, bilateral/NL-means/Gaussian denoising helpers, sharpening, and dimension-preserving variants.
- Added image-space coordinate helpers for 0/90/180/270-degree rotations, inverse point transforms, horizontal mirror transforms, padded/clamped face crops, and ROI validation.
- Added quality display and Original/CLAHE/Gamma/Denoised preview controls to the local Facial Landmark-50 workspace. Enhanced images keep the source dimensions; saved landmark coordinates remain in original-image coordinates.
- Added manual Face Instance ROI drawing, ROI zoom, multiple local instances, instance-specific skeleton progress, point deletion with stable IDs, drag/edit state, undo/redo, JSON persistence, and import/export of face regions.
- The connected CVAT Job workspace can also draw and zoom a manual face ROI. AI inference accepts that crop; the annotation canvas remains usable if detection fails.
- MediaPipe retries the original, modest upscales, quality-selected preprocessing variants, and overlapping tiles. A manual ROI runs on a padded crop; output coordinates are mapped back to the original frame.
- AI merge refreshes pending AI points while retaining manual/imported/accepted/saved points. Failed inference leaves existing work in place and displays a manual-continuation message.
- Added `face_instance_id` support in local draft validation and persistence. Multiple local Face Instances keep independent 50-point groups.
- Added structured missing-sample instructions under `tests/fixtures/facial_landmark/driver_monitoring/` without creating substitute images.
- Updated user, audit, benchmark, and sample-test reports to distinguish implemented behavior from unavailable evidence.

## Partially completed

- MediaPipe is the only installed landmark model. The current mapping proposes a full guideline-ordered 50 points when it receives a mesh; partial-mesh mapping is not implemented.
- Multiple Face Instances work in Canvas local. The connected CVAT workspace remains a single-face editor because its current CVAT serializer saves one seven-skeleton group per frame.
- Rotation and mirror inverse transforms have unit tests. The UI does not yet rotate the displayed view or run inference on rotated candidates.
- Enhancement functions include NL-means and sharpening, but the UI exposes CLAHE, gamma, and bilateral denoising controls only.
- Canvas v2 starts and the app's Canvas local route renders. A real image upload and drag interaction could not be visually driven because the sample files are unavailable to the workspace and the Agent API does not accept file-uploader actions.
- Existing CVAT conversion and read-back code has mock-client tests. No live CVAT Task/Job was connected during this work.

## Not implemented

- SCRFD, RetinaFace, YOLO-face, RTMPose/MMPose, FAN, an inference registry, and per-model mapping configuration.
- Per-point confidence, automatic occlusion classification, verified partial 50-point prediction, custom model training, active learning, and temporal video tracking.
- Camera undistortion or automatic 90/180/270-degree inference. No camera calibration is available, so no distortion correction is guessed.

## Blocked by environment/data

- The six named PNG originals are not accessible under `/mnt/data` or the Windows workspace. Four inline previews appeared in chat, but they are not readable as image files from this process. Samples 5 and 6 are not present as files either.
- No reviewed Facial Landmark-50 ground truth was supplied. No NME/PCK or accuracy claim is made.
- No authenticated live CVAT Job was available for actual Task taxonomy, write, or read-back verification.

## Verification performed

- `python -m pytest tests -q`: 40 passed, 5 skipped.
- Streamlit Agent API: headless app health `ok`; app reached `ready`; navigation to Canvas local / Facial Landmark-50 rendered without a script exception.
- App console was checked after reruns; the repeated component-registration warning was fixed by caching/versioning the inline Canvas v2 registrations.
- The manual Canvas could not be exercised against the provided sample images until their original files are placed in the fixture folder.

## Files added or updated

- `src/facial_landmark/preprocessing/` — image assessment and enhancement helpers.
- `src/facial_landmark/coordinates.py` — rotation, mirror, crop, and ROI transforms.
- `src/facial_landmark/annotation_state.py` — merge AI suggestions without replacing reviewed work.
- `src/facial_landmark/mp_face_landmarker.py` — preprocessing, scale/tile retries, manual ROI and inverse coordinate offsets.
- `src/guideline_annotation.py` and `app.py` — multi-instance manual canvas, persistence, preview, connected ROI, and review merge.
- `tests/test_facial_preprocessing.py` and focused facial guideline/model tests.
- `docs/DRIVER_MONITORING_IMAGE_TEST_REPORT.md` and refreshed Facial Landmark reports.
