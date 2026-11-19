# Facial Landmark Source Audit

Audit date: 2026-11-18

## Scope and evidence

The repository contains the Streamlit source, unit tests, model wrappers and a pasted Vietnamese Facial Landmark-50 guideline from the user. No DOCX/PDF guideline or sample face images are present in the repository. This audit uses the pasted guideline and the current source tree; it does not claim to have inspected missing illustrations or a live CVAT task.

## Current architecture

- `app.py` is a Streamlit application. The CVAT workspace and the independent Canvas local workspace are separate top-level paths.
- The custom annotation canvases use Streamlit Custom Components v2 with inline JavaScript. Canvas click positions are converted to source image coordinates.
- `src/guideline_annotation.py` contains the seven group names, point ranges, open/closed topology, visibility states, local JSON persistence, and geometry checks.
- Facial Canvas local loads user-uploaded images and stores JSON beneath `outputs/local_annotations/facial/`. It does not write to CVAT.
- The connected CVAT workspace loads Task/Job metadata and images through `src/cvat_client.py`. A Facial Landmark adapter strictly preflights skeleton parents/children and explicit IDs, checks SVG topology when available, writes one seven-skeleton group and verifies read-back.
- Existing inference paths produce segmentation masks. `show_prediction_review` displays the empty-mask warning after inference. The seven Facial Landmark labels are not segmentation classes.
- `src/annotation_processor.py` has a generic model-keypoint-to-CVAT-skeleton converter, but there is no face keypoint model, face-specific model adapter, or verified 106/478-to-50 mapping.
- The previous fix recognizes the exact seven-label taxonomy and stops segmentation inference with an explicit explanation.

## Environment observed

- App runtime: Python 3.12.10 in `.venv-sam3`.
- Present: Streamlit, PyTorch, OpenCV and MediaPipe Face Landmarker with its `.task` model asset.
- Absent: MMPose, MMCV, MMDetection, face-alignment and ONNX Runtime.
- The repository includes SAM 3 and YOLO segmentation dependencies/checkpoints, but those models do not output the required Facial Landmark-50 points.
- No authenticated CVAT server/task was available in this workspace, so the active Task label structure, CVAT version, real-frame write and read-back could not be inspected.

## Guideline contract confirmed from pasted text

| CVAT group label | Point IDs | Count | Topology |
|---|---:|---:|---|
| `longmaytrai` | 0–4 | 5 | open |
| `longmayphai` | 5–9 | 5 | open |
| `songmui` | 10–13 | 4 | open |
| `mattrai` | 14–21 | 8 | closed |
| `matphai` | 22–29 | 8 | closed |
| `moingoai` | 30–41 | 12 | closed |
| `moitrong` | 42–49 | 8 | closed |

The guideline says left/right are named by the visible side of the face in the image, not anatomical mirroring. It defines Visible/Occluded/Outside per point, allows coincident points for closed eyes and closed mouths, and requires the inner lip contour to stay inside the outer lip contour. The current app encodes these rules in the local validator and canvas.

## Gaps and risks

1. Canvas local remains a local draft mode. A separate connected Job workspace now supports direct save for a preflight-compatible one-face taxonomy.
2. The local facial schema represents one face per image. Multiple Face Instances are not implemented.
3. The current canvas supports sequential point placement, coordinate/state editing and drag adjustment, but point deletion, per-face instance management, zoom/pan controls and review/source metadata are incomplete.
4. CVAT skeletons are parent labels containing point sublabels. The seven parent labels, child counts and explicit child IDs must pass preflight. SVG topology is checked when the API returns it; otherwise the app warns and uses the guideline topology for canvas annotation.
5. MediaPipe produces a dense face mesh, not this exact 50-point schema. The app maps its official eye, eyebrow, nose and lip contours into a review-pending proposal; accuracy against Facial Landmark-50 ground truth is not benchmarked.
6. One cached cabin-frame smoke inference produced 50 mapped points and passed geometry checks. No reviewed ground truth was supplied, so NME/PCK accuracy cannot be verified.

## Reusable components

- `src/guideline_annotation.py` taxonomy, visibility-state flags, validation and atomic local JSON save/load.
- `app.py` CVAT image loading, Streamlit UI, Canvas v2 component and manual point editing.
- `src/cvat_client.py` PAT/session authentication, Job annotations read/write, append deduplication and read-back after writes.
- `src/annotation_processor.py` CVAT skeleton serialization pattern for child point elements.

## Safe implementation decisions

- Do not send Facial Landmark tasks through YOLO/SAM/semantic mask inference.
- MediaPipe mapping v1 uses explicit official feature contours and guideline-defined order/fractions; keep every generated point review-pending until evaluated against trusted ground truth.
- Do not write to a CVAT Task until its skeleton parent/child labels and point ordering pass preflight.
- Keep the current local canvas and JSON format usable; CVAT-aware save is a separate workflow.
- AI suggestions do not replace point-state review or accuracy validation.

## Official references checked

- CVAT skeleton annotation and task-configuration limitations: https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/skeletons/
- CVAT Job annotation API operations: https://docs.cvat.ai/docs/api_sdk/sdk/reference/apis/jobs-api/
- MediaPipe Face Landmarker outputs dense 3D face landmarks; these are not the task's 50 points: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- MMPose project and model support: https://github.com/open-mmlab/mmpose
