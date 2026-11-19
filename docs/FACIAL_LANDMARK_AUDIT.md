# Facial Landmark Source Audit

Audit date: 2026-11-19

## Current architecture

- `app.py` is a Streamlit app with a standalone Canvas local workflow and a connected CVAT Job workflow.
- `src/guideline_annotation.py` holds the seven group definitions, guideline point cues, geometry validation, local JSON persistence, face ROI persistence and CVAT state flags.
- `src/facial_landmark/preprocessing/` measures image signals and creates same-size CLAHE, gamma and denoising alternatives. `coordinates.py` holds crop, rotation, mirror and inverse-coordinate helpers.
- `src/facial_landmark/mp_face_landmarker.py` wraps MediaPipe Face Landmarker. It can process a manual ROI, try modest upscales and quality-selected variants, and fall back to overlapping tiles. Coordinates are translated to the original frame.
- `src/facial_landmark/annotation_state.py` merges suggestions while retaining manual/imported/reviewed points.
- Canvas v2 supports direct point placement, drag, point deletion, visibility state, manual face ROI, ROI zoom and undo/redo. Local Canvas can hold multiple Face Instances in one image; connected CVAT remains one face group per frame.
- `src/facial_landmark/cvat.py` validates CVAT skeleton parents/children and explicit Point IDs, serializes one face group, deduplicates/update-writes and reads back.

## Guideline contract

| Group | Point IDs | Count | Topology |
|---|---:|---:|---|
| `longmaytrai` | 0–4 | 5 | open |
| `longmayphai` | 5–9 | 5 | open |
| `songmui` | 10–13 | 4 | open |
| `mattrai` | 14–21 | 8 | closed |
| `matphai` | 22–29 | 8 | closed |
| `moingoai` | 30–41 | 12 | closed |
| `moitrong` | 42–49 | 8 | closed |

Left/right follows the visible side of the face in the image. The guideline defines per-point Visible/Occluded/Outside state, allows coincident points for specified closed-eye/lip cases, and requires the inner lip contour to stay within the outer contour.

## Gaps

1. Only MediaPipe is installed. SCRFD/RetinaFace, RTMPose/MMPose and FAN are not integrated; there is no model registry.
2. The MediaPipe-to-50 contour mapping has no reviewed-ground-truth benchmark. The model does not return per-point confidence or guideline occlusion states.
3. Automatic face detection and landmark prediction are coupled inside MediaPipe. There is no independent detector when the mesh model fails.
4. Partial 50-point mapping is not implemented. The merge layer can preserve reviewed points and fill missing IDs, but current MediaPipe mapping proposes a full set or no set.
5. Multiple Face Instances work in local JSON/canvas only. The connected CVAT serializer and editor remain single-face.
6. Rotation transforms are implemented and tested, but the UI does not expose view rotation or rotated inference. No lens correction is applied without calibration.
7. Four sample previews appeared inline in chat, but no original image files are accessible to the workspace. The remaining named samples are not available as files; real-image tests are blocked.
8. No authenticated live CVAT Job was available for actual Task preflight, write and read-back verification.

## Available verification

- Automated tests cover local persistence, ROI validation, coordinate transforms, preprocessing, prediction merge, Facial-50 validation and mocked CVAT serialization.
- Streamlit Agent API successfully loaded the app and selected the Canvas local Facial Landmark-50 route.
- The sample report separates these tests from unavailable per-image and live-CVAT evidence.

## Safe operating rules

- Do not send Facial Landmark tasks through mask segmentation.
- Do not invent coordinates for hidden or unmapped points.
- Keep AI points pending until reviewed; preserve manual and accepted work during refresh.
- Do not report model accuracy without reviewed ground truth.
- Do not report CVAT save success unless the live server read-back succeeds.
