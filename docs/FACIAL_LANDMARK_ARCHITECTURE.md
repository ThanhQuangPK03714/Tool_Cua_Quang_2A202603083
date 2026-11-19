# Facial Landmark Architecture

## Current architecture

The application has two different annotation contexts:

1. **Connected CVAT workspace**: loads a Job and uses CVAT metadata and frame APIs. Its inference/review path is mask-oriented.
2. **Canvas local**: uploads an image, draws the seven guideline skeleton groups in a Streamlit Components v2 canvas, and saves a local JSON draft.

The local canvas operates in original-image coordinates and has per-point CVAT-compatible visibility flags. A connected CVAT Job workspace now reuses this canvas and writes seven parent skeleton shapes after taxonomy preflight. The connected workspace currently handles one face group per frame.

## Intended module boundaries

- `guideline_annotation.py`: task schema, point IDs, topology, states, local persistence and deterministic validation.
- `facial_landmark/inference/`: model adapters returning source-coordinate face instances and native model points. No CVAT or UI calls.
- `facial_landmark/mapping/`: versioned, data-backed source-index to target-ID mappings. An unverified mapping is rejected.
- `facial_landmark/cvat/`: taxonomy preflight, coordinate conversion, skeleton payload serialization, append/update, read-back and conflict checks.
- Streamlit canvas: selection, AI preview, manual correction, state editing, face instances and explicit save state.

## CVAT data model

CVAT skeleton labels have a parent skeleton label and child point labels configured in the task. The seven guideline groups are treated as seven expected parent labels only if the connected Task reports them with `type=skeleton` and the expected child counts. The connected writer accepts children whose names explicitly encode global or local Point IDs and never infers correspondence from list order. It checks the parent's SVG graph for exact open/closed topology when available; if the API omits or returns malformed SVG, it logs a warning and uses the known guideline topology in the app. CVAT may not draw edges if the Task's actual SVG is missing or incorrect.

Each source point is stored in source image pixel coordinates. On export it becomes a CVAT skeleton element with the matching child `label_id`, one `[x, y]` coordinate, and that point's own `outside`/`occluded` flags. The parent skeleton shape and all point elements share the target CVAT frame. Existing shapes outside the current face/frame are preserved.

## Face instances

The target data model is Image → Face Instance → seven named groups → 50 keyed points. Point IDs repeat per face instance, while IDs remain globally fixed across the seven groups. A face instance has its own local/server identity and review state. Grouping and update identity must be persisted so a later save updates the same face instead of creating duplicates.

## AI model boundary

Model adapters may provide face boxes and model-native landmarks. Mapping is an independent, versioned stage. It must include evidence for every target point and must not synthesize missing points. Until an annotated validation set confirms correspondence and error, AI output is a visual aid only and cannot be accepted as Facial Landmark-50.

MediaPipe Face Landmarker outputs a dense mesh (documented as 478 3D points). The current proposal adapter follows its official eye, eyebrow, nose and lip contours, maps image-left/right and samples the guideline's stated fractions. This is a versioned proposal mapping, not a benchmark-certified model-to-50 mapping; generated points remain pending review.
