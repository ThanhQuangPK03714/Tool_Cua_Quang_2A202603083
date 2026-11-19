# Facial Landmark-50 User Guide

## Connected CVAT workflow

1. Connect and load the Task/Job whose labels are the seven Facial Landmark-50 skeleton groups.
2. Select **Facial Landmark-50 · CVAT Job** in the workflow. This route bypasses all segmentation inference.
3. Resolve CVAT preflight errors. Each parent label must be a Skeleton with the guideline's point count and child names that explicitly encode Point IDs. SVG edges are checked when CVAT returns them; missing/invalid SVG produces a warning and uses the guideline topology in the app.
4. Choose a frame and click **AI tự gán 50 điểm cho frame**. MediaPipe detects the largest face and creates a review-pending proposal. If several faces are detected, confirm that the largest one is the driver.
5. Review the 50 coordinates and visibility states. Use **Accept All** after checking, confirm the exact Job/frame write, then click **Save to CVAT**. The app adds or updates the seven skeletons in one face group, preserves unrelated annotations, and reads the result back.

The current connected workspace handles one driver face per frame. It does not yet load arbitrary existing CVAT face groups into the editor or edit several faces in one frame. If SVG is unavailable through the API, the app can still save mapped child points, but CVAT may not draw connecting edges for that Task. Face Landmarker does not return per-point visibility states or confidence; AI suggestions stay pending review.

## Local manual workflow

1. Start the app with `streamlit run app.py`.
2. In the sidebar choose **Canvas local**.
3. Choose **Facial Landmark-50** and upload the source image.
4. Select one of the seven skeleton groups. Place points in the displayed Point ID order and follow the anatomical cue shown above and below the canvas.
5. Select the state for new points. Edit coordinates/states in the point table; use Shift-drag to adjust an already placed point while a group is incomplete.
6. Review the topology/validation messages and export the local annotation JSON.

The app's local mode remains a local draft workflow. To write directly to CVAT, connect and load the CVAT Task/Job, then use the connected workspace above.

## CVAT task requirements

CVAT skeleton structure must be configured when creating the task/project. The app preflight verifies parent type, child counts, Point IDs encoded in child names, and SVG connections. The current repository has no access to the user's live Job, so that Task's actual taxonomy has not been verified.

## AI status

Facial Landmark mode uses MediaPipe Face Landmarker as a pre-label proposal. Its 478 mesh points are mapped from the official eye, eyebrow, nose and lip contours into the guideline's 50 IDs. This mapping passed one smoke inference and geometry validation, but has not been measured against reviewed ground truth. All points remain review-pending until checked.
