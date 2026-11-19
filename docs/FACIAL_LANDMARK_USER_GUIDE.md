# Facial Landmark-50 User Guide

## Canvas local: manual first, AI optional

1. Start the app, choose **Canvas local**, then **Facial Landmark-50**.
2. Upload the original image. The original dimensions are used for every saved point.
3. In **Canvas tool**, choose **Chọn vùng mặt**, then drag a rectangle around a face. Draw another rectangle to add another local Face Instance.
4. Select the Face Instance. Choose **Zoom vùng mặt** to work on its crop; the canvas converts clicks and drags back to the full source-image coordinates.
5. Choose **Đặt landmark**, select the skeleton and point state, and click to place the next missing Point ID. Select a point in the table to delete it; the ID stays missing and can be placed again. Dragging, table edits, and point deletion mark the point for review.
6. Use the Canvas **Undo/Redo** buttons for point/ROI edits. The JSON draft stores the 7 skeleton groups, source image metadata, face ROIs, point state, source and review status.
7. Use **Chất lượng ảnh và tiền xử lý** to inspect brightness, contrast, blur/noise signals and optional CLAHE/gamma/bilateral previews. Enhanced variants keep the original size, so clicks remain in source-image coordinates.
8. Click **AI gợi ý điểm cho Face Instance** only when you want a proposal. It tries MediaPipe on the image or selected face ROI, plus conservative size/preprocessing fallbacks. It keeps manual and accepted points. If it fails, continue placing points manually.
9. Review validation and export the local annotation JSON. Local drafts do not write to CVAT.

Grayscale appearance is only a visual tag; the app does not assert that the camera is infrared. The enhancement preview does not prove that a variant produces more accurate landmarks.

## Connected CVAT Job

1. Connect to the Task/Job with the seven configured skeleton parents and matching Point ID children.
2. Resolve taxonomy preflight errors. SVG edges are checked when available; missing SVG produces a warning rather than blocking manual canvas use.
3. Choose a frame. Optionally select **Chọn vùng mặt** and draw a manual ROI; use **Zoom vùng mặt** for point placement.
4. Click **AI tự gán 50 điểm cho frame** to request a suggestion. A failed face detection shows a message and leaves manual drawing available. AI refreshes pending AI-generated points and preserves manual/accepted points.
5. Review coordinates and point visibility. Manual edits are marked `needs_review`; use **Accept All** after review.
6. Confirm the exact Job/frame and click **Save to CVAT**. The app validates taxonomy, writes only the selected group's skeleton data, and reads back the result. This live flow was not verified against a server during the current development run.

The connected workspace still handles one Face Instance per frame. Local Canvas supports multiple instances in its JSON drafts, but this is not converted to multiple CVAT face groups yet.

## Sample images

Copy the six original files to `tests/fixtures/facial_landmark/driver_monitoring/` using the names listed in that directory's README. Do not replace them with generated images. Until the originals are available, real-image inference, enhancement and manual-canvas results remain unverified.

## Model and state limits

- MediaPipe returns a dense mesh, not the task's exact 50-point annotations. The contour mapping is a pending-review proposal and has no accuracy benchmark against reviewed ground truth.
- MediaPipe has no per-point confidence or guideline Visible/Occluded state. Do not infer confidence values or mark fully hidden points Visible without evidence.
- SCRFD/RetinaFace, RTMPose/MMPose and FAN are not installed. Current AI fallback uses MediaPipe retries only.
- Rotation/mirror coordinate transforms are tested as helpers; the current UI does not offer arbitrary view rotation or automatic rotated inference.
