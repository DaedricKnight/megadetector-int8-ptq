"""
Decode the ONNX detector output to (boxes_xyxy, scores, classes) in the
letterboxed `imgsz` pixel space.

⚠ VERIFY per model card / export flags — the layout depends on how the model
was exported:

  * Ultralytics YOLOv10 export is NMS-free and usually emits [1, 300, 6] =
    (x1, y1, x2, y2, score, class_idx), already top-k. -> `decode_v10_topk`.
  * A "raw head" export emits [1, N, 4+nc] (xywh + per-class scores),
    needing arg-max + score threshold (still NMS-free for v10). -> `decode_raw`.

Run `eval.py --inspect IMG` once to print the real output name/shape, then keep
the matching decoder. Getting this wrong silently corrupts every metric.
"""
from __future__ import annotations
import numpy as np


def decode_v10_topk(output: np.ndarray, conf_thres: float):
    """[1, K, 6] -> (boxes[M,4] xyxy, scores[M], classes[M])."""
    out = np.asarray(output)
    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2 or out.shape[1] < 6:
        raise ValueError(
            f"decode_v10_topk expected [K,6], got {output.shape}. "
            f"Run eval.py --inspect and switch decoder (see decode.py)."
        )
    boxes, scores, classes = out[:, :4], out[:, 4], out[:, 5].astype(int)
    keep = scores >= conf_thres
    return boxes[keep], scores[keep], classes[keep]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.7):
    """Greedy per-class NMS. Returns the kept indices, highest score first.

    YOLOv9's head is not NMS-free the way YOLOv10's is: its export emits one
    prediction per anchor (8400 at 640), so without suppression every object is
    counted many times. Recall survives that, but mAP collapses — precision is
    destroyed by the duplicates.
    """
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = ((boxes[rest, 2] - boxes[rest, 0]) *
                  (boxes[rest, 3] - boxes[rest, 1]))
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[iou <= iou_thres]
    return np.array(keep, dtype=int)


def decode_rtdetr(output: np.ndarray, conf_thres: float, imgsz: int):
    """[1, 300, 6] = (cx, cy, w, h, score, class) with boxes **normalised to
    0..1** -> xyxy in `imgsz` pixel space.

    Same shape as the YOLOv10 top-k output but not the same contents, so the two
    are not interchangeable: feeding this to `decode_v10_topk` yields boxes a
    few pixels wide in the top-left corner and silently near-zero recall.
    Verified by inspecting value ranges — cx/cy/w/h all fall inside 0..1 and the
    class column tops out at exactly 2.0. DETR-style heads emit one query per
    object, so no NMS.
    """
    out = np.asarray(output)
    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2 or out.shape[1] < 6:
        raise ValueError(f"decode_rtdetr expected [K,6], got {output.shape}.")
    cx, cy, w, h = (out[:, i] * imgsz for i in range(4))
    scores, classes = out[:, 4], out[:, 5].astype(int)
    keep = scores >= conf_thres
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes[keep], scores[keep], classes[keep]


def decode_raw(output: np.ndarray, conf_thres: float, nc: int,
               nms_iou: float | None = None, pre_nms: int = 3000,
               max_det: int = 300):
    """[1, N, 4+nc] with xywh boxes + per-class scores -> xyxy detections.

    `nms_iou` applies per-class suppression — required for YOLOv9-style heads
    that emit one prediction per anchor; leave it None for NMS-free YOLOv10.
    `pre_nms` caps the candidate set by score first and `max_det` caps the
    result, matching Ultralytics' defaults; `max_det` also matches the 300 that
    the YOLOv10 and RT-DETR heads emit, so the three architectures are compared
    under the same detection budget.
    """
    out = np.asarray(output)
    if out.ndim == 3:
        out = out[0]
    if out.shape[1] == 4 + nc:            # [N, 4+nc]
        xywh, cls_scores = out[:, :4], out[:, 4:]
    elif out.shape[0] == 4 + nc:          # [4+nc, N] (transposed)
        out = out.T
        xywh, cls_scores = out[:, :4], out[:, 4:]
    else:
        raise ValueError(f"decode_raw: cannot read layout {output.shape} with nc={nc}.")
    classes = cls_scores.argmax(1)
    scores = cls_scores.max(1)
    keep = scores >= conf_thres
    xywh, scores, classes = xywh[keep], scores[keep], classes[keep]
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    if nms_iou is not None and len(xyxy):
        # Cap the candidate set before suppressing, the way Ultralytics' own
        # non_max_suppression does. At the low confidence floor used for mAP
        # nearly all 8400 anchors survive the threshold, and feeding those
        # straight into a greedy O(n^2) NMS is both extremely slow and worse:
        # thousands of scattered background boxes rarely overlap each other, so
        # they are all kept and then crowd out the real detections under COCO's
        # maxDets cap.
        if len(scores) > pre_nms:
            top = scores.argsort()[::-1][:pre_nms]
            xyxy, scores, classes = xyxy[top], scores[top], classes[top]
        kept = []
        for c in np.unique(classes):
            idx = np.flatnonzero(classes == c)
            kept.append(idx[nms(xyxy[idx], scores[idx], nms_iou)])
        order = np.concatenate(kept)
        order = order[scores[order].argsort()[::-1]][:max_det]
        xyxy, scores, classes = xyxy[order], scores[order], classes[order]
    return xyxy, scores, classes
