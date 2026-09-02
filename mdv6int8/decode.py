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


def decode_raw(output: np.ndarray, conf_thres: float, nc: int):
    """[1, N, 4+nc] with xywh boxes + per-class scores -> xyxy detections.
    NMS-free (v10). If your export needs NMS, add it here."""
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
    return xyxy, scores, classes
