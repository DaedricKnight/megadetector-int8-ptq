"""
Shared preprocessing — the SINGLE source of truth used by calibration, the
FP32/FP16 baseline and the INT8 eval alike. If calibration and evaluation
letterbox differently, the INT8 table is wrong for a reason that has nothing
to do with quantization — so everything imports letterbox/preprocess from here.

Matches Ultralytics letterbox: aspect-preserving resize, centre pad to a fixed
square with grey (114) fill, /255, RGB, NCHW float32.
"""
from __future__ import annotations
import cv2
import numpy as np


def letterbox(img_bgr: np.ndarray, imgsz: int = 1280, color=(114, 114, 114)):
    h0, w0 = img_bgr.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    nw, nh = round(w0 * r), round(h0 * r)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), color, dtype=np.uint8)
    dw, dh = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, (dw, dh)


def preprocess(img_bgr: np.ndarray, imgsz: int = 1280):
    """BGR HxWx3 uint8 -> (NCHW float32 [1,3,imgsz,imgsz], scale r, (dw, dh))."""
    lb, r, (dw, dh) = letterbox(img_bgr, imgsz)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])  # NCHW
    return x, r, (dw, dh)


def unletterbox_boxes(boxes_xyxy: np.ndarray, r: float, dw: int, dh: int) -> np.ndarray:
    """Boxes in letterboxed `imgsz` pixel space -> original image pixel space.
    Must be applied before IoU against ground truth, which is in original coords."""
    if len(boxes_xyxy) == 0:
        return boxes_xyxy.astype(np.float32)
    b = boxes_xyxy.astype(np.float32).copy()
    b[:, [0, 2]] -= dw
    b[:, [1, 3]] -= dh
    b /= r
    return b
