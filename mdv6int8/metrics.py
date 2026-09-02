"""
The metrics the issue table is built on.

Headline (camera-trap-critical), computed with pure numpy — no heavy deps:
  * per-class recall @ operating threshold τ  (instance-level, IoU-matched)
  * animal false-negative RATE                (image-level: GT has an animal,
    model produced no animal detection ≥ τ — "we reported empty, there was one")

mAP@.5 and mAP@.5:.95 are added by eval.py via pycocotools when installed; they
are secondary here precisely because INT8 can hold mAP while dropping the
low-confidence animal recall that actually matters.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _greedy_matched(pred_boxes, pred_scores, gt_boxes, iou_thr) -> int:
    """One-to-one greedy match, preds high→low score. Returns #GT matched."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0
    order = np.argsort(-pred_scores)
    ious = iou_matrix(pred_boxes[order], gt_boxes)
    taken = np.zeros(len(gt_boxes), dtype=bool)
    matched = 0
    for i in range(ious.shape[0]):
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thr and not taken[j]:
            taken[j] = True
            matched += 1
    return matched


class RecallAccumulator:
    """Accumulates per-class recall @ τ and image-level animal FN-rate."""

    def __init__(self, class_names, animal_name="animal", iou_thr=0.5):
        self.names = list(class_names)
        self.animal = animal_name
        self.iou_thr = iou_thr
        self.gt_total = defaultdict(int)      # per class: #GT instances
        self.gt_matched = defaultdict(int)    # per class: #GT matched by a pred≥τ
        self.img_with_animal = 0
        self.img_animal_missed = 0

    def add_image(self, pred_boxes, pred_scores, pred_classes,
                  gt_boxes, gt_classes):
        """All predictions passed in are already thresholded at τ."""
        pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
        pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
        pred_classes = np.asarray(pred_classes).reshape(-1)
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
        gt_classes = np.asarray(gt_classes).reshape(-1)

        for c, name in enumerate(self.names):
            gm = gt_boxes[gt_classes == c]
            self.gt_total[name] += len(gm)
            pm = pred_classes == c
            self.gt_matched[name] += _greedy_matched(
                pred_boxes[pm], pred_scores[pm], gm, self.iou_thr)

        # image-level animal false negative
        a = self.names.index(self.animal)
        if np.any(gt_classes == a):
            self.img_with_animal += 1
            if not np.any(pred_classes == a):
                self.img_animal_missed += 1

    def summary(self) -> dict:
        recall = {
            name: (self.gt_matched[name] / self.gt_total[name]) if self.gt_total[name] else None
            for name in self.names
        }
        fn_rate = (self.img_animal_missed / self.img_with_animal) if self.img_with_animal else None
        return {
            "recall_at_tau": recall,
            "gt_counts": dict(self.gt_total),
            "animal_fn_rate": fn_rate,
            "images_with_animal": self.img_with_animal,
        }
