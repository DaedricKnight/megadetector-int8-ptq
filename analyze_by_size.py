#!/usr/bin/env python3
"""
Step 4 — where does INT8 actually hurt? Recall bucketed by ground-truth object
size, FP32 vs INT8.

Aggregate mAP hides this: quantization erodes *small-object* recall far more
than large-object recall, and small distant subjects at dawn/dusk are exactly
the camera-trap case that matters. Buckets are fractions of frame area, not
COCO's absolute px² — camera-trap images vary from 1 MP to 4 MP.

Also reports, per bucket, how misses split between "no detection at all" and
"detected but IoU < 0.5". That second class is a localization/annotation
convention mismatch rather than a detection failure — on datasets whose large
boxes are drawn to a different convention than MegaDetector's it dominates the
large bucket, and reading it as a model failure would be wrong.

Usage:
  python analyze_by_size.py --gt val_nacti/instances.json --images val_nacti \
      --models models/mdv6_v10c_fp32.onnx models/mdv6_v10c_int8.onnx
"""
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort

from mdv6int8.preprocess import preprocess, unletterbox_boxes
from mdv6int8.decode import decode_v10_topk, decode_raw, decode_rtdetr
from mdv6int8.metrics import iou_matrix

BUCKETS = [("tiny (<0.5%)", 0.0, 0.005), ("small (0.5-2%)", 0.005, 0.02),
           ("medium (2-10%)", 0.02, 0.10), ("large (>10%)", 0.10, 1.01)]


def bucket_of(frac):
    return next(n for n, lo, hi in BUCKETS if lo <= frac < hi)


def evaluate(onnx, gt, images_dir, cat2idx, tau, imgsz, iou_thr,
             decoder='topk', nc=3, nms_iou=None):
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    imgs = {i["id"]: i for i in gt["images"]}
    by_img = defaultdict(list)
    for a in gt["annotations"]:
        by_img[a["image_id"]].append(a)

    hit, total, missed_undetected = defaultdict(int), defaultdict(int), defaultdict(int)
    for img_id, anns in by_img.items():
        meta = imgs[img_id]
        img = cv2.imread(os.path.join(images_dir, meta["file_name"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        W, H = meta.get("width") or w, meta.get("height") or h

        x, r, (dw, dh) = preprocess(img, imgsz)
        raw = sess.run(None, {in_name: x})[0]
        if decoder == "rtdetr":
            boxes, _, classes = decode_rtdetr(raw, tau, imgsz)
        elif decoder == "raw":
            boxes, _, classes = decode_raw(raw, tau, nc, nms_iou)
        else:
            boxes, _, classes = decode_v10_topk(raw, tau)
        boxes = unletterbox_boxes(boxes, r, dw, dh)

        for a in anns:
            gx, gy, gw, gh = a["bbox"]
            b = bucket_of((gw * gh) / (W * H))
            total[b] += 1
            same = [i for i, c in enumerate(classes) if int(c) == cat2idx[a["category_id"]]]
            best = (iou_matrix(np.array([[gx, gy, gx + gw, gy + gh]], np.float32),
                               boxes[same]).max() if same else 0.0)
            if best >= iou_thr:
                hit[b] += 1
            elif best < 0.01:
                missed_undetected[b] += 1        # nothing there vs. badly placed
    return hit, total, missed_undetected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--models", nargs="+", required=True,
                    help="first one is the baseline the deltas are taken against")
    ap.add_argument("--names", default="animal,person,vehicle")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--decoder", choices=["topk", "raw", "rtdetr"], default="topk")
    ap.add_argument("--nms-iou", type=float, default=None)
    args = ap.parse_args()

    gt = json.load(open(args.gt))
    names = args.names.split(",")
    catid_to_name = {c["id"]: c["name"] for c in gt["categories"]}
    cat2idx = {cid: names.index(n) for cid, n in catid_to_name.items() if n in names}

    results = {}
    for m in args.models:
        print(f"running {os.path.basename(m)} …")
        results[m] = evaluate(m, gt, args.images, cat2idx, args.tau, args.imgsz,
                              args.iou, args.decoder, len(names), args.nms_iou)

    base = args.models[0]
    hdr = f"{'GT size bucket':18s} {'n':>4s} " + " ".join(
        f"{os.path.basename(m).replace('mdv6_v10c_', '').replace('.onnx', ''):>8s}"
        for m in args.models) + f" {'rel. Δ':>9s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, _, _ in BUCKETS:
        n = results[base][1][name]
        if not n:
            continue
        recalls = [results[m][0][name] / n for m in args.models]
        rel = (recalls[-1] - recalls[0]) / recalls[0] * 100 if recalls[0] else float("nan")
        print(f"{name:18s} {n:4d} " + " ".join(f"{r:8.3f}" for r in recalls) + f" {rel:8.1f}%")

    print(f"\nbaseline miss breakdown ({os.path.basename(base)}):")
    hit, total, undet = results[base]
    for name, _, _ in BUCKETS:
        n = total[name]
        if not n:
            continue
        miss = n - hit[name]
        if not miss:
            print(f"  {name:18s} no misses")
            continue
        print(f"  {name:18s} {miss:3d} missed — {undet[name] / miss * 100:5.1f}% not detected, "
              f"{(miss - undet[name]) / miss * 100:5.1f}% detected but IoU<{args.iou}")


if __name__ == "__main__":
    main()
