#!/usr/bin/env python3
"""
Step 3 — evaluate any ONNX (FP32 / FP16 / INT8) against a COCO-format GT set and
print one issue-table row: per-class recall@τ, animal FN-rate, and (if
pycocotools is installed) mAP@.5 / mAP@.5:.95, plus median latency.

GT is standard COCO detection json (images / annotations / categories). The
category NAMES must include animal, person, vehicle. Model class index → name
is given by --names (default MD order: animal,person,vehicle) — VERIFY against
model.names printed by export_onnx.py.

Usage:
  python eval.py --onnx models/mdv6_v10c_int8.onnx --images val/ \
      --gt val/instances.json --tau 0.2
  python eval.py --onnx models/mdv6_v10c_fp32.onnx --images val/ --inspect one.jpg
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

from mdv6int8.preprocess import preprocess, unletterbox_boxes
from mdv6int8.decode import decode_v10_topk, decode_raw, decode_rtdetr
from mdv6int8.metrics import RecallAccumulator


def build_session(path, providers):
    return ort.InferenceSession(path, providers=providers)


def run_one(sess, in_name, img, imgsz, decoder, conf, nc, nms_iou=None):
    x, r, (dw, dh) = preprocess(img, imgsz)
    t0 = time.perf_counter()
    out = sess.run(None, {in_name: x})[0]
    dt = (time.perf_counter() - t0) * 1000.0
    if decoder == "topk":
        boxes, scores, classes = decode_v10_topk(out, conf)
    elif decoder == "rtdetr":
        boxes, scores, classes = decode_rtdetr(out, conf, imgsz)
    else:
        boxes, scores, classes = decode_raw(out, conf, nc, nms_iou)
    boxes = unletterbox_boxes(boxes, r, dw, dh)
    return boxes, scores, classes, dt, out.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--images", required=True, help="dir with the eval images")
    ap.add_argument("--gt", help="COCO-format GT json (omit only with --inspect)")
    ap.add_argument("--names", default="animal,person,vehicle")
    ap.add_argument("--tau", type=float, default=0.2, help="operating conf threshold")
    ap.add_argument("--map-conf", type=float, default=0.001, help="low conf floor for mAP")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--decoder", choices=["topk", "raw", "rtdetr"], default="topk",
                    help="topk=YOLOv10 [1,300,6]; raw=YOLOv9 [1,4+nc,N] (use --nms-iou); "
                         "rtdetr=[1,300,6] normalised cxcywh")
    ap.add_argument("--nms-iou", type=float, default=None,
                    help="per-class NMS IoU; required for --decoder raw on YOLOv9")
    ap.add_argument("--providers", default="CPUExecutionProvider")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--inspect", metavar="IMG",
                    help="print the raw output name/shape for one image and exit")
    args = ap.parse_args()

    names = args.names.split(",")
    sess = build_session(args.onnx, args.providers.split(","))
    in_name = sess.get_inputs()[0].name

    if args.inspect:
        img = cv2.imread(args.inspect if os.path.isabs(args.inspect)
                         else os.path.join(args.images, args.inspect))
        x, *_ = preprocess(img, args.imgsz)
        outs = sess.run(None, {in_name: x})
        for o, meta in zip(outs, sess.get_outputs()):
            print(f"output '{meta.name}': shape {np.asarray(o).shape} dtype {np.asarray(o).dtype}")
        print("Pick the decoder in decode.py that matches this layout.")
        return

    coco = json.load(open(args.gt))
    catid_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    name_to_idx = {n: i for i, n in enumerate(names)}
    imgs = coco["images"]
    if args.limit:
        imgs = imgs[: args.limit]
    ann_by_img = {}
    for a in coco["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    acc = RecallAccumulator(names)
    dets_coco = []       # for optional pycocotools mAP
    latencies = []

    for im in imgs:
        path = os.path.join(args.images, im["file_name"])
        img = cv2.imread(path)
        if img is None:
            print(f"  skip unreadable {path}")
            continue

        # τ-thresholded predictions -> recall / FN accounting
        boxes, scores, classes, dt, _ = run_one(
            sess, in_name, img, args.imgsz, args.decoder, args.tau, len(names),
            args.nms_iou)
        latencies.append(dt)

        gts = ann_by_img.get(im["id"], [])
        gt_boxes = np.array([[a["bbox"][0], a["bbox"][1],
                              a["bbox"][0] + a["bbox"][2],
                              a["bbox"][1] + a["bbox"][3]] for a in gts], dtype=np.float32)
        gt_classes = np.array(
            [name_to_idx.get(catid_to_name.get(a["category_id"]), -1) for a in gts])
        acc.add_image(boxes, scores, classes, gt_boxes, gt_classes)

        # low-conf predictions -> COCO detections for mAP
        lb, sc, cl, _, _ = run_one(
            sess, in_name, img, args.imgsz, args.decoder, args.map_conf, len(names),
            args.nms_iou)
        name_to_catid = {v: k for k, v in catid_to_name.items()}
        for (x1, y1, x2, y2), s, c in zip(lb, sc, cl):
            cid = name_to_catid.get(names[c]) if 0 <= c < len(names) else None
            if cid is None:
                continue
            dets_coco.append({
                "image_id": im["id"], "category_id": cid,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            })

    s = acc.summary()
    size_mb = os.path.getsize(args.onnx) / 1e6
    lat = float(np.median(latencies)) if latencies else float("nan")

    print("\n================ RESULT ROW ================")
    print(f"model file      : {os.path.basename(args.onnx)}  ({size_mb:.1f} MB)")
    print(f"images / τ      : {len(latencies)} / {args.tau}")
    print(f"recall@τ        : " + "  ".join(
        f"{n}={('%.3f' % v) if v is not None else 'n/a'}" for n, v in s['recall_at_tau'].items()))
    print(f"animal FN-rate  : "
          f"{('%.3f' % s['animal_fn_rate']) if s['animal_fn_rate'] is not None else 'n/a'}"
          f"  ({s['images_with_animal']} imgs w/ animal)")
    print(f"latency (median): {lat:.1f} ms/img  [{args.providers}]")

    # mAP is persisted alongside recall/FN — a number that only ever reaches
    # stdout can't be audited later, and un-auditable numbers can't be published.
    mAP = {"map_50_95": None, "map_50": None, "per_class_ap_50_95": None}
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        cg = COCO(args.gt)
        cd = cg.loadRes(dets_coco) if dets_coco else None
        if cd is not None:
            e = COCOeval(cg, cd, "bbox")
            e.evaluate(); e.accumulate(); e.summarize()
            mAP["map_50_95"], mAP["map_50"] = float(e.stats[0]), float(e.stats[1])
            print(f"mAP@.5:.95 / mAP@.5 : {e.stats[0]:.3f} / {e.stats[1]:.3f}")

            # Per-class AP: precision is [T, R, K, A, M]; K is the category axis
            # in the order of cg.getCatIds(). -1 marks "class absent from GT".
            per_class = {}
            cat_ids = cg.getCatIds()
            for k, cid in enumerate(cat_ids):
                p = e.eval["precision"][:, :, k, 0, 2]
                p = p[p > -1]
                per_class[catid_to_name.get(cid, str(cid))] = (
                    float(np.mean(p)) if p.size else None)
            mAP["per_class_ap_50_95"] = per_class
            print("AP@.5:.95 / class : " + "  ".join(
                f"{n}={('%.3f' % v) if v is not None else 'n/a'}"
                for n, v in per_class.items()))
    except ImportError:
        print("mAP              : (install pycocotools for mAP@.5 / mAP@.5:.95)")

    json.dump({"summary": s, "map": mAP, "size_mb": size_mb, "latency_ms": lat,
               "tau": args.tau, "n_images": len(latencies), "gt": args.gt,
               "providers": args.providers},
              open(os.path.splitext(args.onnx)[0] + "_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
