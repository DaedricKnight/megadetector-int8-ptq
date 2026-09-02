#!/usr/bin/env python3
"""
Build a COCO-detection eval set (val/ + instances.json) for eval.py from a LILA
camera-trap dataset's bounding-box annotations.

MegaDetector detects animal / person / vehicle, but LILA categories are species,
so every non-empty species is remapped to `animal` (id 1); person (2) / vehicle
(3) categories are declared for completeness even when a dataset has none.

Only images carrying at least one bbox are usable for IoU-matched mAP/recall —
most LILA labels are image-level and are skipped here.

Default source: Missouri Camera Traps (~956 human bboxes, all animals).
For a table spanning person/vehicle, add a road/urban set the same way.

Usage:
  python build_eval_from_lila.py --limit 200
  python eval.py --onnx models/mdv6_v10c_int8.onnx --images val --gt val/instances.json --tau 0.2
"""
import argparse
import json
import os
import ssl
import urllib.parse
import urllib.request
import zipfile

META_URL = ("https://lilawildlife.blob.core.windows.net/lila-wildlife/"
            "missouricameratraps/missouri_camera_traps_set1_1.21.json.zip")
IMG_BASE = ("https://lilawildlife.blob.core.windows.net/lila-wildlife/"
            "missouricameratraps/images")

CATEGORIES = [
    {"id": 1, "name": "animal"},
    {"id": 2, "name": "person"},
    {"id": 3, "name": "vehicle"},
]


def load_metadata(meta_zip="meta.json.zip"):
    if not os.path.exists(meta_zip):
        print(f"downloading metadata -> {meta_zip}")
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(META_URL, timeout=120, context=ctx) as r, open(meta_zip, "wb") as f:
            f.write(r.read())
    z = zipfile.ZipFile(meta_zip)
    return json.load(z.open(z.namelist()[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="val")
    ap.add_argument("--limit", type=int, default=200, help="max bbox-images to fetch")
    args = ap.parse_args()

    d = load_metadata()
    img_by_id = {im["id"]: im for im in d["images"]}

    # image_id -> its bbox annotations (species remapped to animal)
    boxes_by_img = {}
    for a in d["annotations"]:
        if a.get("bbox"):                       # skip image-level labels
            boxes_by_img.setdefault(a["image_id"], []).append(a["bbox"])
    bbox_image_ids = sorted(boxes_by_img)[: args.limit]
    print(f"{len(boxes_by_img)} images have bboxes; taking {len(bbox_image_ids)}")

    os.makedirs(args.out, exist_ok=True)
    ctx = ssl.create_default_context()
    images, annotations = [], []
    ann_id = 1
    kept = 0
    for new_id, img_id in enumerate(bbox_image_ids, start=1):
        meta = img_by_id.get(img_id)
        if not meta:
            continue
        fn = meta["file_name"]
        local = f"img_{new_id:05d}.jpg"
        dst = os.path.join(args.out, local)
        if not (os.path.exists(dst) and os.path.getsize(dst) > 1000):
            url = IMG_BASE + "/" + urllib.parse.quote(fn)
            try:
                with urllib.request.urlopen(url, timeout=30, context=ctx) as r, open(dst, "wb") as f:
                    f.write(r.read())
            except Exception as e:
                print(f"  skip {fn}: {e}")
                continue
        images.append({
            "id": new_id, "file_name": local,
            "width": meta.get("width", 0), "height": meta.get("height", 0),
        })
        for (x, y, w, h) in boxes_by_img[img_id]:
            annotations.append({
                "id": ann_id, "image_id": new_id, "category_id": 1,   # -> animal
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w) * float(h), "iscrowd": 0,
            })
            ann_id += 1
        kept += 1
        if kept % 40 == 0:
            print(f"  {kept} images…")

    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    gt = os.path.join(args.out, "instances.json")
    json.dump(coco, open(gt, "w"))
    print(f"\n✅ {len(images)} images, {len(annotations)} boxes (all class=animal)")
    print(f"   {gt}")
    print(f"\nrun:\n  python eval.py --onnx models/mdv6_v10c_int8.onnx "
          f"--images {args.out} --gt {gt} --tau 0.2")


if __name__ == "__main__":
    main()
