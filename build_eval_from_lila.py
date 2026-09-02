#!/usr/bin/env python3
"""
Build a COCO-detection eval set (val/ + instances.json) for eval.py from LILA
camera-trap bounding-box annotations.

Two sources:

  --source nacti     (default) North American Camera Trap Images. Its boxes are
                     already in MegaDetector's own schema — animal / person /
                     vehicle — so all three classes can be evaluated without any
                     remapping. 8892 boxed images: 3660 animal, 1251 person,
                     5652 vehicle boxes. Sampling is class-balanced.

  --source missouri  Missouri Camera Traps. Species labels, all remapped to
                     `animal`; no person/vehicle. Kept so the animal-only row
                     stays reproducible.

Note on person: most LILA datasets strip human images for privacy (Channel
Islands, Snapshot Serengeti/Enonkishu, Orinoquía, Idaho all do) — NACTI is the
exception that still ships them, which is why it's the default here.

All boxes of a selected image are kept, never just the boxes of the class it was
sampled for: partial ground truth would score real detections as false
positives and quietly deflate mAP.

Usage:
  python build_eval_from_lila.py --per-class 70
  python eval.py --onnx models/mdv6_v10c_int8.onnx --images val --gt val/instances.json --tau 0.2
"""
import argparse
import json
import os
import ssl
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

SOURCES = {
    "nacti": {
        "meta_url": ("https://lilawildlife.blob.core.windows.net/lila-wildlife/"
                     "nacti/nacti_20230920_bboxes.zip"),
        "meta_zip": "nacti_bboxes.zip",
        "img_base": ("https://lilawildlife.blob.core.windows.net/lila-wildlife/"
                     "nacti-unzipped"),
    },
    "missouri": {
        "meta_url": ("https://lilawildlife.blob.core.windows.net/lila-wildlife/"
                     "missouricameratraps/missouri_camera_traps_set1_1.21.json.zip"),
        "meta_zip": "meta.json.zip",
        "img_base": ("https://lilawildlife.blob.core.windows.net/lila-wildlife/"
                     "missouricameratraps/images"),
    },
}

CATEGORIES = [
    {"id": 1, "name": "animal"},
    {"id": 2, "name": "person"},
    {"id": 3, "name": "vehicle"},
]
KEEP_IDS = {1, 2, 3}          # NACTI also has 0=empty and 4=group; drop those
NAME_OF = {1: "animal", 2: "person", 3: "vehicle"}


def load_metadata(src):
    zip_path = src["meta_zip"]
    if not os.path.exists(zip_path):
        print(f"downloading metadata -> {zip_path}")
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(src["meta_url"], timeout=180, context=ctx) as r, \
                open(zip_path, "wb") as f:
            f.write(r.read())
    z = zipfile.ZipFile(zip_path)
    return json.load(z.open(z.namelist()[0]))


def collect_nacti(d, per_class):
    """Class-balanced pick: images containing each class, round-robin, deduped."""
    boxes_by_img = defaultdict(list)
    for a in d["annotations"]:
        if a.get("bbox"):
            boxes_by_img[a["image_id"]].append(a)

    # An image with a box we can't map (e.g. `group`) has incomplete GT — skip it.
    clean = {i: anns for i, anns in boxes_by_img.items()
             if all(a["category_id"] in KEEP_IDS for a in anns)}

    by_class = defaultdict(list)
    for img_id, anns in clean.items():
        for cid in {a["category_id"] for a in anns}:
            by_class[cid].append(img_id)

    chosen, seen = [], set()
    for cid in (1, 2, 3):
        pool = sorted(by_class[cid])
        took = 0
        for img_id in pool:
            if took >= per_class:
                break
            if img_id in seen:
                continue
            seen.add(img_id)
            chosen.append(img_id)
            took += 1
        print(f"  {NAME_OF[cid]:8s}: {len(pool):5d} images available, took {took}")
    return chosen, clean


def collect_missouri(d, limit):
    boxes_by_img = defaultdict(list)
    for a in d["annotations"]:
        if a.get("bbox"):
            boxes_by_img[a["image_id"]].append(
                {"bbox": a["bbox"], "category_id": 1})     # species -> animal
    ids = sorted(boxes_by_img)[:limit]
    print(f"  {len(boxes_by_img)} images have bboxes; taking {len(ids)} (all animal)")
    return ids, boxes_by_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(SOURCES), default="nacti")
    ap.add_argument("--out", default="val")
    ap.add_argument("--per-class", type=int, default=70,
                    help="nacti: images per class (animal/person/vehicle)")
    ap.add_argument("--limit", type=int, default=200, help="missouri: total images")
    ap.add_argument("--workers", type=int, default=8, help="parallel downloads")
    args = ap.parse_args()

    src = SOURCES[args.source]
    d = load_metadata(src)
    img_by_id = {im["id"]: im for im in d["images"]}

    print(f"source: {args.source}")
    if args.source == "nacti":
        chosen, anns_by_img = collect_nacti(d, args.per_class)
    else:
        chosen, anns_by_img = collect_missouri(d, args.limit)

    os.makedirs(args.out, exist_ok=True)
    ctx = ssl.create_default_context()

    def fetch(job):
        new_id, img_id = job
        meta = img_by_id.get(img_id)
        if not meta:
            return None
        local = f"img_{new_id:05d}.jpg"
        dst = os.path.join(args.out, local)
        if not (os.path.exists(dst) and os.path.getsize(dst) > 1000):
            url = src["img_base"] + "/" + urllib.parse.quote(meta["file_name"])
            try:
                with urllib.request.urlopen(url, timeout=60, context=ctx) as r, \
                        open(dst, "wb") as f:
                    f.write(r.read())
            except Exception as e:
                print(f"  skip {meta['file_name']}: {e}")
                return None
        return new_id, img_id, local, meta

    jobs = list(enumerate(chosen, start=1))
    print(f"downloading {len(jobs)} images with {args.workers} workers…")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        got = [r for r in ex.map(fetch, jobs) if r]

    images, annotations = [], []
    ann_id = 1
    for new_id, img_id, local, meta in sorted(got):
        images.append({"id": new_id, "file_name": local,
                       "width": meta.get("width", 0), "height": meta.get("height", 0)})
        for a in anns_by_img[img_id]:
            x, y, w, h = a["bbox"]
            annotations.append({
                "id": ann_id, "image_id": new_id, "category_id": a["category_id"],
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w) * float(h), "iscrowd": 0,
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    gt = os.path.join(args.out, "instances.json")
    json.dump(coco, open(gt, "w"))

    per_cat = defaultdict(int)
    for a in annotations:
        per_cat[a["category_id"]] += 1
    print(f"\n✅ {len(images)} images, {len(annotations)} boxes")
    for cid in (1, 2, 3):
        print(f"   {NAME_OF[cid]:8s}: {per_cat[cid]} boxes")
    print(f"   {gt}")
    print(f"\nrun:\n  python eval.py --onnx models/mdv6_v10c_int8.onnx "
          f"--images {args.out} --gt {gt} --tau 0.2")


if __name__ == "__main__":
    main()
