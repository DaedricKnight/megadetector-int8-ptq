#!/usr/bin/env python3
"""
Step 5 — the accuracy/cost frontier over input resolution × precision.

INT8 is usually presented as *the* lever for putting a detector on edge hardware.
It isn't the only one, and it may not be the biggest: cost scales with pixel
count, so halving the input side is a 4× saving against INT8's ~2×, and the two
compose. What nobody has published for MegaDetector is where those trade-offs
actually land — which is what this sweeps.

For each resolution it exports FP32, quantizes to INT8 with calibration at that
same resolution (calibration preprocessed at one size and inference at another
would put the activation ranges in the wrong place), evaluates both, and records
accuracy, cost and recall bucketed by object size.

The size buckets are the point of the exercise. MDV6-yolov10-c is trained at 640
(`train_args.imgsz`) while Pytorch-Wildlife infers at 1280, and the reason that
is not simply a bug is that upscaling helps small objects and hurts large ones —
so the best resolution should depend on what you are looking for, not on the
model alone. This produces the evidence either way.

Usage:
  python sweep_resolution.py                      # 640/960/1280/1600, FP32+INT8
  python sweep_resolution.py --sizes 640 1280     # a subset
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, log):
    """Run a pipeline step, echoing progress; raise with context if it fails."""
    print(f"    $ {' '.join(cmd[1:4])} …", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    log.write(f"\n$ {' '.join(cmd)}\n{r.stdout}\n{r.stderr}\n")
    if r.returncode != 0:
        raise RuntimeError(f"failed: {' '.join(cmd)}\n{r.stderr[-1500:]}")
    return r.stdout


def eval_json_path(onnx):
    return os.path.splitext(onnx)[0] + "_eval.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 960, 1280, 1600],
                    help="input sides to sweep; YOLO strides need multiples of 32")
    ap.add_argument("--pw-version", default="MDV6-yolov10-c")
    ap.add_argument("--images", default="val_nacti")
    ap.add_argument("--gt", default="val_nacti/instances.json")
    ap.add_argument("--calib-glob", default="calib/*.jpg")
    # Entropy histograms over 1600x1600 activations are memory-hungry and buy
    # nothing here; min-max is stable across the whole sweep, which matters more
    # than squeezing the last point out of any single row.
    ap.add_argument("--calib-limit", type=int, default=60)
    ap.add_argument("--calib-method", default="minmax")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--out", default="results/sweep_resolution.json")
    args = ap.parse_args()

    for s in args.sizes:
        if s % 32:
            sys.exit(f"--sizes must be multiples of 32; {s} is not")

    os.makedirs("results", exist_ok=True)
    os.makedirs("models/sweep", exist_ok=True)
    rows, t_start = [], time.time()

    with open("results/sweep_resolution.log", "w") as log:
        for size in args.sizes:
            print(f"\n=== {size}×{size} ===", flush=True)
            fp32 = f"models/sweep/mdv6_v10c_fp32_{size}.onnx"
            int8 = f"models/sweep/mdv6_v10c_int8_{size}.onnx"

            if not os.path.exists(fp32):
                run([sys.executable, "export_onnx.py", "--pw-version", args.pw_version,
                     "--imgsz", str(size), "--out", fp32], log)
            if not os.path.exists(int8):
                run([sys.executable, "quantize_int8.py", "--fp32", fp32, "--out", int8,
                     "--imgsz", str(size), "--calib-glob", args.calib_glob,
                     "--calib-limit", str(args.calib_limit),
                     "--calib-method", args.calib_method], log)

            for precision, onnx in (("FP32", fp32), ("INT8", int8)):
                run([sys.executable, "eval.py", "--onnx", onnx, "--imgsz", str(size),
                     "--images", args.images, "--gt", args.gt,
                     "--tau", str(args.tau)], log)
                d = json.load(open(eval_json_path(onnx)))
                row = {
                    "imgsz": size, "precision": precision,
                    "size_mb": d["size_mb"], "latency_ms": d["latency_ms"],
                    "map_50": d["map"]["map_50"], "map_50_95": d["map"]["map_50_95"],
                    "per_class_ap": d["map"]["per_class_ap_50_95"],
                    "recall": d["summary"]["recall_at_tau"],
                    "animal_fn_rate": d["summary"]["animal_fn_rate"],
                }
                rows.append(row)
                print(f"    {precision}: mAP@.5 {row['map_50']:.3f}  "
                      f"animal {row['recall']['animal']:.3f}  "
                      f"{row['latency_ms']:.0f} ms  {row['size_mb']:.1f} MB", flush=True)

            # Where the loss lands by object size — run once per resolution with
            # both precisions so FP32 and INT8 share the identical matcher.
            out = run([sys.executable, "analyze_by_size.py", "--gt", args.gt,
                       "--images", args.images, "--imgsz", str(size),
                       "--models", fp32, int8, "--tau", str(args.tau)], log)
            buckets = []
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 6 and parts[-1].endswith("%") and "(" in line:
                    buckets.append({
                        "bucket": " ".join(parts[:-4]), "n": int(parts[-4]),
                        "fp32": float(parts[-3]), "int8": float(parts[-2]),
                        "rel_delta_pct": float(parts[-1].rstrip("%")),
                    })
            for r in rows[-2:]:
                r["by_size"] = buckets

    json.dump({"pw_version": args.pw_version, "gt": args.gt, "tau": args.tau,
               "calib": {"glob": args.calib_glob, "limit": args.calib_limit,
                         "method": args.calib_method},
               "rows": rows},
              open(args.out, "w"), indent=2)
    print(f"\n✅ {len(rows)} rows -> {args.out}  ({(time.time()-t_start)/60:.1f} min)")
    print("   full stdout/stderr in results/sweep_resolution.log")


if __name__ == "__main__":
    main()
