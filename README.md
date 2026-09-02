# MDV6-yolov10-c → ONNX → INT8 PTQ

Reproducible pipeline behind the INT8 quantization benchmark for MegaDetector V6.
Exports the compact YOLOv10 variant to ONNX, runs static INT8 post-training
quantization, and evaluates any precision against a COCO-format ground-truth set,
printing one issue-table row: **per-class recall@τ**, **animal false-negative
rate**, model size, latency, and (with pycocotools) mAP@.5 / mAP@.5:.95.

Designed for the Android target: INT8 uses **QDQ + per-channel weights**, the
format ORT's NNAPI / QNN execution providers consume.

**The measurements are in [`results/`](results/)** — 30-minute sustained-load
curves on four SoCs (Tensor G3, Snapdragon 8 Elite, Exynos 2400e, Dimensity 900)
and accuracy at both 640 and 1280. Headlines: a single latency measurement ranks
the devices *wrong* for continuous work; INT8's accuracy cost lands almost
entirely on small objects; and NNAPI is the slowest execution path on every
vendor tested. Discussion:
[microsoft/MegaDetector#34](https://github.com/microsoft/MegaDetector/issues/34).
If you intend to ship a model on a device, read [`LICENSING.md`](LICENSING.md)
first — the repositories are MIT but the default weights are not.

## What's verified vs. what you verify

`python smoke_test.py` passes in a plain onnx/onnxruntime/opencv env — it proves
the **INT8 plumbing** (ORT static QDQ, per-channel, entropy calibration → a graph
that loads and runs), the shared **letterbox/preprocess** round-trip, the two
**decoders**, and the **recall/FN metric**. It does **not** touch the real
weights or data.

Four model-specific things must be confirmed on a real run — they are the only
places this pipeline assumes anything, each flagged in code:

1. **Weights identifier** — get `MDV6-yolov10-c` from the current model card
   (microsoft/MegaDetector). `export_onnx.py` prints `model.names` on load.
2. **ONNX output layout** — run `eval.py --inspect one.jpg` once; keep the
   matching decoder (`topk` for `[1,300,6]`, `raw` for `[1,N,4+nc]`).
3. **Class-index → name mapping** — pass `--names` in the same order as
   `model.names` (default `animal,person,vehicle`).
4. **Eval split** — use the split the lab reports on, so your FP32 baseline
   reproduces their mAP (the open question in the issue).

## Install

```bash
pip install -r requirements.txt
python smoke_test.py          # optional: confirm the core runs in your env
```

## Run

```bash
# 0) build a class-balanced eval set (NACTI — the one LILA source that still
#    ships person boxes, already in MegaDetector's animal/person/vehicle schema)
python build_eval_from_lila.py --per-class 70 --out val_nacti

# 1) export; --pw-version downloads the weights via PytorchWildlife.
#    Default --imgsz 1280 matches the framework; pass 640 to match the weights.
python export_onnx.py --pw-version MDV6-yolov10-c --out models/mdv6_v10c_fp32.onnx

# 2) static INT8 PTQ — calibration images MUST be disjoint from eval
python quantize_int8.py --fp32 models/mdv6_v10c_fp32.onnx \
    --out models/mdv6_v10c_int8.onnx --calib-glob "calib/*.jpg" --calib-limit 60

# 3) evaluate each precision -> one table row apiece
python eval.py --onnx models/mdv6_v10c_fp32.onnx --images val_nacti --gt val_nacti/instances.json --tau 0.2
python eval.py --onnx models/mdv6_v10c_int8.onnx --images val_nacti --gt val_nacti/instances.json --tau 0.2

# 4) where INT8 actually hurts: recall by object size, FP32 vs INT8
python analyze_by_size.py --gt val_nacti/instances.json --images val_nacti \
    --models models/mdv6_v10c_fp32.onnx models/mdv6_v10c_int8.onnx
```

Ablations map to the columns/rows discussed: `--activation qint8`,
`--calib-method minmax`, `--no-per-channel`, and a `--tau` sweep (0.1/0.2/0.5).
On-device latency rows: run step 3 with `--providers NnapiExecutionProvider` (or
a QNN/GPU build of onnxruntime) on the phone.

## Gotchas confirmed on the first real run (MDV6-yolov10-c)

- **Resolution: the framework and the weights disagree.** PytorchWildlife's
  `MegaDetectorV6` runs at `IMAGE_SIZE=1280` and this pipeline defaults to the
  same, so the two are comparable — but every compact checkpoint records
  `train_args.imgsz: 640`. Measured on the same set, 640 gives much better
  aggregate accuracy (mAP@.5 0.689 vs 0.498) and is 4.3× cheaper, while 1280
  gives better *animal* recall (0.813 vs 0.714). Neither is simply correct;
  see [`results/`](results/#accuracy-640-vs-1280) and the open question on the
  issue. (An earlier version of this file claimed 640 "tanks recall" — that was
  written before either resolution had been measured, and is wrong.)
- **Do NOT quantize the whole graph.** Ultralytics bakes the YOLOv10 postprocess
  (TopK / sigmoid / box-decode) into the ONNX; quantizing those ops zeroes every
  score — the INT8 model detects *nothing* (peak confidence 0.00 on all images).
  Fix: quantize only the compute layers — `--op-types Conv,MatMul` (the default).
  With the head kept float, INT8 behaves sanely: strong detections survive
  (~0.9 → ~0.78), borderline animals drop below τ — the real per-class signal.
- **Calibration:** MinMax and Entropy both work once the head is excluded;
  entropy histograms at 1280² are memory-heavy, so a modest `--calib-limit`
  (~30–60) is plenty for PTQ.
- **PW export:** weights auto-download to `~/.cache/torch/hub/checkpoints/`; the
  PW wrapper exposes no `YOLO`, so `--pw-version` loads that cached `.pt` with
  Ultralytics. Class order confirmed `{0:animal, 1:person, 2:vehicle}`.

## Ground-truth format

Standard COCO detection JSON (`images`, `annotations` with `bbox` in `[x,y,w,h]`,
`categories`). Category **names** must include `animal`, `person`, `vehicle`.
LILA datasets convert to this via MegaDetector's own format tools; keep the
calibration and eval image lists in the repo so every table cell is re-runnable.

## Files

| | |
|---|---|
| `export_onnx.py` | `.pt` → ONNX (static shapes, opset 13, NMS-free; `--imgsz`) |
| `quantize_int8.py` | static INT8 PTQ (QDQ, per-channel, `Conv,MatMul` only) |
| `eval.py` | recall@τ · animal FN-rate · mAP · latency → table row |
| `analyze_by_size.py` | recall bucketed by object size, and why each miss missed |
| `build_eval_from_lila.py` | builds a class-balanced COCO eval set from LILA |
| `plot_sustained.py` | renders the sustained-load curves (light + dark) |
| `mdv6int8/preprocess.py` | letterbox — single source of truth for calib + eval |
| `mdv6int8/decode.py` | YOLOv10 output → detections (two layouts) |
| `mdv6int8/metrics.py` | IoU matcher, recall@τ, animal FN-rate |
| `smoke_test.py` | verifies the core without weights/data |
| `android-bench/` | on-device latency + sustained-load harness (ONNX Runtime) |
| [`results/`](results/) | **measurements and findings** — 30-min curves on four SoCs, 640-vs-1280 accuracy |
| [`LICENSING.md`](LICENSING.md) | **which MDV6 weights can actually be embedded** — the repos are MIT, the default weights declare AGPL-3.0 |
