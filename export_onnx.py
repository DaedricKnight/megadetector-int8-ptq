#!/usr/bin/env python3
"""
Step 1 — MDV6-yolov10-c (.pt) -> ONNX, fixed 640x640, static batch=1.

Static shape (dynamic=False) is deliberate: it makes the graph friendlier to
static INT8 PTQ and to the Android NNAPI/QNN execution providers downstream.

There is NO pre-quantized INT8 model to download — this pipeline produces it.
The canonical FP32 weights auto-download through PytorchWildlife when you name a
variant; there's no loose .pt on the Model Zoo page. Two entry points:

  --pw-version MDV6-yolov10-c   load via PytorchWildlife (auto-downloads the
                                canonical weights) and export the underlying
                                Ultralytics model. Preferred — no mystery URL.
  --weights path/to.pt          if you already have the .pt (e.g. the file PW
                                cached), export it directly with Ultralytics.

Usage:
  python export_onnx.py --pw-version MDV6-yolov10-c --out models/mdv6_v10c_fp32.onnx --fp16
  python export_onnx.py --weights MDV6-yolov10-c.pt  --out models/mdv6_v10c_fp32.onnx
"""
import argparse
import os


def _disable_flops_profiling():
    """Ultralytics prints a FLOPs count via thop on model load. thop 0.1.1 leaves
    hooks on modules it never registered counters for, which under recent torch
    raises `'ReLU' object has no attribute 'total_ops'` — RT-DETR trips it and
    the export dies. The number is cosmetic here, so switch it off rather than
    pinning a dependency for a log line."""
    try:
        import ultralytics.utils.torch_utils as tu
        tu.get_flops = lambda *a, **k: 0.0
        tu.get_flops_with_torch_profiler = lambda *a, **k: 0.0
    except Exception:
        pass


def _ultralytics_model_from_pw(version):
    """Trigger PytorchWildlife's canonical weight download, then load the cached
    .pt with Ultralytics — PW keeps only an AutoBackend/predictor internally
    (no YOLO wrapper with .export), so we go via the checkpoint it downloaded to
    the torch-hub cache. Also confirms the class-name order for eval/decode."""
    import torch
    from PytorchWildlife.models import detection as pw_detection
    from ultralytics import YOLO

    md = pw_detection.MegaDetectorV6(version=version)     # downloads on first run
    name = getattr(md, "MODEL_NAME", f"{version}.pt")
    print("PW CLASS_NAMES:", getattr(md, "CLASS_NAMES", "?"),
          "· PW IMAGE_SIZE:", getattr(md, "IMAGE_SIZE", "?"))

    pt = os.path.join(torch.hub.get_dir(), "checkpoints", name)
    if not os.path.exists(pt):
        hits = [p for p in (
            os.path.join(torch.hub.get_dir(), "checkpoints", name),
        ) if os.path.exists(p)]
        if not hits:
            raise SystemExit(f"Downloaded checkpoint not found (expected {pt}). "
                             "Pass it explicitly with --weights.")
        pt = hits[0]
    print("checkpoint:", pt)
    return YOLO(pt)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pw-version", help="e.g. MDV6-yolov10-c (auto-downloads via PytorchWildlife)")
    src.add_argument("--weights", help="a local MDV6 .pt checkpoint")
    ap.add_argument("--out", required=True, help="output FP32 .onnx path")
    ap.add_argument("--imgsz", type=int, default=1280)  # MegaDetector runs at 1280, NOT 640
    ap.add_argument("--opset", type=int, default=13)   # >=13 for QDQ quantization
    ap.add_argument("--fp16", action="store_true", help="also emit an FP16 copy")
    args = ap.parse_args()
    _disable_flops_profiling()

    if args.pw_version:
        model = _ultralytics_model_from_pw(args.pw_version)
    else:
        from ultralytics import YOLO
        model = YOLO(args.weights)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print("model.names:", getattr(model, "names", "?"),
          " <-- VERIFY this is the animal/person/vehicle order you pass to eval")

    exported = model.export(
        format="onnx", imgsz=args.imgsz, opset=args.opset,
        dynamic=False, simplify=True, nms=False,  # v10 is NMS-free
    )
    # Ultralytics writes next to the weights; move to the requested path.
    if os.path.abspath(exported) != os.path.abspath(args.out):
        os.replace(exported, args.out)
    print(f"✅ FP32 ONNX -> {args.out}")

    if args.fp16:
        from onnxconverter_common import float16
        import onnx
        m = onnx.load(args.out)
        m16 = float16.convert_float_to_float16(m, keep_io_types=True)
        out16 = args.out.replace(".onnx", "_fp16.onnx")
        onnx.save(m16, out16)
        print(f"✅ FP16 ONNX -> {out16}")


if __name__ == "__main__":
    main()
