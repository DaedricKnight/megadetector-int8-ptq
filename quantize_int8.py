#!/usr/bin/env python3
"""
Step 2 — static INT8 post-training quantization of the FP32 ONNX.

Defaults chosen for the Android target:
  * QDQ format          — what ORT's NNAPI / QNN execution providers consume.
  * per-channel weights — the accuracy-preserving default for conv detectors.
  * QInt8 weights, QUInt8 activations — the broadly-supported ARM combo
    (flip activations to QInt8 as an ablation row).
  * Entropy calibration — MinMax as the ablation.

Calibration images MUST be preprocessed exactly like eval (letterbox, /255,
RGB, NCHW) — hence the shared mdv6int8.preprocess. They must also be DISJOINT
from the eval set.

Usage:
  python quantize_int8.py --fp32 models/mdv6_v10c_fp32.onnx \
      --out models/mdv6_v10c_int8.onnx \
      --calib-glob "calib/*.jpg" --calib-limit 200
"""
import argparse
import glob
import os

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, CalibrationMethod,
    QuantType, QuantFormat,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from mdv6int8.preprocess import preprocess


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, image_paths, input_name, imgsz=1280):
        self.paths = list(image_paths)
        self.input_name = input_name
        self.imgsz = imgsz
        self._it = iter(self.paths)

    def get_next(self):
        p = next(self._it, None)
        if p is None:
            return None
        img = cv2.imread(p)
        if img is None:
            return self.get_next()
        x, _, _ = preprocess(img, self.imgsz)
        return {self.input_name: x}

    def rewind(self):
        self._it = iter(self.paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-glob", required=True, help='e.g. "calib/*.jpg"')
    ap.add_argument("--calib-limit", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--activation", choices=["quint8", "qint8"], default="quint8")
    ap.add_argument("--calib-method", choices=["entropy", "minmax", "percentile"],
                    default="entropy")
    ap.add_argument("--no-per-channel", action="store_true")
    # Detectors die if the baked-in postprocess (TopK / Sigmoid / box-decode) is
    # quantized — restrict to the heavy compute layers and keep the head float.
    ap.add_argument("--op-types", default="Conv,MatMul",
                    help='comma list, or "all" to quantize every supported op')
    args = ap.parse_args()

    paths = sorted(glob.glob(args.calib_glob))[: args.calib_limit]
    if not paths:
        raise SystemExit(f"No calibration images matched {args.calib_glob!r}")
    print(f"Calibrating on {len(paths)} images")

    input_name = ort.InferenceSession(
        args.fp32, providers=["CPUExecutionProvider"]
    ).get_inputs()[0].name

    # ORT strongly recommends this shape-inference + optimization pass before
    # static quantization; skipping it degrades quality and can fail on some ops.
    prep = args.fp32.replace(".onnx", "_prep.onnx")
    quant_pre_process(args.fp32, prep)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    quantize_static(
        prep,
        args.out,
        ImageCalibrationReader(paths, input_name, args.imgsz),
        quant_format=QuantFormat.QDQ,
        per_channel=not args.no_per_channel,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8 if args.activation == "qint8" else QuantType.QUInt8,
        op_types_to_quantize=(
            [] if args.op_types.lower() == "all" else args.op_types.split(",")
        ),
        calibrate_method={
            "entropy": CalibrationMethod.Entropy,
            "minmax": CalibrationMethod.MinMax,
            "percentile": CalibrationMethod.Percentile,
        }[args.calib_method],
    )
    os.path.exists(prep) and os.remove(prep)

    fp32_mb = os.path.getsize(args.fp32) / 1e6
    int8_mb = os.path.getsize(args.out) / 1e6
    print(f"✅ INT8 ONNX -> {args.out}")
    print(f"   size {fp32_mb:.1f} MB -> {int8_mb:.1f} MB  ({int8_mb / fp32_mb:.0%})")


if __name__ == "__main__":
    main()
