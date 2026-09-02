#!/usr/bin/env python3
"""
Environment smoke test — proves the parts that DON'T need the real weights /
LILA data actually run here: the ORT static-INT8 plumbing, preprocess, decode
and the recall/FN metric. It does NOT validate the model-specific bits
(weights, output layout, class mapping) — those are verified on a real run.

Run from the repo root:  python smoke_test.py
"""
import os
import tempfile

import cv2
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

from mdv6int8.preprocess import preprocess, unletterbox_boxes
from mdv6int8.decode import decode_v10_topk, decode_raw
from mdv6int8.metrics import RecallAccumulator


def make_tiny_onnx(path):
    W = np.random.randn(6, 3, 3, 3).astype(np.float32)
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1], kernel_shape=[3, 3]),
        helper.make_node("Relu", ["c"], ["y"]),
    ]
    g = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6, 640, 640])],
        [numpy_helper.from_array(W, name="w")],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 9
    onnx.save(m, path)


def test_preprocess_and_letterbox():
    img = np.zeros((480, 640, 3), np.uint8)
    x, r, (dw, dh) = preprocess(img, 640)
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    # round-trip a box through letterbox and back
    orig = np.array([[100, 50, 200, 150]], np.float32)
    lb = orig * r + np.array([dw, dh, dw, dh])
    back = unletterbox_boxes(lb, r, dw, dh)
    assert np.allclose(back, orig, atol=1e-3), back
    print("  preprocess + letterbox round-trip .......... ok")


def test_decoders():
    topk = np.array([[[10, 10, 20, 20, 0.9, 0], [0, 0, 5, 5, 0.1, 1]]], np.float32)
    b, s, c = decode_v10_topk(topk, 0.2)
    assert len(b) == 1 and c[0] == 0
    raw = np.zeros((1, 2, 4 + 3), np.float32)
    raw[0, 0] = [15, 15, 10, 10, 0.05, 0.8, 0.0]   # xywh + 3 class scores
    b2, s2, c2 = decode_raw(raw, 0.2, nc=3)
    assert len(b2) == 1 and c2[0] == 1 and abs(b2[0, 0] - 10) < 1e-4
    print("  decode_v10_topk + decode_raw ............... ok")


def test_metrics():
    acc = RecallAccumulator(["animal", "person", "vehicle"])
    gt = np.array([[0, 0, 10, 10]], np.float32)
    # image 1: animal perfectly detected
    acc.add_image(np.array([[0, 0, 10, 10]]), np.array([0.9]), np.array([0]), gt, np.array([0]))
    # image 2: animal present, nothing predicted -> a false negative
    acc.add_image(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)), gt, np.array([0]))
    s = acc.summary()
    assert abs(s["recall_at_tau"]["animal"] - 0.5) < 1e-6, s
    assert abs(s["animal_fn_rate"] - 0.5) < 1e-6, s
    print("  recall@τ + animal FN-rate .................. ok")


def test_int8_plumbing():
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantType, QuantFormat
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from quantize_int8 import ImageCalibrationReader

    d = tempfile.mkdtemp()
    fp32, prep, int8 = (os.path.join(d, f) for f in ("tiny.onnx", "tiny_prep.onnx", "tiny_int8.onnx"))
    make_tiny_onnx(fp32)

    calib = os.path.join(d, "calib"); os.makedirs(calib)
    paths = []
    for i in range(6):
        p = os.path.join(calib, f"{i}.jpg")
        cv2.imwrite(p, (np.random.rand(48, 64, 3) * 255).astype(np.uint8))
        paths.append(p)

    in_name = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"]).get_inputs()[0].name
    quant_pre_process(fp32, prep)
    quantize_static(
        prep, int8, ImageCalibrationReader(paths, in_name, 640),
        quant_format=QuantFormat.QDQ, per_channel=True,
        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.Entropy,
    )
    assert os.path.exists(int8)
    # the quantized graph must load and run
    sess = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])
    img = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)
    x, *_ = preprocess(img, 640)
    out = sess.run(None, {in_name: x})[0]
    assert out.shape == (1, 6, 640, 640)
    has_qdq = any("Quantize" in n.op_type for n in onnx.load(int8).graph.node)
    assert has_qdq, "expected QuantizeLinear/DequantizeLinear nodes in INT8 graph"
    fp32_mb, int8_mb = os.path.getsize(fp32) / 1e6, os.path.getsize(int8) / 1e6
    print(f"  ORT static INT8 (QDQ, per-channel, entropy) ok  "
          f"[{fp32_mb:.3f} -> {int8_mb:.3f} MB, QDQ nodes present, runs]")


if __name__ == "__main__":
    print("smoke test:")
    test_preprocess_and_letterbox()
    test_decoders()
    test_metrics()
    test_int8_plumbing()
    print("ALL GREEN — quantization core + pre/post verified in this environment.")
