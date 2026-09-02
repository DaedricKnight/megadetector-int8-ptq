# Drop the model here

Copy the quantized model produced by `quantize_int8.py` into this folder as:

    mdv6_v10c_int8.onnx

(and/or the FP32 / FP16 exports, renaming the `assetModel` argument in
`MdLatencyBenchmark` to benchmark each precision).

The `.onnx` is large and machine-generated — it is git-ignored; keep the
export/quantize step as the source of truth, not the binary.
