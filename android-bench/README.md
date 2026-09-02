# MDV6 on-device latency harness (Android)

Measures inference latency of the exported ONNX detector on a real phone across
ONNX Runtime execution providers — the **on-device latency** rows of the
MegaDetector V6 INT8 issue table. Target device: Pixel 8 Pro (Tensor G3), but it
runs on any Android ≥ 8.0.

Uses `com.microsoft.onnxruntime:onnxruntime-android` — the same runtime as the
Python pipeline, so the numbers are comparable end to end. Plain Android Views,
no Compose, to keep the module trivially portable.

## Providers measured

| Row | Session option |
|---|---|
| CPU (1 thread) | `setIntraOpNumThreads(1)` — single-core baseline |
| CPU (4 threads) | `setIntraOpNumThreads(4)` |
| XNNPACK (4 threads) | `addXnnpack(...)` — optimized CPU kernels |
| NNAPI | `addNnapi()` — device NPU/GPU/CPU; consumes the QDQ INT8 graph |

Each provider is attempted independently; if an AAR build lacks XNNPACK or NNAPI,
that row is reported `unavailable` instead of failing the run. Latency is
**inference-only** over a fixed reused input (compute is data-independent), which
isolates the model cost and keeps providers comparable.

## Setup

This is a standard single-module Android app project. Because a Gradle wrapper
binary can't be checked in cleanly here, generate one once:

```bash
cd android-bench
gradle wrapper --gradle-version 8.9      # or: open the folder in Android Studio
```

Then drop the model into `bench/src/main/assets/mdv6_v10c_int8.onnx`
(see `PUT_MODEL_HERE.md`).

## Run

**Headless (recommended for the table) — via adb/CI:**

```bash
./gradlew :bench:connectedDebugAndroidTest
adb logcat -d -s MdBench:I
adb pull /sdcard/Android/data/com.megadetector.bench/files/mdv6_latency.csv
```

**Interactive:** install the app, tap **Run benchmark**, read the table on screen
(also written to the same CSV).

## Reading the numbers honestly

- Report **median and p90** over 100 iterations after 20 warmup — the harness
  already does this. Median is the table cell; p90 shows tail behaviour.
- **Thermals:** Tensor G3 throttles. Run cool, in airplane mode, screen on; take
  the best of a few runs or note the thermal state. For publication-grade rigor,
  port the loop to **Jetpack Microbenchmark**, which locks clocks and reports
  stabilized stats — cite that if a maintainer pushes on methodology.
- **NNAPI caveat:** an INT8 QDQ op unsupported by NNAPI falls back to CPU per
  subgraph, so a "NNAPI" number can hide partial CPU execution. If NNAPI ≈ CPU,
  check `adb logcat` for ORT partitioning logs before concluding "no NPU win".
- Benchmark FP32 / FP16 / INT8 by pointing `assetModel` at each export — the
  INT8-vs-FP32 latency ratio on NNAPI is the headline of this table.

## Files

| | |
|---|---|
| `bench/.../MdLatencyBenchmark.kt` | core: per-provider session + warmup/measure + CSV |
| `bench/.../MainActivity.kt` | one-button UI (plain Views) |
| `bench/.../LatencyBenchmarkTest.kt` | adb/CI runner, writes CSV + logcat |

> Not built here (no Android toolchain in the authoring env). The ORT usage
> mirrors a shipping app on the same `onnxruntime-android` runtime; the four
> model/EP assumptions are the ones to confirm on first run (XNNPACK/NNAPI
> availability in your AAR, and the NNAPI-fallback check above).
