# MDV6 on-device harness (Android)

Two benchmarks over the exported ONNX detector on a real phone, both using
`com.microsoft.onnxruntime:onnxruntime-android` — the same runtime as the Python
pipeline, so numbers are comparable end to end. Plain Android Views, no Compose.

1. **Latency** — inference time per execution provider. One warm median each.
2. **Sustained load** — what a field unit actually gets: throughput decay under
   continuous inference, thermal throttling, battery drain and temperature.

The app **auto-starts the sustained run on launch** (no tap: remote farm
sessions are time-capped and their taps are laggy) and holds the screen awake.
**Latency test** is a button.

## 1. Latency — providers measured

| Row | Session option |
|---|---|
| CPU (1 thread) | `setIntraOpNumThreads(1)` — single-core baseline |
| CPU (4 threads) | `setIntraOpNumThreads(4)` |
| XNNPACK (4 threads) | `addXnnpack(...)` — optimized CPU kernels |
| NNAPI | `addNnapi()` — device NPU/GPU/CPU; consumes the QDQ INT8 graph |

8 warmup + 40 timed iterations, except NNAPI (2 + 6): where the driver rejects
the quantized graph it falls back to the CPU *reference* kernels, and six
samples are enough to record a penalty that large without spending ten minutes
on it.

Each provider is attempted independently; a missing EP is reported
`unavailable` rather than failing the run. Latency is **inference-only** over a
fixed reused input (compute is data-independent), isolating model cost.

### What four SoCs showed

| SoC | CPU-1 | CPU-4 | XNNPACK | NNAPI |
|---|--:|--:|--:|--:|
| Tensor G3 (Pixel 8 Pro) | 878 | **444** | 492 | ~9500 |
| Snapdragon 8 Elite | 164 | **145** | 168 | 2736 |
| Exynos 2400e | 366 | **274** | 284 | 4883 |
| Dimensity 900 | 684 | 508 | **492** | 1009 |

**NNAPI is the slowest path on every vendor.** On Tensor/Qualcomm/Exynos the
driver rejects the QDQ graph (`input.type != TENSOR_FLOAT32`) and falls back to
`nnapi-reference` on CPU (×18–21). MediaTek's NeuroPilot accepts it but is still
×2 slower than plain CPU. Real NPU offload needs a vendor EP (ORT QNN), not
NNAPI.

## 2. Sustained load

Runs inference back-to-back **indefinitely**, aggregating per minute: fps,
median/p90 latency, thermal status (`PowerManager.getCurrentThermalStatus`),
battery level and temperature. There is deliberately no fixed duration — a farm
session may be cut at 25 minutes while a local device runs an hour, so results
stream continuously and **any screenshot is a valid datapoint for its elapsed
time**. A CSV is rewritten every minute, so an interrupted run still yields the
curve.

Headline: **median of minute 1 vs the latest full minute**. Deliberately not
"first 1000 vs last 1000 frames" — a slow device needs 20+ minutes to reach 2000
frames, and on a capped session the number would never appear. Minute-based, it
shows up from minute 2 on any device.

### Two things it refuses to fake

- **Energy.** `mAh / 1000 frames` is only meaningful on battery. Plugged in —
  always true on a remote farm, usually true on a USB-attached phone — the
  charge counter is dominated by charging, so the figure is reported as
  unavailable instead of as a plausible wrong number.
- **Cold start.** If the device is already warm at launch (thermal status above
  `NONE`, or battery > 35 °C), minute 1 isn't a fresh baseline and the
  degradation figure understates the real drop, so the run says so on screen.
  Measured on a Pixel 8 Pro: minute 1 was **289 ms** starting cool and **477 ms**
  starting warm — a 65% spread from thermal state alone.

## Setup

Standard single-module Android app. Drop the model into
`bench/src/main/assets/mdv6_v10c_int8.onnx` (see `PUT_MODEL_HERE.md`), then:

```bash
cd android-bench
./gradlew :bench:assembleDebug
adb install -r bench/build/outputs/apk/debug/bench-debug.apk
```

The APK is restricted to `arm64-v8a` (28.6 MB) so it uploads comfortably to
device farms; drop `abiFilters` in `bench/build.gradle.kts` for other ABIs.

## Run

**Sustained (auto-starts):** launch the app, let it run, screenshot whenever.

```bash
adb shell am start -n com.megadetector.bench/.MainActivity
adb pull /sdcard/Android/data/com.megadetector.bench/files/mdv6_sustained.csv
```

**Latency, headless via adb/CI:**

```bash
./gradlew :bench:connectedDebugAndroidTest
adb logcat -d -s MdBench:I
adb pull /sdcard/Android/data/com.megadetector.bench/files/mdv6_latency.csv
```

Note the instrumentation test runs the latency suite **twice** and reports the
second pass, since the first is cold. That also means it heats the device: on a
passively-cooled SoC the "warm" pass can come back throttled (Tensor G3: CPU-4
444 → ~1380 ms). For a clean latency number, run once on a cooled device and
read the CPU providers *before* the NNAPI block, which is what does the heating.

## Reading the numbers honestly

- Report **median and p90**; median is the table cell, p90 shows tail behaviour.
- **Thermals dominate.** Order matters too: a provider benchmarked after 40
  seconds of prior load is already warm, so single-shot tables carry an ordering
  artifact. This is the whole reason the sustained benchmark exists.
- **NNAPI caveat:** an unsupported INT8 QDQ op falls back to CPU *per subgraph*,
  so an "NNAPI" number can hide partial CPU execution. Check `adb logcat` for
  ORT partitioning logs before concluding anything about the NPU.
- For publication-grade clock-locked stats, port the loop to **Jetpack
  Microbenchmark** — worth citing if a maintainer pushes on methodology.

## Files

| | |
|---|---|
| `bench/.../MdLatencyBenchmark.kt` | per-provider session + warmup/measure + CSV |
| `bench/.../SustainedBenchmark.kt` | continuous run, per-minute aggregation, thermal/battery |
| `bench/.../MainActivity.kt` | auto-starts sustained; button for latency |
| `bench/.../LatencyBenchmarkTest.kt` | adb/CI runner, writes CSV + logcat |
