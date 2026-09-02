package com.megadetector.bench

import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * adb / CI runner for the latency harness — no UI needed.
 *
 *   ./gradlew :bench:connectedDebugAndroidTest
 *   adb pull /sdcard/Android/data/com.megadetector.bench/files/mdv6_latency.csv
 *
 * The model must be at bench/src/main/assets/mdv6_v10c_int8.onnx (see
 * PUT_MODEL_HERE.md). Results are logged (tag MdBench) and written to CSV.
 */
@RunWith(AndroidJUnit4::class)
class LatencyBenchmarkTest {

    @Test
    fun measureLatencyAcrossProviders() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val bench = MdLatencyBenchmark(ctx)
        // First full pass is cold (frequencies not ramped, caches empty) — the
        // per-provider warmup isn't enough on a fresh process. Discard it and
        // report the second, warm pass.
        bench.run()
        val rows = bench.run()
        val device = MdLatencyBenchmark.deviceLabel()

        Log.i("MdBench", "=== $device · model ${"%.1f".format(bench.modelSizeMb())} MB ===")
        rows.forEach {
            Log.i("MdBench", if (it.ok)
                "%-20s median %6.1f ms  p90 %6.1f ms  min %6.1f ms"
                    .format(it.provider, it.medianMs, it.p90Ms, it.minMs)
            else "%-20s UNAVAILABLE (%s)".format(it.provider, it.note))
        }

        val csv = File(ctx.getExternalFilesDir(null), "mdv6_latency.csv")
        csv.writeText(MdLatencyBenchmark.toCsv(rows, bench.modelSizeMb(), device))
        Log.i("MdBench", "csv -> ${csv.absolutePath}")

        // At minimum the plain-CPU provider must produce a number on any device.
        assertTrue("no provider produced a latency", rows.any { it.ok })
    }
}
