package com.megadetector.bench

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import java.io.File
import java.nio.FloatBuffer

/**
 * Sustained-load throughput, the number a field deployment actually gets.
 *
 * A single warm median says what a phone does for two seconds on a bench. A
 * camera-trap unit runs for hours in a sealed enclosure with no airflow, where
 * the SoC heats up, the governor caps frequency, and throughput settles far
 * below the headline figure. This harness runs inference back-to-back
 * indefinitely and reports how throughput decays: per-minute fps and median
 * latency, thermal-throttling status, battery drain and temperature.
 *
 * There is no fixed duration on purpose. A remote farm session can be cut off
 * at 25 minutes while a local device runs an hour, so the run streams results
 * continuously and any screenshot is a valid datapoint for its elapsed time.
 *
 * Two honest limits, surfaced in the UI rather than buried:
 *  - Like [MdLatencyBenchmark] this times inference on one fixed input tensor.
 *    A real pipeline adds JPEG decode and letterboxing, so field throughput is
 *    lower than this and the thermal load somewhat higher.
 *  - Energy per frame is only meaningful on battery. Plugged in — which is
 *    always the case on a remote farm, and usually on a USB-attached phone —
 *    the charge counter is dominated by charging, so those figures are reported
 *    as unavailable instead of as a plausible-looking wrong number.
 */
class SustainedBenchmark(
    private val context: Context,
    private val assetModel: String = "mdv6_v10c_int8.onnx",
    private val imgsz: Int = 1280,
    private val threads: Int = 4,
) {
    private val env = OrtEnvironment.getEnvironment()
    private val modelPath: String by lazy { copyAsset(assetModel) }
    private val input = FloatArray(3 * imgsz * imgsz) { 0.5f }
    private val shape = longArrayOf(1, 3, imgsz.toLong(), imgsz.toLong())

    @Volatile private var running = false

    /** One aggregated minute of the run. */
    data class Minute(
        val index: Int,
        val frames: Int,
        val medianMs: Double,
        val p90Ms: Double,
        val thermal: String,
        val batteryPct: Int,
        val batteryTempC: Double,
        val chargeUAh: Long,
    )

    data class Snapshot(
        val elapsedMs: Long,
        val frames: Int,
        val fpsNow: Double,
        val minutes: List<Minute>,
        val firstKMedianMs: Double,   // median of minute 1 — the fresh baseline
        val lastKMedianMs: Double,    // median of the latest full minute
        val degradationPct: Double,   // latest vs minute 1, + means slower
        val thermalNow: String,
        val thermalWorst: String,
        val thermalStart: String,
        /** True when the device was already warm at start, so minute 1 is not a
         * cold baseline and the degradation figure understates the real drop. */
        val startedHot: Boolean,
        val batteryStartPct: Int,
        val batteryNowPct: Int,
        val batteryTempStartC: Double,
        val batteryTempNowC: Double,
        val charging: Boolean,
        val mAhPer1000: Double?,      // null when charging — not measurable
        val provider: String,
    )

    fun stop() { running = false }

    /**
     * Runs until [stop]. [onUpdate] is called about once a second on the calling
     * thread with a fresh snapshot; drive the UI from it.
     */
    fun run(onUpdate: (Snapshot) -> Unit) {
        running = true
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val session = env.createSession(modelPath, opts)
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape)
        val feed = mapOf(session.inputNames.first() to tensor)

        val all = ArrayList<Double>(8192)
        val minutes = ArrayList<Minute>()
        var minuteLatencies = ArrayList<Double>(512)

        val bStart = battery()
        val thermalAtStart = thermalStatus()
        var worstThermal = thermalAtStart
        val t0 = System.nanoTime()
        var lastEmit = t0
        var minuteStart = t0
        var framesAtMinuteStart = 0

        try {
            // Warm up outside the measurement: the session's first runs allocate
            // arenas and JIT-warm kernels, which is not what we're studying here.
            repeat(4) { session.run(feed).close() }

            while (running) {
                val s = System.nanoTime()
                session.run(feed).use { }
                val ms = (System.nanoTime() - s) / 1e6
                all.add(ms)
                minuteLatencies.add(ms)

                val now = System.nanoTime()
                worstThermal = maxOf(worstThermal, thermalStatus())

                if (now - minuteStart >= 60_000_000_000L) {
                    val b = battery()
                    minutes.add(
                        Minute(
                            index = minutes.size + 1,
                            frames = all.size - framesAtMinuteStart,
                            medianMs = percentile(minuteLatencies, 50),
                            p90Ms = percentile(minuteLatencies, 90),
                            thermal = THERMAL[thermalStatus().coerceIn(THERMAL.indices)],
                            batteryPct = b.pct,
                            batteryTempC = b.tempC,
                            chargeUAh = b.chargeUAh,
                        )
                    )
                    minuteLatencies = ArrayList(512)
                    minuteStart = now
                    framesAtMinuteStart = all.size
                }

                if (now - lastEmit >= 1_000_000_000L) {
                    lastEmit = now
                    onUpdate(snapshot(all, minutes, t0, now, bStart, worstThermal, thermalAtStart))
                }
            }
        } finally {
            tensor.close()
            session.close()
        }
        onUpdate(snapshot(all, minutes, t0, System.nanoTime(), bStart, worstThermal,
            thermalAtStart))
    }

    private fun snapshot(
        all: List<Double>, minutes: List<Minute>, t0: Long, now: Long,
        bStart: Battery, worstThermal: Int, thermalAtStart: Int,
    ): Snapshot {
        val elapsedMs = (now - t0) / 1_000_000
        val b = battery()
        // Degradation is measured between whole minutes, not between fixed frame
        // counts: a slow device would need 20+ minutes to accumulate 2000 frames,
        // and on a farm session capped at 25 the headline number would never
        // appear. Minute 1 is the fresh-device baseline; the latest full minute
        // is what a unit running that long actually gets.
        val firstK = minutes.firstOrNull()?.medianMs ?: Double.NaN
        val lastK = if (minutes.size >= 2) minutes.last().medianMs else Double.NaN
        val degradation =
            if (firstK.isNaN() || lastK.isNaN()) Double.NaN else (lastK - firstK) / firstK * 100.0

        // Charge counter falls as the device discharges; while plugged in it is
        // being topped up at the same time, so the delta says nothing about the
        // model's energy cost.
        val charging = b.charging || bStart.charging
        val usedUAh = (bStart.chargeUAh - b.chargeUAh).toDouble()
        val mAhPer1000 = if (charging || all.isEmpty() || usedUAh <= 0) null
        else usedUAh / 1000.0 / all.size * 1000.0

        return Snapshot(
            elapsedMs = elapsedMs,
            frames = all.size,
            fpsNow = if (minutes.isEmpty()) all.size * 1000.0 / elapsedMs.coerceAtLeast(1)
            else minutes.last().frames / 60.0,
            minutes = minutes,
            firstKMedianMs = firstK,
            lastKMedianMs = lastK,
            degradationPct = degradation,
            thermalNow = THERMAL[thermalStatus().coerceIn(THERMAL.indices)],
            thermalWorst = THERMAL[worstThermal.coerceIn(THERMAL.indices)],
            thermalStart = THERMAL[thermalAtStart.coerceIn(THERMAL.indices)],
            // Battery temperature lags the SoC but is the only skin-side reading
            // available without root; >35C at rest means the previous run's heat
            // has not dissipated.
            startedHot = thermalAtStart > 0 ||
                (!bStart.tempC.isNaN() && bStart.tempC > 35.0),
            batteryStartPct = bStart.pct,
            batteryNowPct = b.pct,
            batteryTempStartC = bStart.tempC,
            batteryTempNowC = b.tempC,
            charging = charging,
            mAhPer1000 = mAhPer1000,
            provider = "ORT CPU ($threads threads)",
        )
    }

    private data class Battery(
        val pct: Int, val tempC: Double, val chargeUAh: Long, val charging: Boolean,
    )

    private fun battery(): Battery {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val sticky = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val plugged = sticky?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0
        val tempTenths = sticky?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
        return Battery(
            pct = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY),
            tempC = if (tempTenths >= 0) tempTenths / 10.0 else Double.NaN,
            chargeUAh = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER),
            charging = plugged != 0,
        )
    }

    /** 0..6, or 0 when the API predates Q. */
    private fun thermalStatus(): Int =
        if (Build.VERSION.SDK_INT >= 29)
            (context.getSystemService(Context.POWER_SERVICE) as PowerManager).currentThermalStatus
        else 0

    private fun percentile(v: List<Double>, p: Int): Double {
        if (v.isEmpty()) return Double.NaN
        val s = v.sorted()
        return s[((s.size - 1) * p / 100).coerceIn(s.indices)]
    }

    private fun copyAsset(name: String): String {
        val out = File(context.cacheDir, name)
        context.assets.open(name).use { i -> out.outputStream().use { o -> i.copyTo(o) } }
        return out.absolutePath
    }

    companion object {
        private val THERMAL = arrayOf(
            "NONE", "LIGHT", "MODERATE", "SEVERE", "CRITICAL", "EMERGENCY", "SHUTDOWN")

        fun toCsv(s: Snapshot, device: String): String = buildString {
            appendLine("# device,$device")
            appendLine("# provider,${s.provider}")
            appendLine("# frames,${s.frames}")
            appendLine("# elapsed_s,${s.elapsedMs / 1000}")
            appendLine("# minute1_median_ms,${"%.1f".format(s.firstKMedianMs)}")
            appendLine("# last_minute_median_ms,${"%.1f".format(s.lastKMedianMs)}")
            appendLine("# degradation_pct,${"%.1f".format(s.degradationPct)}")
            appendLine("# thermal_start,${s.thermalStart}")
            appendLine("# thermal_worst,${s.thermalWorst}")
            appendLine("# started_hot,${s.startedHot}")
            appendLine("# charging,${s.charging}")
            appendLine("# mah_per_1000,${s.mAhPer1000?.let { "%.1f".format(it) } ?: "n/a (charging)"}")
            appendLine("minute,frames,fps,median_ms,p90_ms,thermal,battery_pct,battery_temp_c,charge_uah")
            for (m in s.minutes) {
                appendLine(
                    "${m.index},${m.frames},${"%.2f".format(m.frames / 60.0)}," +
                        "${"%.1f".format(m.medianMs)},${"%.1f".format(m.p90Ms)},${m.thermal}," +
                        "${m.batteryPct},${"%.1f".format(m.batteryTempC)},${m.chargeUAh}"
                )
            }
        }
    }
}
