package com.megadetector.bench

import android.content.ContentValues
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.io.File
import java.util.Locale

/**
 * Auto-starts the sustained-load run on launch — no tap needed, which matters on
 * a remote device farm where taps are laggy and session time is capped. The
 * summary is rendered above the per-minute table so a screenshot taken without
 * scrolling already carries the headline numbers.
 *
 * The screen is held awake for the duration: the run dies with the CPU if the
 * device sleeps. That costs some battery, which is one more reason the energy
 * figures are only claimed when running unplugged.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var output: TextView
    private lateinit var stopButton: Button
    private lateinit var latencyButton: Button
    private var sustained: SustainedBenchmark? = null
    private var busy = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        output = TextView(this).apply {
            textSize = 12f
            setPadding(24, 16, 24, 48)
            typeface = android.graphics.Typeface.MONOSPACE
            setTextIsSelectable(true)
        }
        stopButton = Button(this).apply {
            text = "Stop"
            setOnClickListener { sustained?.stop() }
        }
        latencyButton = Button(this).apply {
            text = "Latency test"
            setOnClickListener { runLatency() }
        }

        val buttons = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            val lp = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            addView(stopButton, lp)
            addView(latencyButton, lp)
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(buttons)
            addView(ScrollView(this@MainActivity).apply { addView(output) })
        }
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val b = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(0, b.top + 16, 0, b.bottom)
            insets
        }
        setContentView(root)

        output.text = "MDV6 sustained-load benchmark\n" +
            "${MdLatencyBenchmark.deviceLabel()}\n\nstarting…"
        root.post { runSustained() }
    }

    override fun onDestroy() {
        sustained?.stop()
        super.onDestroy()
    }

    private fun runSustained() {
        if (busy) return
        busy = true
        val bench = SustainedBenchmark(this)
        sustained = bench
        val device = MdLatencyBenchmark.deviceLabel()
        val csv = File(getExternalFilesDir(null), "mdv6_sustained.csv")
        var lastMinuteWritten = 0

        Thread {
            runCatching {
                bench.run { snap ->
                    val text = render(device, snap)
                    // Flush a CSV each new minute: a farm session can be cut off
                    // mid-run, and a partial file still carries the curve.
                    if (snap.minutes.size > lastMinuteWritten) {
                        lastMinuteWritten = snap.minutes.size
                        val csvText = SustainedBenchmark.toCsv(snap, device)
                        runCatching { csv.writeText(csvText) }
                        runCatching { writeToDownloads(csvText) }
                    }
                    runOnUiThread { output.text = text }
                }
            }.onFailure { e ->
                runOnUiThread {
                    output.text = "ERROR: ${e.message}\n\nIs mdv6_v10c_int8.onnx in assets?"
                }
            }
            busy = false
        }.start()
    }

    /**
     * Mirror the CSV into public Downloads. On a remote device farm there is no
     * adb, and a farm's file browser can't reach `Android/data` under scoped
     * storage — Downloads it can, which is the difference between exact numbers
     * and transcribing a 25-row table out of a screenshot.
     */
    private var downloadsUri: android.net.Uri? = null

    private fun writeToDownloads(text: String) {
        if (Build.VERSION.SDK_INT < 29) return          // pre-scoped-storage: adb is fine
        val name = "mdv6_sustained.csv"
        val uri = downloadsUri ?: run {
            // Drop a stale file from a previous run so the browser shows one entry.
            contentResolver.delete(
                MediaStore.Downloads.EXTERNAL_CONTENT_URI,
                "${MediaStore.Downloads.DISPLAY_NAME} = ?", arrayOf(name))
            contentResolver.insert(
                MediaStore.Downloads.EXTERNAL_CONTENT_URI,
                ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, name)
                    put(MediaStore.Downloads.MIME_TYPE, "text/csv")
                })?.also { downloadsUri = it } ?: return
        }
        contentResolver.openOutputStream(uri, "wt")?.use { it.write(text.toByteArray()) }
    }

    private fun runLatency() {
        if (busy) return
        busy = true
        sustained?.stop()
        output.text = "${MdLatencyBenchmark.deviceLabel()}\n\nsingle-shot latency, 1–4 min…"
        Thread {
            val text = runCatching {
                val b = MdLatencyBenchmark(this)
                val rows = b.run()
                buildString {
                    appendLine(MdLatencyBenchmark.deviceLabel())
                    appendLine("INT8 ${"%.1f".format(b.modelSizeMb())} MB · 1280×1280\n")
                    appendLine("%-20s %9s %8s".format("provider", "median", "p90"))
                    appendLine("-".repeat(42))
                    rows.forEach {
                        appendLine(
                            if (it.ok) "%-20s %7.1fms %6.1fms".format(it.provider, it.medianMs, it.p90Ms)
                            else "%-20s   unavailable".format(it.provider)
                        )
                    }
                }
            }.getOrElse { "ERROR: ${it.message}" }
            runOnUiThread { output.text = text }
            busy = false
        }.start()
    }

    private fun render(device: String, s: SustainedBenchmark.Snapshot) = buildString {
        val el = s.elapsedMs / 1000
        appendLine(device)
        appendLine("INT8 1280x1280 · ${s.provider}")
        appendLine()
        appendLine("elapsed %d:%02d   frames %,d   %.1f fps"
            .format(Locale.US, el / 60, el % 60, s.frames, s.fpsNow))
        appendLine()

        if (s.lastKMedianMs.isNaN()) {
            appendLine("minute 1      : ${fmt(s.firstKMedianMs)}")
            appendLine("(degradation appears after minute 2)")
        } else {
            appendLine("minute 1      : ${fmt(s.firstKMedianMs)}")
            appendLine("minute ${s.minutes.size}%-7s: %s"
                .format(Locale.US, "", fmt(s.lastKMedianMs)))
            appendLine("DEGRADATION   : %+.1f%%".format(Locale.US, s.degradationPct))
        }
        appendLine("thermal : ${s.thermalNow}  (worst ${s.thermalWorst}, start ${s.thermalStart})")
        if (s.startedHot) {
            appendLine("!! STARTED WARM (%.1fC) - minute 1 is not a cold"
                .format(Locale.US, s.batteryTempStartC))
            appendLine("   baseline; let the device rest and rerun.")
        }
        appendLine("battery : %d%% -> %d%%, %.1fC -> %.1fC".format(
            Locale.US, s.batteryStartPct, s.batteryNowPct,
            s.batteryTempStartC, s.batteryTempNowC))
        appendLine("energy  : " + (s.mAhPer1000?.let { "%.1f mAh / 1000 frames".format(Locale.US, it) }
            ?: if (s.charging) "n/a - device is CHARGING" else "n/a - not enough drain yet"))
        appendLine()

        appendLine("%3s %5s %8s %7s %-9s %4s %6s"
            .format("min", "fps", "med ms", "p90", "thermal", "bat", "temp"))
        appendLine("-".repeat(48))
        for (m in s.minutes) {
            appendLine("%3d %5.1f %8.1f %7.1f %-9s %3d%% %5.1fC".format(
                Locale.US, m.index, m.frames / 60.0, m.medianMs, m.p90Ms,
                m.thermal, m.batteryPct, m.batteryTempC))
        }
        appendLine()
        appendLine("(screenshot any time - it's the datapoint)")
    }

    private fun fmt(v: Double) = if (v.isNaN()) "-" else "%.1f ms".format(Locale.US, v)
}
