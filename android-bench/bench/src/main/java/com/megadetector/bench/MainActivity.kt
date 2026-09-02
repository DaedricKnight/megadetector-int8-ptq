package com.megadetector.bench

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.io.File

/**
 * Auto-runs the benchmark on launch — no tap needed, which matters on a remote
 * device farm where taps are laggy. Shows the result table (also written to CSV).
 * A "Run again" button repeats it. Plain Views, no Compose.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var output: TextView
    private lateinit var button: Button
    private var running = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        output = TextView(this).apply {
            textSize = 15f
            setPadding(28, 24, 28, 48)
            typeface = android.graphics.Typeface.MONOSPACE
            setTextIsSelectable(true)
        }
        button = Button(this).apply { text = "Run again" }
        button.setOnClickListener { runBenchmark() }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 40, 0, 0)
            addView(button)
            addView(ScrollView(this@MainActivity).apply { addView(output) })
        }
        // Keep content clear of the status/nav bars on modern devices.
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val b = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(0, b.top + 24, 0, b.bottom)
            insets
        }
        setContentView(root)

        // Fire immediately — the whole point on a farm is "launch and read".
        output.text = "MDV6 latency harness\n${MdLatencyBenchmark.deviceLabel()}\n\nStarting…"
        root.post { runBenchmark() }
    }

    private fun runBenchmark() {
        if (running) return
        running = true
        button.isEnabled = false
        output.text = "${MdLatencyBenchmark.deviceLabel()}\n\nRunning (warmup + iters per provider)…\nThis takes 1–4 min — please wait."
        Thread {
            val text = runCatching {
                val bench = MdLatencyBenchmark(this)
                val rows = bench.run()
                val device = MdLatencyBenchmark.deviceLabel()
                runCatching {
                    File(getExternalFilesDir(null), "mdv6_latency.csv")
                        .writeText(MdLatencyBenchmark.toCsv(rows, bench.modelSizeMb(), device))
                }
                render(device, bench.modelSizeMb(), rows)
            }.getOrElse { "ERROR: ${it.message}\n\nIs mdv6_v10c_int8.onnx in assets?" }
            runOnUiThread {
                output.text = text
                button.isEnabled = true
                running = false
            }
        }.start()
    }

    private fun render(device: String, sizeMb: Double, rows: List<MdLatencyBenchmark.Row>) =
        buildString {
            appendLine(device)
            appendLine("INT8 model: ${"%.1f".format(sizeMb)} MB · 1280×1280\n")
            appendLine("%-20s %9s %8s".format("provider", "median", "p90"))
            appendLine("-".repeat(42))
            rows.forEach {
                appendLine(
                    if (it.ok) "%-20s %7.1fms %6.1fms".format(it.provider, it.medianMs, it.p90Ms)
                    else "%-20s   unavailable".format(it.provider)
                )
            }
            appendLine("\n(screenshot this — it's the datapoint)")
        }
}
