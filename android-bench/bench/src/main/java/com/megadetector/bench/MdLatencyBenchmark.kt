package com.megadetector.bench

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import java.nio.FloatBuffer

/**
 * Inference-only latency of an ONNX detector across ONNX Runtime execution
 * providers on the device. Produces the "on-device latency" rows of the
 * MegaDetector V6 INT8 issue table.
 *
 * Latency is data-independent for conv/attention graphs, so we time a single
 * fixed FP32 [1,3,imgsz,imgsz] input reused across iterations — this isolates
 * the model's compute cost from image I/O and makes providers comparable. The
 * QDQ INT8 model keeps float32 IO (the Quantize/Dequantize nodes are internal),
 * so one input path serves FP32 / FP16 / INT8 alike.
 *
 * Each provider is attempted independently: if a build lacks XNNPACK or NNAPI,
 * that row is reported unavailable rather than failing the whole run.
 */
class MdLatencyBenchmark(
    private val context: Context,
    private val assetModel: String = "mdv6_v10c_int8.onnx",
    private val imgsz: Int = 1280,
    private val warmup: Int = 8,
    private val iters: Int = 40,
) {
    private val env = OrtEnvironment.getEnvironment()
    private val modelPath: String by lazy { copyAsset(assetModel) }
    private val input = FloatArray(3 * imgsz * imgsz) { 0.5f }   // content irrelevant
    private val shape = longArrayOf(1, 3, imgsz.toLong(), imgsz.toLong())

    data class Row(
        val provider: String,
        val ok: Boolean,
        val medianMs: Double = 0.0,
        val p90Ms: Double = 0.0,
        val minMs: Double = 0.0,
        val meanMs: Double = 0.0,
        val note: String = "",
    )

    /**
     * Runs every provider and returns one row each. NNAPI gets far fewer
     * iterations: on some SoCs the driver rejects a QDQ INT8 graph and falls
     * back to the (very slow) nnapi-reference CPU path — we only need enough
     * samples to record that penalty, not 40 of them.
     */
    fun run(): List<Row> = listOf(
        bench("CPU (1 thread)", warmup, iters) { cpu(1) },
        bench("CPU (4 threads)", warmup, iters) { cpu(4) },
        bench("XNNPACK (4 threads)", warmup, iters) { xnnpack(4) },
        bench("NNAPI", 2, 6) { nnapi() },
    )

    fun modelSizeMb(): Double = File(modelPath).length() / 1e6

    private fun bench(name: String, warmup: Int, iters: Int,
                      options: () -> OrtSession.SessionOptions): Row {
        var session: OrtSession? = null
        var tensor: OnnxTensor? = null
        return try {
            session = env.createSession(modelPath, options())
            val inName = session.inputNames.first()
            tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape)
            val feed = mapOf(inName to tensor)

            repeat(warmup) { session.run(feed).close() }

            val t = DoubleArray(iters)
            for (i in 0 until iters) {
                val t0 = System.nanoTime()
                session.run(feed).use { /* discard outputs — timing only */ }
                t[i] = (System.nanoTime() - t0) / 1e6
            }
            t.sort()
            Row(
                provider = name, ok = true,
                medianMs = t[iters / 2],
                p90Ms = t[(iters * 90 / 100).coerceAtMost(iters - 1)],
                minMs = t.first(),
                meanMs = t.average(),
            )
        } catch (e: Throwable) {
            // Missing EP in this AAR, unsupported op on NNAPI, etc. — report, don't crash.
            Log.w(TAG, "provider '$name' unavailable: ${e.message}")
            Row(provider = name, ok = false, note = e.message ?: e.javaClass.simpleName)
        } finally {
            tensor?.close()
            session?.close()
        }
    }

    private fun cpu(threads: Int) = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    }

    private fun xnnpack(threads: Int) = cpu(threads).apply {
        // Throws if the AAR has no XNNPACK EP — caught in bench().
        addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
    }

    private fun nnapi() = OrtSession.SessionOptions().apply {
        // Uses the device NPU/GPU/CPU via Android NNAPI; QDQ INT8 is the format
        // its EP consumes. Throws if unavailable — caught in bench().
        addNnapi()
    }

    private fun copyAsset(name: String): String {
        val out = File(context.cacheDir, name)
        context.assets.open(name).use { i -> out.outputStream().use { o -> i.copyTo(o) } }
        return out.absolutePath
    }

    companion object {
        private const val TAG = "MdBench"

        /** "Google Pixel 8 Pro · Google Tensor G3" — SoC matters: the NNAPI/NPU
         * result depends on it, so farm runs must record which chip they used. */
        fun deviceLabel(): String {
            val soc = if (Build.VERSION.SDK_INT >= 31)
                "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}".trim() else "SoC n/a"
            return "${Build.MANUFACTURER} ${Build.MODEL} · $soc"
        }

        /** CSV for the issue table; header matches the on-device latency columns. */
        fun toCsv(rows: List<Row>, sizeMb: Double, device: String): String = buildString {
            appendLine("device,provider,size_mb,median_ms,p90_ms,min_ms,mean_ms,status")
            for (r in rows) {
                appendLine(
                    "$device,${r.provider},${"%.1f".format(sizeMb)}," +
                        if (r.ok) "${"%.1f".format(r.medianMs)},${"%.1f".format(r.p90Ms)}," +
                            "${"%.1f".format(r.minMs)},${"%.1f".format(r.meanMs)},ok"
                        else ",,,,unavailable: ${r.note}"
                )
            }
        }
    }
}
