#!/usr/bin/env python3
"""
Plot the accuracy/cost frontier from results/sweep_resolution.json.

Two panels, because the headline is a disagreement between them:

  left  — aggregate accuracy (mAP@0.5) against cost. One connected line per
          precision, one marker per resolution. This is the frontier you would
          use to pick an operating point.
  right — recall by ground-truth object size against resolution, FP32. This is
          why a single "best resolution" doesn't exist: small objects want more
          pixels, large objects want the resolution the model was trained at.

Cost is desktop-CPU latency, measured identically across every row — a
consistent proxy, not a phone number. Absolute on-device latency is in
results/README.md.

Usage:  python plot_pareto.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "results/sweep_resolution.json"

# Validated categorical slots, fixed order. Left panel keys on precision,
# right panel on size bucket; both stay inside the first four slots.
SLOT = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"),
        ("#1baf7a", "#199e70"), ("#eda100", "#c98500")]

THEMES = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  muted="#898781", grid="#e1e0d9", axis="#c3c2b7", i=0),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 muted="#898781", grid="#2c2c2a", axis="#383835", i=1),
}


def style(ax, t):
    ax.set_facecolor(t["surface"])
    ax.grid(color=t["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(t["axis"])
        ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=t["muted"], labelsize=9.5, length=0)


def render(data, mode):
    t = THEMES[mode]
    rows = data["rows"]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.4, 5.6), dpi=160)
    fig.patch.set_facecolor(t["surface"])

    # ---- left: accuracy vs cost -------------------------------------------
    for k, precision in enumerate(("FP32", "INT8")):
        pts = sorted((r for r in rows if r["precision"] == precision),
                     key=lambda r: r["latency_ms"])
        if not pts:
            continue
        c = SLOT[k][t["i"]]
        axl.plot([p["latency_ms"] for p in pts], [p["map_50"] for p in pts],
                 color=c, linewidth=2.0, marker="o", markersize=8,
                 markeredgecolor=t["surface"], markeredgewidth=2,
                 label=precision, zorder=3)
        for p in pts:
            axl.annotate(f"{p['imgsz']}", (p["latency_ms"], p["map_50"]),
                         textcoords="offset points", xytext=(0, 11),
                         ha="center", color=c, fontsize=9.5, fontweight="600")
    axl.set_xlabel("cost — desktop CPU latency per image (ms)",
                   color=t["secondary"], fontsize=10.5, labelpad=9)
    axl.set_ylabel("mAP@0.5", color=t["secondary"], fontsize=10.5, labelpad=9)
    axl.set_title("Accuracy vs cost — labels are input resolution",
                  color=t["primary"], fontsize=12.5, fontweight="600",
                  loc="left", pad=12)
    leg = axl.legend(frameon=False, fontsize=10, labelcolor=t["secondary"],
                     loc="lower right")
    for ln in leg.get_lines():
        ln.set_linewidth(2.6)
    style(axl, t)

    # ---- right: recall by object size vs resolution ------------------------
    buckets, sizes = [], sorted({r["imgsz"] for r in rows})
    for r in rows:
        for b in r.get("by_size") or []:
            if b["bucket"] not in buckets:
                buckets.append(b["bucket"])
    for k, bucket in enumerate(buckets):
        xs, ys = [], []
        for s in sizes:
            row = next((r for r in rows
                        if r["imgsz"] == s and r["precision"] == "FP32"), None)
            b = next((x for x in (row or {}).get("by_size") or []
                      if x["bucket"] == bucket), None)
            if b:
                xs.append(s)
                ys.append(b["fp32"])
        if not xs:
            continue
        c = SLOT[k % 4][t["i"]]
        axr.plot(xs, ys, color=c, linewidth=2.0, marker="o", markersize=7,
                 markeredgecolor=t["surface"], markeredgewidth=2, zorder=3)
        axr.annotate(bucket.split(" ")[0], (xs[-1], ys[-1]),
                     textcoords="offset points", xytext=(9, 0), va="center",
                     color=c, fontsize=10, fontweight="600")
    axr.set_xticks(sizes)
    axr.set_xlabel("input resolution (px)", color=t["secondary"],
                   fontsize=10.5, labelpad=9)
    axr.set_ylabel("FP32 recall@0.2", color=t["secondary"], fontsize=10.5, labelpad=9)
    axr.set_title("…and why there is no single best resolution",
                  color=t["primary"], fontsize=12.5, fontweight="600",
                  loc="left", pad=12)
    style(axr, t)
    axr.set_xlim(min(sizes) - 60, max(sizes) + 190)

    fig.suptitle("MDV6-yolov10-c: input resolution is a bigger lever than INT8",
                 color=t["primary"], fontsize=15, fontweight="600",
                 x=0.008, ha="left", y=0.975)
    fig.text(0.008, 0.905,
             "210-image class-balanced NACTI subset · τ = 0.2 · IoU 0.5 · "
             "ONNX Runtime CPU. Size buckets are fractions of frame area.",
             color=t["secondary"], fontsize=10)
    fig.subplots_adjust(left=0.062, right=0.975, top=0.79, bottom=0.115, wspace=0.235)
    out = f"results/pareto_resolution_{mode}.png"
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    if not os.path.exists(SRC):
        raise SystemExit(f"{SRC} missing — run sweep_resolution.py first")
    data = json.load(open(SRC))
    for m in ("light", "dark"):
        render(data, m)
