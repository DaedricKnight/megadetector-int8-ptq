#!/usr/bin/env python3
"""
Plot the sustained-load curves from results/*.csv — light and dark PNGs for the
GitHub issue.

The table already carries the numbers; what it can't show is the *shape* of the
decay, which is the point: Tensor G3 falls down a staircase, Dimensity 900 runs
flat, and the ranking at minute 1 is not the ranking at minute 30.

Colors are the validated 4-slot categorical palette (adjacent pairlist, both
modes pass). Aqua and yellow sit under 3:1 on the light surface, so the relief
rule applies and every series is direct-labeled as well as legended — identity
never rests on hue alone.

Usage:  python plot_sustained.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Color follows the entity, never its rank — fixed slot order, assigned by SoC.
SERIES = [
    ("Snapdragon 8 Elite", "Galaxy Z Fold8 Ultra", "results/sustained_galaxyzfold_sd8elite_30min.csv",
     "#2a78d6", "#3987e5"),
    ("Tensor G3", "Pixel 8 Pro", "results/sustained_pixel8pro_tensorG3_30min.csv",
     "#eb6834", "#d95926"),
    ("Exynos 2400e", "Galaxy S24 FE", "results/sustained_galaxys24fe_exynos2400e_30min.csv",
     "#1baf7a", "#199e70"),
    ("Dimensity 900", "Galaxy M53", "results/sustained_galaxym53_dimensity900_30min.csv",
     "#eda100", "#c98500"),
]

THEMES = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  muted="#898781", grid="#e1e0d9", axis="#c3c2b7", idx=3),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 muted="#898781", grid="#2c2c2a", axis="#383835", idx=4),
}


def load(path):
    rows = list(csv.DictReader([l for l in open(path) if not l.startswith("#")]))
    return ([int(r["minute"]) for r in rows], [float(r["median_ms"]) for r in rows])


def declutter(labels, min_gap):
    """Nudge end-labels apart so close series stay readable (Fold and S24 FE end
    ~11 ms apart). Order is preserved; only the drawn y is moved."""
    labels = sorted(labels, key=lambda t: t[0])
    for i in range(1, len(labels)):
        if labels[i][0] - labels[i - 1][0] < min_gap:
            labels[i] = (labels[i - 1][0] + min_gap, labels[i][1], labels[i][2])
    return labels


def render(mode):
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(12.2, 6.2), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    ends = []
    for soc, device, path, c_light, c_dark in SERIES:
        if not os.path.exists(path):
            print(f"  missing {path} — skipped")
            continue
        color = c_light if mode == "light" else c_dark
        x, y = load(path)
        ax.plot(x, y, color=color, linewidth=2.0, solid_capstyle="round",
                label=f"{soc} · {device}", zorder=3)
        ends.append((y[-1], color, soc))

    for y, color, soc in declutter(ends, min_gap=26):
        ax.text(30.5, y, soc, color=color, fontsize=10.5, va="center",
                ha="left", fontweight="600")

    ax.set_xlim(1, 30)
    ax.set_ylim(0, 660)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.grid(axis="y", color=t["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(t["axis"])
        ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=t["muted"], labelsize=10, length=0)

    ax.set_xlabel("minute of continuous inference", color=t["secondary"],
                  fontsize=10.5, labelpad=10)
    ax.set_ylabel("median latency per frame (ms) — lower is better",
                  color=t["secondary"], fontsize=10.5, labelpad=10)
    ax.set_title("MDV6-yolov10-c INT8: what 30 minutes of continuous inference costs",
                 color=t["primary"], fontsize=14.5, fontweight="600",
                 loc="left", pad=26)
    ax.text(0, 1.035,
            "Every device is slower at minute 30 than at minute 1 — and the ranking changes. "
            "ORT CPU (4 threads), 1280x1280, batch 1.",
            transform=ax.transAxes, color=t["secondary"], fontsize=10.5, va="bottom")

    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.005, 0.98), frameon=False,
                    fontsize=10, labelcolor=t["secondary"], handlelength=1.6)
    for line in leg.get_lines():
        line.set_linewidth(2.6)

    fig.subplots_adjust(left=0.077, right=0.805, top=0.845, bottom=0.115)
    out = f"results/sustained_curves_{mode}.png"
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    for m in ("light", "dark"):
        render(m)
