"""Generate the three front-page figures from the committed result tables.

Deliberately spare: no gridlines competing with the data, no legend where a
direct label will do, one accent colour per figure. Each figure carries exactly
one claim.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "results" / "stage2" / "capture_ladder"
MODEL = ROOT / "results" / "stage2" / "model4_1_visible_coupling"
ASSETS = ROOT / "assets"
LABELS = ("cac40_full", "dax_full", "nikkei_full", "sp500_full")
NAMES = {"cac40_full": "CAC 40", "dax_full": "DAX",
         "nikkei_full": "Nikkei", "sp500_full": "S&P 500"}

INK = "#1b1b1b"
MUTED = "#9aa0a6"
FAINT = "#e3e5e8"
ACCENT = "#c2371f"
BLUE = "#2b5d8a"
GREEN = "#3d7a5a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def _strip(ax):
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(.8)


def figure_one():
    """What a six-factor model spans, and how little of the gap is reachable."""
    summary = pd.concat([pd.read_csv(LADDER / l / "panel_summary.csv") for l in LABELS])
    comparisons = pd.concat(
        [pd.read_csv(LADDER / l / "test_comparisons.csv") for l in LABELS])
    ewma = (comparisons[comparisons.entrant.str.startswith("ewma")]
            .groupby("label").paired_improvement.max())

    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(LABELS))[::-1]
    for position, label in zip(y, LABELS):
        row = summary[summary.label == label].iloc[0]
        floor, frozen = row.random_floor, row.frozen_capture
        reach = frozen + float(ewma[label])
        honest = frozen + row.real_headroom

        ax.barh(position, 1.0, color=FAINT, height=.5, zorder=1)
        ax.barh(position, frozen, color=BLUE, height=.5, zorder=3)
        ax.barh(position, honest - frozen, left=frozen, color="#cfe0ee",
                height=.5, zorder=2)
        ax.plot([reach, reach], [position - .25, position + .25],
                color=ACCENT, lw=2.2, zorder=5, solid_capstyle="butt")
        ax.plot([floor, floor], [position - .25, position + .25],
                color=MUTED, lw=1, ls=(0, (2, 2)), zorder=4)
        ax.text(frozen / 2, position, f"{frozen:.0%}", va="center", ha="center",
                color="white", fontsize=9, fontweight="bold", zorder=6)
        ax.text(1.005, position, f"{1 - frozen:.0%} unspanned", va="center",
                ha="left", fontsize=9, color=MUTED)

    ax.set_yticks(y, [NAMES[l] for l in LABELS], fontsize=10, color=INK)
    ax.set_xticks(np.arange(0, 1.01, .25), ["0", "25%", "50%", "75%", "100%"])
    ax.set_xlim(0, 1.28)
    ax.set_ylim(-.75, len(LABELS) + .05)
    ax.set_xlabel("share of next quarter's realised cross-sectional variance spanned "
                  "by six factors", labelpad=8)
    _strip(ax)

    row = summary[summary.label == "cac40_full"].iloc[0]
    ax.annotate("what perfect hindsight of its own six directions adds",
                xy=(row.frozen_capture + row.real_headroom * .55, 3.28),
                xytext=(row.frozen_capture + .10, 3.62),
                fontsize=8.5, color="#5b7d99", va="center",
                arrowprops=dict(arrowstyle="-", color="#a8c2d6", lw=.8))
    ax.annotate("best EWMA gets this far",
                xy=(row.frozen_capture + float(ewma["cac40_full"]), 2.72),
                xytext=(row.frozen_capture + .045, 2.35),
                fontsize=8.5, color=ACCENT, va="center",
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=.8))
    ax.text(.012, -.52, "dashed line = random six-dimensional frame",
            fontsize=8, color=MUTED)
    fig.suptitle("A six-factor risk model misses most of next quarter's risk — "
                 "and almost none of the gap is reachable",
                 x=.02, ha="left", fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(ASSETS / "capture-headroom.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_two():
    """The entire rotationally-invariant class is pinned at zero."""
    comparisons = pd.concat(
        [pd.read_csv(LADDER / l / "test_comparisons.csv") for l in LABELS])
    entrants = [("ledoit_wolf", "Ledoit–Wolf"), ("oas", "OAS"), ("qis", "QIS"),
                ("ewma_hl_252", "EWMA hl=252")]

    fig, ax = plt.subplots(figsize=(9, 3.4))
    width = .2
    base = np.arange(len(entrants))
    for offset, label in enumerate(LABELS):
        panel = comparisons[comparisons.label == label].set_index("entrant")
        values = [panel.loc[key, "paired_improvement"] for key, _ in entrants]
        shade = plt.cm.Blues(.35 + .18 * offset)
        ax.bar(base + (offset - 1.5) * width, values, width * .9,
               color=[shade if k != "ewma_hl_252" else ACCENT for k, _ in entrants],
               alpha=1 if offset == 3 else .55 + .15 * offset, zorder=3)

    ax.axhline(0, color=MUTED, lw=.9, zorder=2)
    ax.axvspan(-.5, 2.5, color=FAINT, alpha=.55, zorder=1)
    ax.set_xticks(base, [name for _, name in entrants], fontsize=10, color=INK)
    ax.set_ylabel("capture gained over Frozen\n(test origins, four panels)", fontsize=9)
    ax.set_xlim(-.5, 3.5)
    _strip(ax)
    top = ax.get_ylim()[1]
    ax.set_ylim(ax.get_ylim()[0], top * 1.3)
    ax.text(1.0, top * 1.12, "rotationally invariant  →  structurally pinned at zero",
            ha="center", fontsize=9.5, color=INK)
    ax.text(3.0, top * 1.12, "rotates the frame  →  the only entrant that scores",
            ha="center", fontsize=9.5, color=ACCENT)
    for position in range(3):
        ax.text(position, top * .04, "0.0000" if position < 2 else "≈0",
                ha="center", fontsize=8.5, color=MUTED)
    fig.suptitle("Eigenvalue cleaning cannot move a subspace metric — by construction, "
                 "not by accident",
                 x=.02, ha="left", fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(ASSETS / "rie-pinned-at-zero.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_three():
    """Four objectives pick different directions and score the same."""
    model = pd.concat([pd.read_csv(MODEL / l / "panel_summary.csv") for l in LABELS])
    tuning = pd.concat([pd.read_csv(MODEL / l / "validation_tuning.csv") for l in LABELS])
    best = (tuning.sort_values("validation_mean_capture")
            .groupby(["label", "fit"], as_index=False).last())

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), width_ratios=(1, 1.15))
    order = ["geometric", "ridge", "direct"]
    pretty = {"geometric": "pure geometry", "ridge": "gap-weighted geometry",
              "direct": "exact capture search"}

    ax = axes[0]
    for offset, key in enumerate(order):
        values = [model[model.label == l][f"fit_agreement_{key}"].iloc[0] for l in LABELS]
        ax.scatter(values, [offset] * len(LABELS), s=54,
                   color=[ACCENT if v < .3 else BLUE for v in values], zorder=3)
    ax.axvline(0, color=MUTED, lw=.9, ls=(0, (2, 2)), zorder=2)
    ax.axvline(1, color=FAINT, lw=6, zorder=1)
    ax.set_yticks(range(len(order)), [pretty[k] for k in order], fontsize=9.5, color=INK)
    ax.set_xlim(-.45, 1.15)
    ax.set_xticks([-.25, 0, .5, 1], ["−0.25", "0", "0.5", "1"])
    ax.set_xlabel("cosine with the risk-optimal direction", fontsize=9)
    ax.set_ylim(-.95, len(order) - .3)
    _strip(ax)
    ax.text(-.16, -.34, "CAC: pure geometry points\nthe opposite way", fontsize=8.5,
            color=ACCENT, ha="center", va="center")
    ax.set_title("the objectives disagree…", fontsize=10, color=INK, loc="left", pad=8)

    ax = axes[1]
    y = np.arange(len(LABELS))[::-1]
    for position, label in zip(y, LABELS):
        panel = best[best.label == label].set_index("fit")
        values = [panel.loc[k, "validation_minus_frozen"] * 1e4
                  for k in ("gradient", "ridge", "geometric", "direct")]
        ax.plot([min(values), max(values)], [position, position],
                color=FAINT, lw=5, solid_capstyle="round", zorder=1)
        ax.scatter(values, [position] * 4, s=42, color=BLUE, zorder=3)
        ax.text(max(values) + 1.8, position,
                f"spread {max(values) - min(values):.1f}", va="center",
                fontsize=8.5, color=MUTED)
    ax.set_yticks(y, [NAMES[l] for l in LABELS], fontsize=9.5, color=INK)
    ax.set_xlabel("validation capture gained over Frozen  (×10⁻⁴)", fontsize=9)
    ax.set_ylim(-.6, len(LABELS) - .4)
    ax.set_xlim(0, 72)
    _strip(ax)
    ax.set_title("…and score identically anyway", fontsize=10, color=INK,
                 loc="left", pad=8)

    fig.suptitle("The objective is flat: rotate the fitted direction 90° and lose nothing",
                 x=.02, ha="left", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(ASSETS / "flat-objective.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ASSETS.mkdir(exist_ok=True)
    figure_one()
    figure_two()
    figure_three()
    print("wrote capture-headroom.png, rie-pinned-at-zero.png, flat-objective.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
