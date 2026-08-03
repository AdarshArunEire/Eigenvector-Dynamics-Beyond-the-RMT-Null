"""Build the Stage 1 README/social-preview graphic from Regime 4.7 outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "assets" / "stage1-flag-signal.png"

PANELS = (
    ("sp500_full", "S&P 500"),
    ("nikkei_full", "Nikkei"),
    ("dax_full", "DAX"),
    ("cac40_full", "CAC 40"),
)


def load_flag_results():
    rows = []
    for label, display in PANELS:
        matches = sorted(
            RESULTS.glob(
                f"{label}_flag_T*_dims1-3-6_h42_step14_"
                "delta0p25_standardised_summary.csv"
            )
        )
        if len(matches) != 1:
            raise RuntimeError(f"Expected one primary 4.7 summary for {label}, found {matches}")
        result = pd.read_csv(matches[0])
        flag = result.loc[result["component"] == "flag_nested"]
        if len(flag) != 1:
            raise RuntimeError(f"Expected one flag_nested row in {matches[0]}")
        row = flag.iloc[0]
        rows.append(
            {
                "panel": display,
                "observed": float(row["mean_cosine"]),
                "null": float(row["null_mean_cosine_mean"]),
            }
        )
    return rows


def build():
    rows = load_flag_results()

    bg = "#0B1020"
    foreground = "#F7F9FC"
    muted = "#A9B4C8"
    grid = "#293247"
    observed = "#56D6C9"
    null = "#748099"
    accent = "#8AA8FF"

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=bg)
    ax = fig.add_axes([0.54, 0.20, 0.41, 0.62], facecolor=bg)

    fig.text(0.055, 0.865, "EIGENVECTOR DYNAMICS BEYOND THE RMT NULL",
             color=accent, fontsize=12, fontweight="bold")
    fig.text(0.055, 0.735, "Direction survives\nthe noise.",
             color=foreground, fontsize=34, fontweight="bold", linespacing=1.05)
    fig.text(0.055, 0.555,
             "Financial partial-flag motion remains\npersistent beyond matched calendar nulls.",
             color=muted, fontsize=15, linespacing=1.45)
    fig.text(0.055, 0.382, r"$\mathrm{Flag}(N;1,3,6)$",
             color=observed, fontsize=24, fontweight="bold")
    fig.text(0.055, 0.305,
             "Market direction  •  top-three core\n•  six-dimensional collision buffer",
             color=muted, fontsize=12.5, linespacing=1.5)
    fig.text(0.055, 0.105,
             "STAGE 1 COMPLETE   •   FORECASTING NEXT",
             color=foreground, fontsize=12, fontweight="bold")

    y = list(range(len(rows)))[::-1]
    ax.set_xlim(0, 0.175)
    ax.set_ylim(-0.7, len(rows) - 0.3)

    for yi, row in zip(y, rows):
        ax.plot([row["null"], row["observed"]], [yi, yi], color=grid,
                linewidth=5, solid_capstyle="round", zorder=1)
        ax.scatter(row["null"], yi, s=95, facecolor=bg, edgecolor=null,
                   linewidth=2.2, zorder=3)
        ax.scatter(row["observed"], yi, s=125, color=observed,
                   edgecolor=bg, linewidth=1.5, zorder=4)
        ax.text(row["null"], yi - 0.27, f'{row["null"]:.3f}', color=null,
                fontsize=10, ha="center", va="top")
        ax.text(row["observed"] + 0.004, yi, f'{row["observed"]:.3f}',
                color=foreground, fontsize=11, fontweight="bold",
                ha="left", va="center")

    ax.set_yticks(y, [row["panel"] for row in rows])
    ax.tick_params(axis="y", colors=foreground, labelsize=12, length=0, pad=12)
    ax.set_xticks([0.00, 0.05, 0.10, 0.15])
    ax.set_xticklabels(["0", ".05", ".10", ".15"], color=muted, fontsize=10)
    ax.tick_params(axis="x", length=0, pad=9)
    ax.grid(axis="x", color=grid, linewidth=0.8, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(0.0, len(rows) - 0.02, "MEAN TANGENT COSINE",
            transform=ax.transData, color=muted, fontsize=10,
            fontweight="bold", va="bottom")
    ax.scatter(0.0, -0.50, s=75, color=observed, edgecolor=bg, linewidth=1,
               clip_on=False)
    ax.text(0.006, -0.50, "Observed", color=foreground, fontsize=10,
            va="center", clip_on=False)
    ax.scatter(0.065, -0.50, s=65, facecolor=bg, edgecolor=null,
               linewidth=2, clip_on=False)
    ax.text(0.071, -0.50, "Shuffled null", color=muted, fontsize=10,
            va="center", clip_on=False)

    fig.text(0.54, 0.085,
             "Standardised returns  •  42-day horizon  •  S&P exploratory (20 nulls)",
             color=muted, fontsize=9.5)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=100, facecolor=fig.get_facecolor(),
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    build()
