"""Regime 4.3, four-panel view: my Fig. 9 against theirs.

    python scripts/regime4_3_compare.py

Reads the CSVs `regime4_3_tstar.py` writes and lays the four panels out in the
same 2x2 arrangement as Fig. 9 of arXiv:1203.6228, so the two can be held side
by side. Their figure plots $D_{emp}(\\tau=T)$ in red against $T$ with the
$D_{th}$ benchmarks dotted beneath it; this does the same.

The feature to look for in both is the **minimum**. Their caption states the
mechanism: *"The initial decline as T increases follows from reducing the
measurement noise. However, when T becomes very large, the 'true' evolution of
the eigenvectors is being felt, and leads to an increase of $D_{emp}$."* Short
windows are dominated by estimation noise, long windows average the rotation
away, and $T^*$ is where the two cross over.

Their values, from §6 of the full paper: Nikkei 600, SP500 700, DAX 450,
CAC40 400 days. The "around two years ($T^* = 500$ days)" in the Fig. 9 caption
is a round-number summary of those four, not a measurement.

The dotted `n_indep` line on each panel is `total / 2T` -- how many genuinely
non-overlapping back-to-back pairs sit behind each point. It falls below four
well before the right-hand end of every panel, which is worth carrying into any
statement about where exactly the minimum sits.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PANELS = [("nikkei", "Nikkei", 600), ("sp500", "SPX", 700),
          ("cac40", "CAC40", 400), ("dax", "DAX", 450)]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", type=Path, default=Path("results"))
    p.add_argument("--P", type=int, default=5)
    p.add_argument("--Q", type=int, default=10)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def load(indir, label, P, Q):
    for name in (f"{label}_tstar_P{P}Q{Q}_fine.csv", f"{label}_tstar_P{P}Q{Q}.csv"):
        path = indir / name
        if not path.exists():
            continue
        d = pd.read_csv(path, index_col=0)
        if ("raw", "d_emp") in d.columns or "('raw', 'd_emp')" in d.columns:
            d = pd.read_csv(path, header=[0, 1], index_col=0)
            return pd.DataFrame({"d_emp": d[("raw", "d_emp")], "d_th": d[("raw", "d_th")],
                                 "n_indep": d[("raw", "n_indep")]})
        return d[["d_emp", "d_th", "n_indep"]]
    raise SystemExit(f"no CSV for {label} -- run regime4_3_tstar.py --label {label} "
                     f"--P {P} --Q {Q}")


def main(argv=None):
    a = parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    print(f"{'panel':>8} {'their T*':>9} {'mine':>7} {'min D_emp':>10} {'indep @ min':>12}")
    for ax, (label, pretty, their) in zip(axes.ravel(), PANELS):
        v = load(a.indir, label, a.P, a.Q)
        mine = int(v["d_emp"].idxmin())
        ax.plot(v.index, v["d_emp"], color="crimson", lw=1.6, marker="o", ms=3,
                label=r"$D_{emp}(\tau=T)$")
        ax.plot(v.index, v["d_th"], ":", color="green", lw=1.4,
                label=r"$D_{th}(\tau=T)$")
        ax.axvline(mine, color="crimson", lw=0.9, ls="-.", alpha=0.6)
        ax.axvline(their, color="0.45", lw=0.9, ls="--")
        ax.annotate(f"theirs {their}", (their, ax.get_ylim()[1]), fontsize=7.5,
                    rotation=90, va="top", ha="right", color="0.4")
        ax.annotate(f"mine {mine}", (mine, ax.get_ylim()[1]), fontsize=7.5,
                    rotation=90, va="top", ha="left", color="crimson")
        ax.set_title(pretty, fontsize=10)
        ax.set_xlabel("$T$")
        ax.set_ylabel("$D$")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        print(f"{pretty:>8} {their:>9} {mine:>7} {v['d_emp'].min():>10.4f} "
              f"{v.loc[mine, 'n_indep']:>12.1f}")
    fig.suptitle(f"Regime 4.3 — $D_{{emp}}(\\tau=T)$ against window length, "
                 f"P={a.P}, Q={a.Q}  (cf. Fig. 9 of arXiv:1203.6228)", y=1.0)
    fig.tight_layout()
    out = a.out or a.indir / f"regime4_3_fig9_all_panels_P{a.P}Q{a.Q}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
