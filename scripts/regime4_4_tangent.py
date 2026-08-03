"""Regime 4.4: is eigenspace motion directionally persistent?

The subspace variogram in Regime 4.2 establishes displacement, not
forecastability. At each current window Y_t this experiment constructs

    H_minus = -Log_{Y_t}(Y_{t-h})
    H_plus  =  Log_{Y_t}(Y_{t+h})

in the same tangent space. Their cosine is the intrinsic direction-persistence
label. Exp_{Y_t}(H_minus) is the constant-velocity forecast; its bounded
projector loss is compared with the static forecast Y_t.

Overlapping estimation windows can manufacture apparent momentum even under a
static population covariance. The primary comparison is therefore not cosine
against zero. It is the observed statistic against block-permuted-return
surrogates, which preserve daily cross-sections, fat tails, short volatility
episodes, window overlap and every numerical choice while destroying the
calendar ordering at scales longer than ``block_size``.

Primary specification frozen for the first pass:

    T=max(N, 250), P=3, step=14d, horizon=42d, block_size=21d

The 42-day horizon matches the out-of-sample covariance benchmark used by BAHC.
Standardised returns are primary; raw returns are a sensitivity check.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import standardise, to_correlation_panel, to_panel
from src.grassmann import (containment_loss, grassmann_exp, grassmann_log,
                           tangent_cosine)
from src.overlap import sample_covariance, spectral


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="nikkei")
    p.add_argument("--indir", type=Path, default=Path("data/cache"))
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--T", type=int, default=None,
                   help="window length; default max(N, 250)")
    p.add_argument("--P", type=int, default=3)
    p.add_argument("--step", type=int, default=14,
                   help="days between window starts")
    p.add_argument("--horizon", type=int, default=42,
                   help="days from past to current and current to future")
    p.add_argument("--block-size", type=int, default=21,
                   help="calendar-shuffle block length in days")
    p.add_argument("--shuffles", type=int, default=99,
                   help="block-permuted null repetitions; 20 is exploratory, 99 is quotable")
    p.add_argument("--seed", type=int, default=20260802)
    p.add_argument("--mode", choices=("both", "raw", "standardised"),
                   default="both")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def block_permutation_indices(length, block_size, rng):
    """Permutation of all columns in intact consecutive calendar blocks."""
    if length < 1 or block_size < 1:
        raise ValueError("length and block_size must be positive")
    blocks = [np.arange(i, min(i + block_size, length))
              for i in range(0, length, block_size)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def window_bases(panel, T, step, P, use_standardised):
    """Leading P-space for each rolling window."""
    starts, bases = [], []
    for start in range(0, panel.shape[1] - T + 1, step):
        window = panel[:, start:start + T]
        if use_standardised:
            window = standardise(window, window=1)
        corr_panel = to_correlation_panel(window)
        _, vectors = spectral(sample_covariance(corr_panel))
        starts.append(start)
        bases.append(vectors[:, :P])
    if not bases:
        raise ValueError(f"panel has {panel.shape[1]} days, shorter than T={T}")
    return np.asarray(starts), np.asarray(bases)


def tangent_series(starts, bases, horizon, step):
    """One intrinsic direction and forecast comparison per central window."""
    if horizon < step or horizon % step:
        raise ValueError(f"horizon={horizon} must be a positive multiple of step={step}")
    offset = horizon // step
    if len(bases) <= 2 * offset:
        raise ValueError("not enough rolling windows for one past/current/future triple")

    rows = []
    for current in range(offset, len(bases) - offset):
        past, now, future = bases[current - offset], bases[current], bases[current + offset]
        incoming = -grassmann_log(now, past)
        outgoing = grassmann_log(now, future)
        prediction = grassmann_exp(now, incoming)
        static_loss = containment_loss(future, now, normalise=True)
        velocity_loss = containment_loss(future, prediction, normalise=True)
        rows.append({
            "start": int(starts[current]),
            "past_start": int(starts[current - offset]),
            "future_start": int(starts[current + offset]),
            "cosine": tangent_cosine(incoming, outgoing),
            "incoming_speed": float(np.linalg.norm(incoming, ord="fro")),
            "outgoing_speed": float(np.linalg.norm(outgoing, ord="fro")),
            "static_loss": static_loss,
            "constant_velocity_loss": velocity_loss,
            "constant_velocity_skill": (1.0 - velocity_loss / static_loss
                                        if static_loss > 0 else np.nan),
        })
    return pd.DataFrame(rows)


def summarise(series):
    """Statistics that decide the direction and constant-velocity gates."""
    cos = series["cosine"].to_numpy(dtype=float)
    static = series["static_loss"].to_numpy(dtype=float)
    velocity = series["constant_velocity_loss"].to_numpy(dtype=float)
    ok = np.isfinite(cos)
    return {
        "n_triples": int(ok.sum()),
        "mean_cosine": float(np.mean(cos[ok])) if ok.any() else np.nan,
        "median_cosine": float(np.median(cos[ok])) if ok.any() else np.nan,
        "positive_fraction": float(np.mean(cos[ok] > 0)) if ok.any() else np.nan,
        "mean_incoming_speed": float(series["incoming_speed"].mean()),
        "mean_outgoing_speed": float(series["outgoing_speed"].mean()),
        "static_loss": float(np.mean(static)),
        "constant_velocity_loss": float(np.mean(velocity)),
        # Ratio of mean losses, not mean of per-window ratios: stable when an
        # individual future happens to lie almost exactly at the current point.
        "constant_velocity_skill": float(1.0 - np.mean(velocity) / np.mean(static)),
    }


def empirical_upper_p(observed, null):
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(observed):
        return np.nan
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def run_mode(panel, T, P, step, horizon, block_size, shuffles, rng,
             use_standardised, progress_label=""):
    starts, bases = window_bases(panel, T, step, P, use_standardised)
    observed_series = tangent_series(starts, bases, horizon, step)
    observed = summarise(observed_series)

    null_rows = []
    for replicate in range(shuffles):
        indices = block_permutation_indices(panel.shape[1], block_size, rng)
        shuffled = panel[:, indices]
        s_starts, s_bases = window_bases(shuffled, T, step, P, use_standardised)
        row = summarise(tangent_series(s_starts, s_bases, horizon, step))
        row["replicate"] = replicate
        null_rows.append(row)
        if shuffles >= 10 and ((replicate + 1) % 10 == 0 or replicate + 1 == shuffles):
            print(f"    {progress_label} null {replicate + 1}/{shuffles}")
    null = pd.DataFrame(null_rows)

    for statistic in ("mean_cosine", "positive_fraction", "constant_velocity_skill"):
        values = null[statistic].to_numpy() if len(null) else np.array([])
        observed[f"null_{statistic}_mean"] = float(np.mean(values)) if len(values) else np.nan
        observed[f"null_{statistic}_q05"] = float(np.quantile(values, 0.05)) if len(values) else np.nan
        observed[f"null_{statistic}_q95"] = float(np.quantile(values, 0.95)) if len(values) else np.nan
        observed[f"p_upper_{statistic}"] = empirical_upper_p(observed[statistic], values)
    return observed_series, observed, null


def plot_series(series_by_mode, label, T, P, horizon, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, series in series_by_mode.items():
        x = pd.to_datetime(series["date"])
        rolling = series["cosine"].rolling(12, center=True, min_periods=4).mean()
        axes[0].plot(x, rolling, lw=1.4, label=tag)
        axes[1].scatter(series["static_loss"], series["constant_velocity_loss"],
                        s=10, alpha=0.45, label=tag)
    axes[0].axhline(0, color="0.5", lw=0.8)
    axes[0].set_ylabel("rolling mean tangent cosine")
    axes[0].set_xlabel("central window end")
    axes[1].plot([0, 1], [0, 1], color="0.5", lw=0.8, transform=axes[1].transAxes)
    axes[1].set_xlabel("static projector loss")
    axes[1].set_ylabel("constant-velocity projector loss")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(f"{label}: Regime 4.4 tangent persistence, T={T}, P={P}, h={horizon}d")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    a = parse_args(argv)
    returns = pd.read_parquet(a.indir / f"{a.label}_returns.parquet")
    panel = to_panel(returns)
    N, total = panel.shape
    T = a.T or max(N, 250)
    if not (0 < a.P < N):
        raise SystemExit(f"need 0 < P < N; got P={a.P}, N={N}")
    if a.horizon % a.step:
        raise SystemExit(f"--horizon {a.horizon} must be divisible by --step {a.step}")
    if a.block_size >= a.horizon:
        print("warning: block-size >= horizon preserves some horizon-scale ordering; "
              "treat this as a conservative sensitivity null")

    modes = (["raw", "standardised"] if a.mode == "both" else [a.mode])
    print(f"{a.label}: N={N}, {total} days, T={T}, P={a.P}, step={a.step}, "
          f"h={a.horizon}, block={a.block_size}, shuffles={a.shuffles}")
    print(f"  independent windows available: {total / T:.1f}")
    if T < 200:
        print("  warning: T < 200, where Mikkelsen's loading-persistence estimates were biased; "
              "do not interpret a non-result as absence")

    rng = np.random.default_rng(a.seed)
    all_series, all_null, summary_rows = {}, [], []
    for tag in modes:
        use_standardised = tag == "standardised"
        print(f"\n  {tag}")
        series, summary, null = run_mode(
            panel, T, a.P, a.step, a.horizon, a.block_size, a.shuffles, rng,
            use_standardised, progress_label=tag)
        date_positions = series["start"].to_numpy(dtype=int) + T - 1
        series.insert(0, "date", returns.index[date_positions].astype(str))
        series.insert(0, "mode", tag)
        null.insert(0, "mode", tag)
        summary.update({"mode": tag, "label": a.label, "N": N, "days": total,
                        "T": T, "P": a.P, "step": a.step, "horizon": a.horizon,
                        "block_size": a.block_size, "shuffles": a.shuffles})
        summary_rows.append(summary)
        all_series[tag] = series
        all_null.append(null)

        print(f"    mean cosine: {summary['mean_cosine']:+.4f} "
              f"(null {summary['null_mean_cosine_mean']:+.4f}, "
              f"p={summary['p_upper_mean_cosine']:.4f})")
        print(f"    positive fraction: {summary['positive_fraction']:.3f} "
              f"(null {summary['null_positive_fraction_mean']:.3f})")
        print(f"    constant-velocity skill vs static: "
              f"{summary['constant_velocity_skill']:+.2%} "
              f"(null {summary['null_constant_velocity_skill_mean']:+.2%}, "
              f"p={summary['p_upper_constant_velocity_skill']:.4f})")

    stem = f"{a.label}_tangent_T{T}_P{a.P}_h{a.horizon}_step{a.step}"
    a.outdir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_series.values(), ignore_index=True).to_csv(
        a.outdir / f"{stem}_series.csv", index=False)
    pd.concat(all_null, ignore_index=True).to_csv(
        a.outdir / f"{stem}_null.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(a.outdir / f"{stem}_summary.csv", index=False)
    print(f"\n  -> {a.outdir / f'{stem}_summary.csv'}")

    if not a.no_plot:
        path = a.outdir / f"{stem}.png"
        plot_series(all_series, a.label, T, a.P, a.horizon, path)
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
