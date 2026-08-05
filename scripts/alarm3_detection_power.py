"""Alarm 3: what rotation can this instrument actually detect?

Regime 4.9 reported four panel verdicts without ever stating the effect size the
apparatus can resolve.  With 99 replicates and a null sd near 0.013 the minimum
detectable excess is roughly 0.020-0.025, while DAX's observed excess was 0.0053
and CAC's was zero.  Those two panels could not have registered an effect of the
size apparently present, whatever the truth, so "one of four passed" is not a
statement about markets.

This module produces the missing sensitivity curve in two halves.

EMPIRICAL.  Every volatility-null distribution already on disk yields a minimum
detectable excess directly: the threshold an observed value must clear to reach
p <= 0.05.  Reported as ``mde_normal`` (null mean + 1.645 sd, stable at low
replicate counts) and ``mde_q95`` (the empirical quantile, which is what the
p-value actually uses).  The ratio observed/threshold says whether a cell was
ever powered.  Costs nothing and is pure post-processing.

SYNTHETIC.  The empirical half cannot say what true rotation corresponds to a
given excess, because the real truth is unknown.  So inject a known one.  A
Haar basis is rotated at constant angular velocity ``omega`` radians per day in
``planes`` disjoint planes, each carrying one top-six direction into a direction
just outside the top six:

    Q_t = Q_0 G_t,   G_t mixing column k with column 6+k by angle omega*t

The population top-six subspace therefore rotates at constant velocity, so the
population addition tangent direction is *constant* and the true addition cosine
is 1 by construction.  Returns are drawn as r_t = Q_0 G_t sqrt(lam) z_t, which
costs one dense matmul for the whole history rather than one per day.

The null needs no permutation machinery here: ``omega = 0`` is a world with
exactly zero rotation, so the spread of the measured statistic across
independent omega=0 draws *is* the null distribution, exact by construction
rather than approximated by a surrogate.

The minimum detectable rotation is then the smallest omega whose measured mean
addition cosine clears the omega=0 threshold.  It is reported three ways:

    omega                 radians per day
    horizon_degrees       42*omega in degrees -- the subspace rotation over one
                          forecast horizon, which is the interpretable unit
    addition_speed        the measured tangent norm, directly comparable to the
                          ``addition_next_speed`` column the real Regime 4.9
                          series already records for every panel

That last one closes the loop: it puts the synthetic detection floor and the
real panels' observed motion in the same units, so "powered" or "underpowered"
becomes a comparison rather than an assertion.

The default spectrum sets the sixth eigenvalue just above the bulk edge so the
top-6/7 relative gap lands near 0.10, matching the 0.068-0.115 measured on the
real panels.  A synthetic world with a comfortable gap would overstate the
instrument's real sensitivity.

Every row is checkpointed, so an interrupted sweep resumes without repeating
work and without changing any seed.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.regime4_9_deletion_attribution import (
    deletion_attribution_series, flag_histories, summarise)
from src.synth import factor_spectrum, spd_from_spectrum

PRIMARY = "flag_nested"
DEFAULT_TOP = (25.0, 10.0, 6.0, 4.5, 3.0, 1.55)
DEFAULT_BULK = (1.40, 0.40)

# (N, T, horizon, tag).  The published cells and their matched T=750
# counterparts, so the sweep answers "was the published verdict powered, and is
# the corrected window powered".
DEFAULT_CELLS = (
    (23, 250, 42, "cac published"),
    (23, 750, 42, "cac corrected"),
    (29, 250, 42, "dax published"),
    (29, 750, 42, "dax corrected"),
    (131, 250, 42, "nikkei published"),
    (131, 750, 42, "nikkei corrected"),
    (357, 357, 42, "sp500 published"),
    (357, 750, 42, "sp500 corrected"),
)
DEFAULT_OMEGAS = (0.0, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("empirical", "synthetic", "all"),
                        default="all")
    parser.add_argument("--outdir", type=Path,
                        default=Path("results/alarm3_detection_power"))
    parser.add_argument("--scan", type=Path, nargs="*",
                        default=[Path("results/regime4_9"),
                                 Path("results/alarm2_window_sweep")],
                        help="directories of existing *_volatility_null.csv")
    parser.add_argument("--step", type=int, default=14)
    parser.add_argument("--days", type=int, default=6678,
                        help="synthetic history length; the real panels run "
                             "6629-6828 days")
    parser.add_argument("--null-replicates", type=int, default=24)
    parser.add_argument("--signal-replicates", type=int, default=3)
    parser.add_argument("--planes", type=int, default=6,
                        help="how many top-six directions rotate outward")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: short history, two small cells, few "
                             "replicates. Verifies the path before a long run.")
    return parser.parse_args(argv)


# ----------------------------------------------------------------- synthetic

def rotating_panel(N, days, omega, rng, top=DEFAULT_TOP, bulk=DEFAULT_BULK,
                   planes=6):
    """Returns whose population top-six subspace rotates at constant velocity.

    ``omega = 0`` is a static world and is the exact null for this experiment.
    """
    top = tuple(float(value) for value in top)
    if N < 2 * len(top) + 1:
        raise ValueError(f"N={N} too small for {len(top)} rotating planes")
    planes = int(min(planes, len(top)))
    if planes < 1:
        raise ValueError("planes must be positive")
    spectrum = factor_spectrum(N, top, bulk_hi=bulk[0], bulk_lo=bulk[1])
    _, basis = spd_from_spectrum(spectrum, rng)

    coordinates = np.sqrt(spectrum)[:, None] * rng.standard_normal((N, days))
    angles = float(omega) * np.arange(days)
    cosines, sines = np.cos(angles), np.sin(angles)
    for plane in range(planes):
        inner, outer = plane, len(top) + plane
        a = coordinates[inner].copy()
        b = coordinates[outer].copy()
        coordinates[inner] = cosines * a - sines * b
        coordinates[outer] = sines * a + cosines * b
    return basis @ coordinates


def measure(panel, T, horizon, step):
    """Primary Regime 4.9 statistic on one synthetic history."""
    starts, full, retained = flag_histories(panel, T, step, horizon)
    series = deletion_attribution_series(starts, full, retained, horizon, step)
    row = summarise(series).set_index("component").loc[PRIMARY]
    return {
        "addition_cosine": float(row["mean_addition_cosine"]),
        "addition_speed": float(row["mean_addition_next_speed"]),
        "full_cosine": float(row["mean_full_cosine"]),
        "deletion_attributed_fraction":
            float(row["mean_deletion_attributed_fraction"]),
        "n_origins": int(row["n_origins"]),
    }


def _done_keys(path):
    if not path.exists():
        return set()
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return set()
    if not len(frame):
        return set()
    return set(zip(frame["N"], frame["T"], frame["horizon"],
                   frame["omega"].round(12), frame["replicate"]))


def synthetic_sweep(args, cells, omegas):
    """Measured statistic for every (cell, omega, replicate), checkpointed."""
    path = args.outdir / "synthetic_runs.csv"
    done = _done_keys(path)
    for N, T, horizon, tag in cells:
        for omega in omegas:
            repeats = (args.null_replicates if omega == 0.0
                       else args.signal_replicates)
            for replicate in range(repeats):
                key = (N, T, horizon, round(float(omega), 12), replicate)
                if key in done:
                    continue
                rng = np.random.default_rng(np.random.SeedSequence(
                    [args.seed, 493, N, T, horizon,
                     int(round(float(omega) * 1e9)), replicate]))
                panel = rotating_panel(N, args.days, omega, rng,
                                       planes=args.planes)
                row = {"N": N, "T": T, "horizon": horizon, "tag": tag,
                       "omega": float(omega), "replicate": replicate,
                       "q": N / T, "deletion_fraction": horizon / T,
                       "horizon_degrees": np.degrees(float(omega) * horizon),
                       **measure(panel, T, horizon, args.step)}
                pd.DataFrame([row]).to_csv(
                    path, mode="a", header=not path.exists(), index=False)
                print(f"    N={N} T={T} h={horizon} omega={omega:g} "
                      f"rep={replicate + 1}/{repeats} "
                      f"cos={row['addition_cosine']:+.4f}", flush=True)
    return pd.read_csv(path)


def synthetic_thresholds(runs):
    """Per-cell detection floor from the exact omega=0 null."""
    rows = []
    for (N, T, horizon), group in runs.groupby(["N", "T", "horizon"],
                                               sort=True):
        null = group.loc[group["omega"] == 0.0, "addition_cosine"].to_numpy(float)
        null = null[np.isfinite(null)]
        if len(null) < 3:
            continue
        mean, sd = float(np.mean(null)), float(np.std(null, ddof=1))
        threshold = mean + 1.645 * sd
        signal = group.loc[group["omega"] > 0.0].groupby("omega").agg(
            measured=("addition_cosine", "mean"),
            speed=("addition_speed", "mean"),
            degrees=("horizon_degrees", "first")).reset_index()
        detected = signal.loc[signal["measured"] > threshold]
        first = detected.iloc[0] if len(detected) else None
        rows.append({
            "N": N, "T": T, "horizon": horizon, "q": N / T,
            "deletion_fraction": horizon / T,
            "tag": group["tag"].iloc[0],
            "n_null": len(null),
            "null_mean": mean, "null_sd": sd,
            "detection_threshold": threshold,
            "mde_omega": float(first["omega"]) if first is not None else np.nan,
            "mde_horizon_degrees":
                float(first["degrees"]) if first is not None else np.nan,
            "mde_addition_speed":
                float(first["speed"]) if first is not None else np.nan,
            "omegas_tested": len(signal),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- empirical

def _observed_lookup(directories):
    """Observed primary statistic per stem, from summaries or series files."""
    observed = {}
    for directory in directories:
        for path in sorted(Path(directory).glob("*_summary.csv")):
            try:
                frame = pd.read_csv(path)
            except (pd.errors.EmptyDataError, OSError):
                continue
            if "component" not in frame or "mean_addition_cosine" not in frame:
                continue
            primary = frame.loc[frame["component"] == PRIMARY]
            if not len(primary):
                continue
            stem = path.name.replace("_summary.csv", "")
            observed[stem] = {
                "observed": float(primary.iloc[0]["mean_addition_cosine"]),
                "addition_speed": float(primary.iloc[0].get(
                    "mean_addition_next_speed", np.nan)),
            }
    return observed


def empirical_table(args):
    """Minimum detectable excess for every null distribution on disk."""
    observed = _observed_lookup(args.scan)
    rows = []
    for directory in args.scan:
        for path in sorted(Path(directory).glob("*_volatility_null.csv")):
            try:
                frame = pd.read_csv(path)
            except (pd.errors.EmptyDataError, OSError):
                continue
            keys = [n for n in ("replicate", "component") if n in frame]
            if keys:
                frame = frame.drop_duplicates(subset=keys, keep="first")
            values = frame.loc[frame["component"] == PRIMARY,
                               "mean_addition_cosine"].to_numpy(float)
            values = values[np.isfinite(values)]
            if len(values) < 3:
                continue
            stem = path.name.replace("_volatility_null.csv", "")
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            row = {
                "source": str(directory), "cell": stem, "n_null": len(values),
                "null_mean": mean, "null_sd": sd,
                "null_q95": float(np.quantile(values, .95)),
                "mde_normal_excess": 1.645 * sd,
                "mde_q95_excess": float(np.quantile(values, .95)) - mean,
                "p_floor": 1.0 / (len(values) + 1),
            }
            if stem in observed:
                row["observed"] = observed[stem]["observed"]
                row["excess"] = row["observed"] - mean
                row["z"] = row["excess"] / sd if sd > 0 else np.nan
                row["powered_ratio"] = (row["excess"] / row["mde_q95_excess"]
                                        if row["mde_q95_excess"] > 0 else np.nan)
            rows.append(row)
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values(["source", "cell"]).reset_index(drop=True)
    return table


# ---------------------------------------------------------------------- main

def main(argv=None):
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    cells, omegas = DEFAULT_CELLS, DEFAULT_OMEGAS
    if args.quick:
        cells = ((23, 250, 42, "cac published"), (29, 250, 42, "dax published"))
        omegas = (0.0, 1e-3, 5e-3)
        args.days = min(args.days, 1800)
        args.null_replicates = min(args.null_replicates, 5)
        args.signal_replicates = min(args.signal_replicates, 2)
        print("QUICK smoke test: reduced cells, history and replicates.",
              flush=True)

    pd.set_option("display.width", 220)

    if args.mode in ("empirical", "all"):
        table = empirical_table(args)
        table.to_csv(args.outdir / "empirical_mde.csv", index=False)
        print("\n=== Alarm 3a: empirical minimum detectable excess ===",
              flush=True)
        if len(table):
            columns = [c for c in ("cell", "n_null", "null_mean", "null_sd",
                                   "mde_q95_excess", "observed", "excess", "z",
                                   "powered_ratio", "p_floor")
                       if c in table]
            print(table[columns].to_string(
                index=False, float_format=lambda v: f"{v:.4f}"))
            if "powered_ratio" in table:
                weak = table.loc[table["powered_ratio"] < 1.0, "cell"]
                print(f"\n  underpowered cells (excess below the p<=0.05 "
                      f"threshold): {len(weak)}")
                for cell in weak:
                    print(f"    {cell}")
        else:
            print("  no null distributions found under "
                  + ", ".join(str(d) for d in args.scan))

    if args.mode in ("synthetic", "all"):
        print("\n=== Alarm 3b: synthetic detection floor ===", flush=True)
        runs = synthetic_sweep(args, cells, omegas)
        thresholds = synthetic_thresholds(runs)
        thresholds.to_csv(args.outdir / "synthetic_mde.csv", index=False)
        print("\nDetection floor per cell (omega=0 is the exact null)")
        if len(thresholds):
            print(thresholds[[
                "tag", "N", "T", "q", "deletion_fraction", "n_null",
                "null_mean", "null_sd", "detection_threshold", "mde_omega",
                "mde_horizon_degrees", "mde_addition_speed"]].to_string(
                index=False, float_format=lambda v: f"{v:.4f}"))
            print("\n  mde_horizon_degrees is the top-six subspace rotation "
                  "over one 42-day horizon that this cell can just resolve.")
            print("  mde_addition_speed is directly comparable to "
                  "mean_addition_next_speed in the real Regime 4.9 series.")
        else:
            print("  not enough omega=0 replicates yet to set a threshold")

    print(f"\nWritten to {args.outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
