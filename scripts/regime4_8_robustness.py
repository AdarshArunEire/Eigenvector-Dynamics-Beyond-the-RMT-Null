"""Regime 4.8: return-level negative controls for the Stage 2 Flag signal.

4.8A independently IAAFT-randomises every asset return history.  Each marginal
distribution is exact and each linear spectrum is approximately retained, but
the original cross-sectional timing is removed.  The complete Flag persistence
is then rebuilt from returns.

4.8B removes an internally observed equal-weight market factor by rolling OLS,
then rebuilds the residual correlation matrices and Flag history.  Its calendar
null permutes raw multivariate return blocks first and repeats the entire factor
estimation.  Its coherence null independently shifts residual tangent histories.

Both expensive ensembles checkpoint one completed replicate per CSV row and
resume without changing seeds.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.regime4_4_tangent import (block_permutation_indices,
                                       empirical_upper_p)
from scripts.regime4_7_flag import (INDIVIDUAL_COMPONENTS, run_coherence)
from src.data import standardise, to_correlation_panel, to_panel
from src.flag import component_logs, flag_log, tuple_cosine, tuple_inner
from src.grassmann import tangent_cosine
from src.overlap import sample_covariance
from src.surrogates import (independent_iaaft_panel,
                            remove_equal_weight_factor)


ALL_COMPONENTS = INDIVIDUAL_COMPONENTS + ("flag_nested",)


def leading_seven(correlation):
    """Largest seven eigenpairs, descending, without a full decomposition."""
    n = correlation.shape[0]
    values, vectors = eigh(correlation, subset_by_index=[n - 7, n - 1],
                           check_finite=False, driver="evr")
    return values[::-1], vectors[:, ::-1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default="nikkei_full")
    p.add_argument("--indir", type=Path, default=Path("data/cache"))
    p.add_argument("--outdir", type=Path, default=Path("results/regime4_8"))
    p.add_argument("--test", choices=("phase", "market-residual", "all"),
                   default="all")
    p.add_argument("--T", type=int, default=None, help="default max(N, 250)")
    p.add_argument("--step", type=int, default=14)
    p.add_argument("--horizon", type=int, default=42)
    p.add_argument("--block-size", type=int, default=21)
    p.add_argument("--phase-surrogates", type=int, default=99)
    p.add_argument("--calendar-shuffles", type=int, default=99)
    p.add_argument("--coherence-shuffles", type=int, default=999)
    p.add_argument("--iaaft-iterations", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260803)
    return p.parse_args(argv)


def rolling_flag_frames(panel, T, step, remove_market=False):
    """Complete top-six frames, re-estimating every operation per window."""
    starts, frames, diagnostics = [], [], []
    for start in range(0, panel.shape[1] - T + 1, step):
        window = panel[:, start:start + T]
        if remove_market:
            window = remove_equal_weight_factor(window)
        window = standardise(window, window=1)
        corr_panel = to_correlation_panel(window)
        values, vectors = leading_seven(sample_covariance(corr_panel))
        starts.append(start)
        frames.append(vectors[:, :6])
        row = {}
        for d in (1, 3, 6):
            row[f"relative_gap_{d}"] = float(
                (values[d - 1] - values[d]) / max(abs(values[d - 1]), 1e-15))
        diagnostics.append(row)
    if not frames:
        raise ValueError(f"panel has {panel.shape[1]} days, shorter than T={T}")
    return np.asarray(starts), np.asarray(frames), pd.DataFrame(diagnostics)


def flag_persistence_series(starts, frames, horizon, step):
    """Incoming/outgoing tangent alignment for every Flag diagnostic level."""
    if horizon < step or horizon % step:
        raise ValueError("horizon must be a positive multiple of step")
    offset = horizon // step
    if len(frames) <= 2 * offset:
        raise ValueError("not enough rolling windows for one flag triple")
    rows = []
    for current in range(offset, len(frames) - offset):
        past, now, future = (frames[current - offset], frames[current],
                             frames[current + offset])
        past_logs = component_logs(now, past)
        future_logs = component_logs(now, future)
        for component in INDIVIDUAL_COMPONENTS:
            incoming, outgoing = -past_logs[component], future_logs[component]
            cosine = tangent_cosine(incoming, outgoing)
            rows.append({"start": int(starts[current]), "component": component,
                         "cosine": cosine, "positive_cosine": cosine > 0,
                         "incoming_speed": float(np.linalg.norm(incoming)),
                         "outgoing_speed": float(np.linalg.norm(outgoing))})
        incoming = tuple(-part for part in flag_log(now, past))
        outgoing = flag_log(now, future)
        cosine = tuple_cosine(incoming, outgoing)
        rows.append({"start": int(starts[current]), "component": "flag_nested",
                     "cosine": cosine, "positive_cosine": cosine > 0,
                     "incoming_speed": float(np.sqrt(tuple_inner(incoming, incoming))),
                     "outgoing_speed": float(np.sqrt(tuple_inner(outgoing, outgoing)))})
    return pd.DataFrame(rows)


def summarise_persistence(series):
    rows = []
    for component in ALL_COMPONENTS:
        group = series.loc[series["component"] == component]
        cosine = group["cosine"].to_numpy(dtype=float)
        cosine = cosine[np.isfinite(cosine)]
        rows.append({
            "component": component, "n_triples": len(cosine),
            "mean_cosine": float(np.mean(cosine)),
            "median_cosine": float(np.median(cosine)),
            "q25_cosine": float(np.quantile(cosine, .25)),
            "q75_cosine": float(np.quantile(cosine, .75)),
            "positive_fraction": float(np.mean(cosine > 0)),
            "mean_incoming_speed": float(group["incoming_speed"].mean()),
            "mean_outgoing_speed": float(group["outgoing_speed"].mean()),
        })
    return pd.DataFrame(rows)


def _seed(seed, experiment, replicate):
    tag = 481 if experiment == "phase" else 482
    return np.random.default_rng(np.random.SeedSequence([seed, tag, replicate]))


def _completed(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


def _append_checkpoint(path, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _attach_null(observed, null):
    out = observed.copy()
    means, medians, q25, q75, q95, pvalues = [], [], [], [], [], []
    for _, row in out.iterrows():
        values = null.loc[null["component"] == row["component"],
                          "mean_cosine"].to_numpy(dtype=float)
        means.append(float(np.mean(values)) if len(values) else np.nan)
        medians.append(float(np.median(values)) if len(values) else np.nan)
        q25.append(float(np.quantile(values, .25)) if len(values) else np.nan)
        q75.append(float(np.quantile(values, .75)) if len(values) else np.nan)
        q95.append(float(np.quantile(values, .95)) if len(values) else np.nan)
        pvalues.append(empirical_upper_p(row["mean_cosine"], values))
    out["null_mean_cosine_mean"] = means
    out["null_mean_cosine_median"] = medians
    out["null_mean_cosine_q25"] = q25
    out["null_mean_cosine_q75"] = q75
    out["null_mean_cosine_q95"] = q95
    out["p_upper_mean_cosine"] = pvalues
    return out


def _surrogate_diagnostics(original, surrogate):
    original_corr = np.corrcoef(original)
    surrogate_corr = np.corrcoef(surrogate)
    upper = np.triu_indices(original.shape[0], 1)
    spectral_errors, lag_errors = [], []
    for x, y in zip(original, surrogate):
        target = np.abs(np.fft.rfft(x))[1:]
        actual = np.abs(np.fft.rfft(y))[1:]
        spectral_errors.append(np.linalg.norm(actual - target) /
                               max(np.linalg.norm(target), 1e-15))
        lag_x = np.corrcoef(x[:-1], x[1:])[0, 1]
        lag_y = np.corrcoef(y[:-1], y[1:])[0, 1]
        lag_errors.append(abs(lag_x - lag_y))
    return {
        "mean_abs_cross_correlation": float(np.mean(np.abs(surrogate_corr[upper]))),
        "original_mean_abs_cross_correlation": float(np.mean(np.abs(original_corr[upper]))),
        "median_relative_spectrum_error": float(np.median(spectral_errors)),
        "median_abs_lag1_error": float(np.nanmedian(lag_errors)),
    }


def run_phase(panel, T, args, stem):
    starts, frames, observed_diag = rolling_flag_frames(panel, T, args.step)
    series = flag_persistence_series(starts, frames, args.horizon, args.step)
    observed = summarise_persistence(series)
    null_path = args.outdir / f"{stem}_phase_null.csv"
    existing = _completed(null_path)
    done = set(existing["replicate"].astype(int)) if len(existing) else set()
    for replicate in range(args.phase_surrogates):
        if replicate in done:
            continue
        surrogate = independent_iaaft_panel(
            panel, _seed(args.seed, "phase", replicate),
            max_iter=args.iaaft_iterations)
        s_starts, s_frames, gaps = rolling_flag_frames(
            surrogate, T, args.step)
        summary = summarise_persistence(flag_persistence_series(
            s_starts, s_frames, args.horizon, args.step))
        shared = {"replicate": replicate, **_surrogate_diagnostics(panel, surrogate)}
        for d in (1, 3, 6):
            shared[f"mean_relative_gap_{d}"] = float(gaps[f"relative_gap_{d}"].mean())
        _append_checkpoint(null_path, [dict(row, **shared)
                                      for row in summary.to_dict("records")])
        print(f"    phase surrogate {replicate + 1}/{args.phase_surrogates}", flush=True)
    null = _completed(null_path)
    summary = _attach_null(observed, null)
    for d in (1, 3, 6):
        summary[f"observed_mean_relative_gap_{d}"] = float(
            observed_diag[f"relative_gap_{d}"].mean())
    series.to_csv(args.outdir / f"{stem}_phase_series.csv", index=False)
    summary.to_csv(args.outdir / f"{stem}_phase_summary.csv", index=False)
    return summary


def run_market_residual(panel, T, args, stem):
    starts, frames, gaps = rolling_flag_frames(
        panel, T, args.step, remove_market=True)
    series = flag_persistence_series(starts, frames, args.horizon, args.step)
    observed = summarise_persistence(series)
    null_path = args.outdir / f"{stem}_market_residual_calendar_null.csv"
    existing = _completed(null_path)
    done = set(existing["replicate"].astype(int)) if len(existing) else set()
    for replicate in range(args.calendar_shuffles):
        if replicate in done:
            continue
        rng = _seed(args.seed, "market-residual", replicate)
        indices = block_permutation_indices(panel.shape[1], args.block_size, rng)
        s_starts, s_frames, s_gaps = rolling_flag_frames(
            panel[:, indices], T, args.step, remove_market=True)
        summary = summarise_persistence(flag_persistence_series(
            s_starts, s_frames, args.horizon, args.step))
        shared = {"replicate": replicate}
        for d in (1, 3, 6):
            shared[f"mean_relative_gap_{d}"] = float(
                s_gaps[f"relative_gap_{d}"].mean())
        _append_checkpoint(null_path, [dict(row, **shared)
                                      for row in summary.to_dict("records")])
        print(f"    residual calendar null {replicate + 1}/{args.calendar_shuffles}",
              flush=True)
    null = _completed(null_path)
    summary = _attach_null(observed, null)
    coherence, coherence_null = run_coherence(
        frames, args.horizon, args.step, args.coherence_shuffles,
        _seed(args.seed, "market-residual", 1_000_000))
    summary = summary.merge(coherence, on="component", how="left")
    for d in (1, 3, 6):
        summary[f"observed_mean_relative_gap_{d}"] = float(
            gaps[f"relative_gap_{d}"].mean())
    series.to_csv(args.outdir / f"{stem}_market_residual_series.csv", index=False)
    coherence_null.to_csv(
        args.outdir / f"{stem}_market_residual_coherence_null.csv", index=False)
    summary.to_csv(args.outdir / f"{stem}_market_residual_summary.csv", index=False)
    return summary


def _print_primary(name, summary):
    row = summary.loc[summary["component"] == "flag_nested"].iloc[0]
    print(f"\n  {name}: complete Flag")
    print(f"    observed cosine {row['mean_cosine']:+.4f}; "
          f"null {row['null_mean_cosine_mean']:+.4f}; "
          f"p={row['p_upper_mean_cosine']:.4f}")
    if "leading_share" in row:
        print(f"    coherent share {row['leading_share']:.2%}; "
              f"shift null {row['null_leading_share_mean']:.2%}; "
              f"p={row['p_upper_leading_share']:.4f}")


def main(argv=None):
    args = parse_args(argv)
    for count in (args.phase_surrogates, args.calendar_shuffles,
                  args.coherence_shuffles):
        if count < 0:
            raise SystemExit("surrogate counts must be non-negative")
    returns = pd.read_parquet(args.indir / f"{args.label}_returns.parquet")
    panel = to_panel(returns)
    N, days = panel.shape
    T = args.T or max(N, 250)
    if N <= 6:
        raise SystemExit(f"Flag(N;1,3,6) requires N>6, got N={N}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.label}_T{T}_h{args.horizon}_step{args.step}"
    print(f"{args.label}: N={N}, days={days}, T={T}, test={args.test}")
    if args.test in ("phase", "all"):
        _print_primary("4.8A IAAFT", run_phase(panel, T, args, stem))
    if args.test in ("market-residual", "all"):
        _print_primary("4.8B market residual", run_market_residual(
            panel, T, args, stem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
