"""Regime 4.6: is tangent persistence distinct from ERSE shrinkage?

The experiment preserves the original Regime 4.4 cosine, then attributes rather
than blindly subtracts it.  At every current leading space Y_t it constructs

    E_t       = Log_{Y_t}(Y_t^ERSE)
    H_minus   = -Log_{Y_t}(Y_{t-h})
    H_plus    =  Log_{Y_t}(Y_{t+h})

and reports (1) the original H_minus/H_plus persistence, (2) alignment of the
future movement with ERSE, (3) persistence after projecting both observed
tangents off E_t, and (4) the share of the covariance transition carried by
top-to-complement off-diagonal blocks, which an eigenvalue-only update in the
current basis cannot reproduce.

The primary specification inherits Regime 4.4: standardised returns, P=3,
horizon=42d, step=14d, T=357 for S&P and 250 elsewhere.  Liu & Liu's primary
ERSE threshold delta=0.25 is frozen; 0.15 and 0.35 are sensitivity runs.  A
21-day block-permuted-return null rebuilds the entire pipeline.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.regime4_4_tangent import (block_permutation_indices,
                                       empirical_upper_p)
from src.data import standardise, to_correlation_panel, to_panel
from src.erse import erse
from src.grassmann import containment_loss, grassmann_log, tangent_cosine
from src.overlap import sample_covariance, spectral


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="nikkei_full")
    p.add_argument("--indir", type=Path, default=Path("data/cache"))
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--T", type=int, default=None,
                   help="window length; default max(N, 250)")
    p.add_argument("--P", type=int, default=3)
    p.add_argument("--step", type=int, default=14)
    p.add_argument("--horizon", type=int, default=42)
    p.add_argument("--block-size", type=int, default=21)
    p.add_argument("--shuffles", type=int, default=99,
                   help="matched null histories; use 0 only for a smoke run")
    p.add_argument("--seed", type=int, default=20260802)
    p.add_argument("--delta", type=float, default=0.25,
                   help="ERSE minimum deviation (paper primary: 0.25)")
    p.add_argument("--mode", choices=("raw", "standardised"),
                   default="standardised")
    return p.parse_args(argv)


def residualise_tangent(tangent, direction, eps=1e-14):
    """Orthogonally remove one tangent direction; return residual and shares."""
    H, E = np.asarray(tangent, dtype=float), np.asarray(direction, dtype=float)
    if H.shape != E.shape:
        raise ValueError(f"tangent shapes differ: {H.shape} vs {E.shape}")
    h2, e2 = float(np.sum(H * H)), float(np.sum(E * E))
    if e2 <= eps:
        return H.copy(), 0.0, 1.0
    coefficient = float(np.sum(H * E) / e2)
    residual = H - coefficient * E
    residual_fraction = float(np.sum(residual * residual) / h2) if h2 > eps else np.nan
    attributed_fraction = max(0.0, min(1.0, 1.0 - residual_fraction))
    return residual, attributed_fraction, residual_fraction


def top_cross_covariance_share(current_basis, current_covariance,
                               future_covariance, eps=1e-20):
    """Fraction of covariance change crossing the current P/complement boundary.

    In the eigenbasis of the current covariance, an eigenvalue-only update is
    diagonal.  The top-to-complement block is therefore a conservative piece of
    the transition that eigenvalue-only shrinkage cannot express.
    """
    U = np.asarray(current_basis, dtype=float)
    C0 = np.asarray(current_covariance, dtype=float)
    C1 = np.asarray(future_covariance, dtype=float)
    if C0.shape != C1.shape or C0.shape[0] != U.shape[0]:
        raise ValueError("basis and covariance dimensions do not agree")
    future_u = C1 @ U
    within = U.T @ future_u
    one_sided = float(np.sum(future_u * future_u) - np.sum(within * within))
    cross_energy = max(0.0, 2.0 * one_sided)
    transition_energy = float(np.sum((C1 - C0) ** 2))
    share = cross_energy / transition_energy if transition_energy > eps else np.nan
    # Cross energy is a subset of the full change in exact arithmetic.
    if np.isfinite(share):
        share = max(0.0, min(1.0, share))
    return cross_energy, transition_energy, share


def window_states(panel, T, step, P, delta, use_standardised):
    """Current and ERSE leading spaces plus covariances for each rolling window."""
    starts, bases, erse_bases, covariances, diagnostics = [], [], [], [], []
    for start in range(0, panel.shape[1] - T + 1, step):
        window = panel[:, start:start + T]
        if use_standardised:
            window = standardise(window, window=1)
        corr_panel = to_correlation_panel(window)
        corr = sample_covariance(corr_panel)
        _, vectors = spectral(corr)
        corrected = erse(corr, delta)
        starts.append(start)
        bases.append(vectors[:, :P])
        erse_bases.append(corrected["corrected_vectors"][:, :P])
        # float32 halves the S&P full-history state from about 460 MB to 230 MB;
        # all decompositions above were computed in float64.
        covariances.append(corr.astype(np.float32))
        diagnostics.append({
            "positive_correlation_fraction": corrected["positive_correlation_fraction"],
            "all_correlations_positive": corrected["all_correlations_positive"],
            "minimum_correlation": corrected["minimum_correlation"],
            "erse_rotations": len(corrected["rotations"]),
            "minimum_deviation_before": float(corrected["deviation_before"].min()),
            "minimum_deviation_after": float(corrected["deviation_after"].min()),
        })
    if not bases:
        raise ValueError(f"panel has {panel.shape[1]} days, shorter than T={T}")
    return (np.asarray(starts), np.asarray(bases), np.asarray(erse_bases),
            np.asarray(covariances), pd.DataFrame(diagnostics))


def attribution_series(starts, bases, erse_bases, covariances, diagnostics,
                       horizon, step):
    """One ERSE attribution row per past/current/future triple."""
    if horizon < step or horizon % step:
        raise ValueError(f"horizon={horizon} must be a positive multiple of step={step}")
    offset = horizon // step
    if len(bases) <= 2 * offset:
        raise ValueError("not enough rolling windows for one past/current/future triple")

    rows = []
    for current in range(offset, len(bases) - offset):
        past = bases[current - offset]
        now = bases[current]
        future = bases[current + offset]
        incoming = -grassmann_log(now, past)
        outgoing = grassmann_log(now, future)
        erse_direction = grassmann_log(now, erse_bases[current])
        incoming_residual, incoming_attr, incoming_left = residualise_tangent(
            incoming, erse_direction)
        outgoing_residual, outgoing_attr, outgoing_left = residualise_tangent(
            outgoing, erse_direction)
        cross, transition, cross_share = top_cross_covariance_share(
            now, covariances[current], covariances[current + offset])
        static_loss = containment_loss(future, now, normalise=True)
        erse_loss = containment_loss(future, erse_bases[current], normalise=True)
        diag = diagnostics.iloc[current]
        rows.append({
            "start": int(starts[current]),
            "past_start": int(starts[current - offset]),
            "future_start": int(starts[current + offset]),
            "original_cosine": tangent_cosine(incoming, outgoing),
            "residual_cosine": tangent_cosine(incoming_residual, outgoing_residual),
            "erse_outgoing_cosine": tangent_cosine(erse_direction, outgoing),
            "erse_incoming_cosine": tangent_cosine(erse_direction, incoming),
            "incoming_erse_attributed_fraction": incoming_attr,
            "outgoing_erse_attributed_fraction": outgoing_attr,
            "incoming_residual_energy_fraction": incoming_left,
            "outgoing_residual_energy_fraction": outgoing_left,
            "incoming_speed": float(np.linalg.norm(incoming, ord="fro")),
            "outgoing_speed": float(np.linalg.norm(outgoing, ord="fro")),
            "erse_speed": float(np.linalg.norm(erse_direction, ord="fro")),
            "static_loss": static_loss,
            "erse_loss": erse_loss,
            "erse_skill": 1.0 - erse_loss / static_loss if static_loss > 0 else np.nan,
            "top_cross_covariance_energy": cross,
            "covariance_transition_energy": transition,
            "top_cross_covariance_share": cross_share,
            **diag.to_dict(),
        })
    return pd.DataFrame(rows)


KEY_STATISTICS = (
    "original_cosine",
    "residual_cosine",
    "erse_outgoing_cosine",
    "outgoing_erse_attributed_fraction",
    "outgoing_residual_energy_fraction",
    "erse_skill",
    "top_cross_covariance_share",
)


def summarise(series):
    """Distribution-aware Regime 4.6 summary."""
    out = {"n_triples": int(len(series))}
    for statistic in KEY_STATISTICS:
        values = series[statistic].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        out[f"mean_{statistic}"] = float(np.mean(values)) if len(values) else np.nan
        out[f"median_{statistic}"] = float(np.median(values)) if len(values) else np.nan
        out[f"q25_{statistic}"] = float(np.quantile(values, 0.25)) if len(values) else np.nan
        out[f"q75_{statistic}"] = float(np.quantile(values, 0.75)) if len(values) else np.nan
    out["all_positive_window_fraction"] = float(
        series["all_correlations_positive"].astype(float).mean())
    out["mean_positive_correlation_fraction"] = float(
        series["positive_correlation_fraction"].mean())
    out["mean_erse_rotations"] = float(series["erse_rotations"].mean())
    return out


def run_mode(panel, T, P, step, horizon, block_size, shuffles, delta, rng,
             use_standardised, progress_label=""):
    observed_states = window_states(panel, T, step, P, delta, use_standardised)
    observed_series = attribution_series(*observed_states, horizon, step)
    observed = summarise(observed_series)

    null_rows = []
    for replicate in range(shuffles):
        indices = block_permutation_indices(panel.shape[1], block_size, rng)
        shuffled = panel[:, indices]
        states = window_states(shuffled, T, step, P, delta, use_standardised)
        row = summarise(attribution_series(*states, horizon, step))
        row["replicate"] = replicate
        null_rows.append(row)
        if shuffles >= 10 and ((replicate + 1) % 10 == 0 or replicate + 1 == shuffles):
            print(f"    {progress_label} null {replicate + 1}/{shuffles}")
    null = pd.DataFrame(null_rows)

    for statistic in ("original_cosine", "residual_cosine",
                      "erse_outgoing_cosine", "top_cross_covariance_share",
                      "erse_skill"):
        key = f"mean_{statistic}"
        values = null[key].to_numpy(dtype=float) if len(null) else np.array([])
        observed[f"null_{key}_mean"] = float(np.mean(values)) if len(values) else np.nan
        observed[f"null_{key}_q95"] = float(np.quantile(values, 0.95)) if len(values) else np.nan
        observed[f"p_upper_{key}"] = empirical_upper_p(observed[key], values)
    return observed_series, observed, null


def main(argv=None):
    a = parse_args(argv)
    if not (0.0 <= a.delta <= 1.0):
        raise SystemExit(f"--delta must lie in [0,1], got {a.delta}")
    if a.shuffles < 0:
        raise SystemExit("--shuffles must be non-negative")
    returns = pd.read_parquet(a.indir / f"{a.label}_returns.parquet")
    panel = to_panel(returns)
    N, total = panel.shape
    T = a.T or max(N, 250)
    if not (0 < a.P < N):
        raise SystemExit(f"need 0 < P < N; got P={a.P}, N={N}")

    use_standardised = a.mode == "standardised"
    print(f"{a.label}: N={N}, {total} days, T={T}, P={a.P}, step={a.step}, "
          f"h={a.horizon}, delta={a.delta}, block={a.block_size}, "
          f"shuffles={a.shuffles}, mode={a.mode}")
    rng = np.random.default_rng(a.seed)
    series, summary, null = run_mode(
        panel, T, a.P, a.step, a.horizon, a.block_size, a.shuffles, a.delta,
        rng, use_standardised, progress_label=a.mode)

    date_positions = series["start"].to_numpy(dtype=int) + T - 1
    series.insert(0, "date", returns.index[date_positions].astype(str))
    summary.update({"label": a.label, "N": N, "days": total, "T": T,
                    "P": a.P, "step": a.step, "horizon": a.horizon,
                    "block_size": a.block_size, "shuffles": a.shuffles,
                    "delta": a.delta, "mode": a.mode})

    print("\n  What is tested: does 4.4 persistence survive ERSE attribution?")
    print(f"  original cosine: {summary['mean_original_cosine']:+.4f} "
          f"(null {summary['null_mean_original_cosine_mean']:+.4f}, "
          f"p={summary['p_upper_mean_original_cosine']:.4f})")
    print(f"  ERSE/outgoing cosine: {summary['mean_erse_outgoing_cosine']:+.4f} "
          f"(null {summary['null_mean_erse_outgoing_cosine_mean']:+.4f}, "
          f"p={summary['p_upper_mean_erse_outgoing_cosine']:.4f})")
    print(f"  outgoing energy attributed to ERSE: "
          f"{summary['mean_outgoing_erse_attributed_fraction']:.1%} "
          f"(median {summary['median_outgoing_erse_attributed_fraction']:.1%})")
    print(f"  residual persistence: {summary['mean_residual_cosine']:+.4f} "
          f"(null {summary['null_mean_residual_cosine_mean']:+.4f}, "
          f"p={summary['p_upper_mean_residual_cosine']:.4f})")
    print(f"  top/complement covariance share: "
          f"{summary['mean_top_cross_covariance_share']:.1%} "
          f"(null {summary['null_mean_top_cross_covariance_share_mean']:.1%}, "
          f"p={summary['p_upper_mean_top_cross_covariance_share']:.4f})")
    print(f"  ERSE forecast skill vs static: {summary['mean_erse_skill']:+.1%} "
          f"(null {summary['null_mean_erse_skill_mean']:+.1%}, "
          f"p={summary['p_upper_mean_erse_skill']:.4f})")
    print(f"  paper all-positive assumption holds in "
          f"{summary['all_positive_window_fraction']:.1%} of windows "
          f"(mean positive pairs {summary['mean_positive_correlation_fraction']:.1%})")

    suffix = str(a.delta).replace(".", "p")
    stem = (f"{a.label}_erse_T{T}_P{a.P}_h{a.horizon}_step{a.step}_"
            f"delta{suffix}_{a.mode}")
    a.outdir.mkdir(parents=True, exist_ok=True)
    series.to_csv(a.outdir / f"{stem}_series.csv", index=False)
    null.to_csv(a.outdir / f"{stem}_null.csv", index=False)
    pd.DataFrame([summary]).to_csv(a.outdir / f"{stem}_summary.csv", index=False)
    print(f"\n  -> {a.outdir / f'{stem}_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
