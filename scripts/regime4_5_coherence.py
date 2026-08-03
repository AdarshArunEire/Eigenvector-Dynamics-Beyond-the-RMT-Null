"""Regime 4.5: is the tangent motion cross-sectionally coherent?

Regime 4.4 asks whether the leading subspace has a predictable direction.
This experiment asks whether its asset-level loading increments move together,
or whether the subspace displacement can be explained by many independent beta
wiggles.

Rolling bases are put in a sequential Procrustes gauge. Their horizon-h
Grassmann logarithms form H[time, asset, factor]. After removing each
asset-factor's time mean, the leading eigenvalue share of

    K = sum_{time,factor} H[:, :, factor] H[:, :, factor]^T

measures the common cross-sectional component. The leading eigenvector's
participation ratio guards against calling a movement in a handful of names
"coherent".

The null independently circular-shifts each asset's tangent history, destroying
synchrony while preserving that asset's exact marginal path. Because row-wise
shifting breaks the horizontal tangent constraint, every surrogate is projected
back onto the tangent space at its original base and rescaled to the observed
per-time speed before it is scored.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.regime4_4_tangent import window_bases
from src.data import to_panel
from src.grassmann import grassmann_log


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="nikkei")
    p.add_argument("--indir", type=Path, default=Path("data/cache"))
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--T", type=int, default=None, help="default max(N, 250)")
    p.add_argument("--P", type=int, default=3)
    p.add_argument("--step", type=int, default=14)
    p.add_argument("--horizon", type=int, default=42)
    p.add_argument("--shuffles", type=int, default=999)
    p.add_argument("--seed", type=int, default=20260803)
    p.add_argument("--mode", choices=("raw", "standardised"),
                   default="standardised")
    return p.parse_args(argv)


def sequential_procrustes(bases):
    """Choose a smooth representative for each point in a subspace path."""
    bases = np.asarray(bases, dtype=float)
    aligned = [bases[0]]
    for current in bases[1:]:
        left, _, right_t = np.linalg.svd(current.T @ aligned[-1], full_matrices=False)
        aligned.append(current @ (left @ right_t))
    return np.asarray(aligned)


def tangent_increments(bases, horizon, step):
    """Horizon-h forward logs and their current bases in a consistent gauge."""
    if horizon < step or horizon % step:
        raise ValueError(f"horizon={horizon} must be a positive multiple of step={step}")
    offset = horizon // step
    aligned = sequential_procrustes(bases)
    if len(aligned) <= offset:
        raise ValueError("not enough bases for one tangent increment")
    H = np.asarray([grassmann_log(aligned[i], aligned[i + offset])
                    for i in range(len(aligned) - offset)])
    return aligned[:-offset], H


def coherence_statistics(tangents):
    """Cross-sectional spike size and how broadly its eigenvector participates."""
    H = np.asarray(tangents, dtype=float)
    if H.ndim != 3:
        raise ValueError(f"tangents must be time x N x P, got {H.shape}")
    centred = H - H.mean(axis=0, keepdims=True)
    Z = centred.transpose(1, 0, 2).reshape(H.shape[1], -1)
    K = (Z @ Z.T) / Z.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.maximum(eigenvalues, 0.0)[::-1]
    lead = eigenvectors[:, -1]
    total = float(eigenvalues.sum())
    share = float(eigenvalues[0] / total) if total > 0 else np.nan
    participation = float(1.0 / np.sum(lead ** 4))
    return {
        "leading_share": share,
        "spike_ratio": float(H.shape[1] * share),
        "leading_participation": participation,
        "participation_fraction": participation / H.shape[1],
        "effective_rank": (float(total ** 2 / np.sum(eigenvalues ** 2))
                           if np.any(eigenvalues) else np.nan),
    }


def desynchronise(tangents, bases, rng):
    """Shift each asset independently, then restore horizontality and speed."""
    H = np.asarray(tangents, dtype=float)
    out = np.empty_like(H)
    for asset in range(H.shape[1]):
        out[:, asset, :] = np.roll(H[:, asset, :],
                                   int(rng.integers(0, H.shape[0])), axis=0)
    for t, U in enumerate(bases):
        original_speed = np.linalg.norm(H[t], ord="fro")
        out[t] -= U @ (U.T @ out[t])
        surrogate_speed = np.linalg.norm(out[t], ord="fro")
        if surrogate_speed > 0:
            out[t] *= original_speed / surrogate_speed
    return out


def empirical_upper_p(observed, null):
    null = np.asarray(null, dtype=float)
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def main(argv=None):
    a = parse_args(argv)
    returns = pd.read_parquet(a.indir / f"{a.label}_returns.parquet")
    panel = to_panel(returns)
    N, total = panel.shape
    T = a.T or max(N, 250)
    if a.horizon % a.step:
        raise SystemExit(f"--horizon {a.horizon} must be divisible by --step {a.step}")

    print(f"{a.label}: N={N}, {total} days, T={T}, P={a.P}, step={a.step}, "
          f"h={a.horizon}, mode={a.mode}, shuffles={a.shuffles}")
    starts, bases = window_bases(panel, T, a.step, a.P,
                                 use_standardised=a.mode == "standardised")
    current_bases, tangents = tangent_increments(bases, a.horizon, a.step)
    observed = coherence_statistics(tangents)

    rng = np.random.default_rng(a.seed)
    null_rows = []
    for replicate in range(a.shuffles):
        row = coherence_statistics(desynchronise(tangents, current_bases, rng))
        row["replicate"] = replicate
        null_rows.append(row)
        if a.shuffles >= 10 and ((replicate + 1) % 100 == 0
                                 or replicate + 1 == a.shuffles):
            print(f"  synchrony null {replicate + 1}/{a.shuffles}")
    null = pd.DataFrame(null_rows)

    for statistic in ("leading_share", "spike_ratio", "participation_fraction"):
        values = null[statistic].to_numpy(dtype=float)
        observed[f"null_{statistic}_mean"] = float(values.mean())
        observed[f"null_{statistic}_q95"] = float(np.quantile(values, 0.95))
        observed[f"p_upper_{statistic}"] = empirical_upper_p(observed[statistic], values)
    observed.update({"label": a.label, "mode": a.mode, "N": N, "days": total,
                     "T": T, "P": a.P, "step": a.step, "horizon": a.horizon,
                     "n_increments": len(tangents), "shuffles": a.shuffles})

    print(f"  leading share: {observed['leading_share']:.4f} "
          f"(null {observed['null_leading_share_mean']:.4f}, "
          f"q95 {observed['null_leading_share_q95']:.4f}, "
          f"p={observed['p_upper_leading_share']:.4f})")
    print(f"  spike / mean eigenvalue: {observed['spike_ratio']:.2f}x "
          f"(null {observed['null_spike_ratio_mean']:.2f}x)")
    print(f"  leading-vector participation: {observed['leading_participation']:.1f}/{N} "
          f"({observed['participation_fraction']:.1%}; null "
          f"{observed['null_participation_fraction_mean']:.1%})")

    a.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{a.label}_coherence_T{T}_P{a.P}_h{a.horizon}_step{a.step}_{a.mode}"
    pd.DataFrame([observed]).to_csv(a.outdir / f"{stem}_summary.csv", index=False)
    null.to_csv(a.outdir / f"{stem}_null.csv", index=False)
    np.savez_compressed(a.outdir / f"{stem}_tangents.npz",
                        starts=starts[:len(tangents)], tangents=tangents)
    print(f"  -> {a.outdir / f'{stem}_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
