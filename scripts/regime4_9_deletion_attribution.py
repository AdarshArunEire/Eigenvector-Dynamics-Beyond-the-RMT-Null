"""Regime 4.9: does persistence survive deterministic rolling deletion?

For a current T-day window, the oldest ``horizon`` observations are known to
leave before the target window is formed.  Their removal defines the retained
Flag B_t.  The primary target is then the realised addition tangent

    A_t = Log_{B_t}(F_{t+h}),

where F_{t+h} is the full future Flag after the unknown incoming block arrives.
The preceding realised addition tangent is transported from B_{t-h} to B_t and
compared with A_t.  Retained Window is exactly the zero-addition forecast.

Two return-level nulls rebuild the entire construction.  The calendar null is
the established intact 21-day block permutation.  The volatility-matched null
permutes intact 42-day blocks only within realised-volatility strata, retaining
the coarse volatility-regime sequence while destroying the identity and exact
cross-sectional organisation of the arriving blocks.
"""
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.regime4_4_tangent import (block_permutation_indices,
                                       empirical_upper_p)
from scripts.regime4_7_flag import INDIVIDUAL_COMPONENTS
from src.data import standardise, to_correlation_panel, to_panel
from src.flag import (component_logs, flag_log, residualise_tuple,
                      tuple_cosine, tuple_inner)
from src.grassmann import tangent_cosine
from src.oracle_line import align_flag_frame
from src.overlap import sample_covariance


ALL_COMPONENTS = INDIVIDUAL_COMPONENTS + ("flag_nested",)
DEFAULT_LABELS = ("sp500_full", "nikkei_full", "dax_full", "cac40_full")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", choices=DEFAULT_LABELS, default="nikkei_full")
    parser.add_argument("--indir", type=Path, default=Path("data/cache"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("results/regime4_9"))
    parser.add_argument("--T", type=int, default=None,
                        help="default max(N, 250)")
    parser.add_argument("--step", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=42)
    parser.add_argument("--calendar-block-size", type=int, default=21)
    parser.add_argument("--volatility-block-size", type=int, default=42)
    parser.add_argument("--volatility-bins", type=int, default=5)
    parser.add_argument("--shuffles", type=int, default=99)
    parser.add_argument("--null", choices=("all", "calendar", "volatility"),
                        default="all")
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args(argv)


def volatility_matched_block_indices(panel, block_size, bins, rng):
    """Permute blocks within realised-volatility strata.

    Positions retain their volatility stratum.  The final short block is only
    exchanged with blocks of the same length, so the result is always an exact
    permutation of the original columns.
    """
    panel = np.asarray(panel, dtype=float)
    if panel.ndim != 2 or panel.shape[1] < 1:
        raise ValueError("panel must have shape (assets, positive time)")
    if block_size < 1 or bins < 1:
        raise ValueError("block_size and bins must be positive")
    blocks = [np.arange(start, min(start + block_size, panel.shape[1]))
              for start in range(0, panel.shape[1], block_size)]
    order = np.arange(len(blocks))
    lengths = np.asarray([len(block) for block in blocks])
    scores = np.asarray([
        float(np.mean(panel[:, block] ** 2)) for block in blocks
    ])
    for length in np.unique(lengths):
        eligible = np.flatnonzero(lengths == length)
        if len(eligible) < 2:
            continue
        ranked = eligible[np.argsort(scores[eligible], kind="stable")]
        for group in np.array_split(ranked, min(int(bins), len(ranked))):
            if len(group) > 1:
                order[group] = rng.permutation(group)
    return np.concatenate([blocks[index] for index in order])


def _leading_flag(panel):
    """Top-six Flag of one already daily-standardised return window."""
    adjusted = to_correlation_panel(panel)
    correlation = sample_covariance(adjusted)
    n = correlation.shape[0]
    if n < 7:
        raise ValueError("Regime 4.9 requires at least seven assets")
    _, vectors = eigh(correlation, subset_by_index=[n - 6, n - 1],
                      check_finite=False, driver="evr")
    return vectors[:, ::-1]


def flag_histories(panel, T, step, horizon):
    """Full T-day Flags and their deletion-adjusted retained Flags."""
    panel = np.asarray(panel, dtype=float)
    if not 0 < horizon < T - 1:
        raise ValueError("horizon must leave at least two retained observations")
    if step < 1 or horizon % step:
        raise ValueError("horizon must be a positive multiple of step")
    # For window=1 the normalising mean is a common scalar, which cancels in
    # the subsequent row standardisation.  Computing it once is therefore
    # exactly equivalent at the correlation level and avoids repeating work.
    daily = standardise(panel, window=1)
    starts, full, retained = [], [], []
    for start in range(0, panel.shape[1] - T + 1, step):
        starts.append(start)
        full.append(_leading_flag(daily[:, start:start + T]))
        retained.append(_leading_flag(
            daily[:, start + horizon:start + T]))
    if len(full) <= 2 * (horizon // step):
        raise ValueError("not enough rolling windows for one triplet")
    return np.asarray(starts), np.asarray(full), np.asarray(retained)


def _rotate(matrix, source, companion, cosine, sine):
    """Apply one plane rotation to an ambient-by-width matrix."""
    source_coordinates = source @ matrix
    companion_coordinates = companion @ matrix
    return (matrix
            + (cosine - 1.0) * (
                np.outer(source, source_coordinates)
                + np.outer(companion, companion_coordinates))
            + sine * (
                np.outer(companion, source_coordinates)
                - np.outer(source, companion_coordinates)))


def transport_objects(source, target, objects, atol=1e-10):
    """Ordered orthogonal Flag transport without constructing a dense matrix."""
    source = np.asarray(source, dtype=float)
    aligned = align_flag_frame(source, target)
    frame = np.array(source, copy=True)
    outputs = [np.array(value, copy=True) for value in objects]
    fixed = []
    identity = np.eye(source.shape[0])
    for column in range(source.shape[1]):
        origin = frame[:, column]
        origin /= np.linalg.norm(origin)
        destination = aligned[:, column]
        destination /= np.linalg.norm(destination)
        cosine = float(np.clip(origin @ destination, -1.0, 1.0))
        sine = float(np.sqrt(max(0.0, 1.0 - cosine ** 2)))
        if sine <= atol and cosine > 0:
            fixed.append(destination)
            continue
        if sine <= atol:
            candidates = identity.copy()
            if fixed:
                fixed_frame = np.column_stack(fixed)
                candidates -= fixed_frame @ (fixed_frame.T @ candidates)
            candidates -= origin[:, None] * (origin @ candidates)[None, :]
            norms = np.linalg.norm(candidates, axis=0)
            index = int(np.argmax(norms))
            if norms[index] <= atol:
                raise ValueError("cannot transport an antipodal Flag")
            companion = candidates[:, index] / norms[index]
            cosine, sine = -1.0, 0.0
        else:
            companion = (destination - cosine * origin) / sine
        frame = _rotate(frame, origin, companion, cosine, sine)
        outputs = [_rotate(value, origin, companion, cosine, sine)
                   for value in outputs]
        fixed.append(destination)
    if not np.allclose(frame, aligned, atol=2e-8):
        raise RuntimeError("ordered transport did not reach the retained Flag")
    return aligned, outputs


def _residualise(tangent, direction, eps=1e-14):
    tangent = np.asarray(tangent, dtype=float)
    direction = np.asarray(direction, dtype=float)
    h2 = float(np.sum(tangent * tangent))
    d2 = float(np.sum(direction * direction))
    if d2 <= eps:
        return tangent.copy(), 0.0
    coefficient = float(np.sum(tangent * direction)) / d2
    residual = tangent - coefficient * direction
    attributed = (1.0 - float(np.sum(residual * residual)) / h2
                  if h2 > eps else np.nan)
    return residual, float(np.clip(attributed, 0.0, 1.0))


def deletion_attribution_series(starts, full, retained, horizon, step):
    """Residual-addition persistence and deletion attribution per origin."""
    offset = horizon // step
    rows = []
    for current in range(offset, len(full) - offset):
        past_full, now, future = (full[current - offset], full[current],
                                  full[current + offset])
        previous_base, current_base = (retained[current - offset],
                                       retained[current])

        previous_components = component_logs(previous_base, now)
        previous_nested = flag_log(previous_base, now)
        names = list(previous_components)
        objects = ([previous_components[name] for name in names]
                   + list(previous_nested))
        aligned_base, transported = transport_objects(
            previous_base, current_base, objects)
        transported_components = OrderedDict(
            (name, transported[index]) for index, name in enumerate(names))
        transported_nested = tuple(transported[len(names):])

        next_components = component_logs(aligned_base, future)
        next_nested = flag_log(aligned_base, future)
        past_logs = component_logs(now, past_full)
        future_logs = component_logs(now, future)
        deletion_logs = component_logs(now, current_base)

        for name in names:
            incoming = -past_logs[name]
            outgoing = future_logs[name]
            deletion = deletion_logs[name]
            incoming_residual, _ = _residualise(incoming, deletion)
            outgoing_residual, attributed = _residualise(outgoing, deletion)
            rows.append({
                "start": int(starts[current]), "component": name,
                "addition_cosine": tangent_cosine(
                    transported_components[name], next_components[name]),
                "addition_previous_speed": float(np.linalg.norm(
                    transported_components[name])),
                "addition_next_speed": float(np.linalg.norm(next_components[name])),
                "full_cosine": tangent_cosine(incoming, outgoing),
                "deletion_outgoing_cosine": tangent_cosine(deletion, outgoing),
                "deletion_attributed_fraction": attributed,
                "full_residual_cosine": tangent_cosine(
                    incoming_residual, outgoing_residual),
            })

        incoming = tuple(-part for part in flag_log(now, past_full))
        outgoing = flag_log(now, future)
        deletion = flag_log(now, current_base)
        incoming_residual, _, _ = residualise_tuple(incoming, deletion)
        outgoing_residual, attributed, _ = residualise_tuple(outgoing, deletion)
        rows.append({
            "start": int(starts[current]), "component": "flag_nested",
            "addition_cosine": tuple_cosine(transported_nested, next_nested),
            "addition_previous_speed": float(np.sqrt(tuple_inner(
                transported_nested, transported_nested))),
            "addition_next_speed": float(np.sqrt(tuple_inner(
                next_nested, next_nested))),
            "full_cosine": tuple_cosine(incoming, outgoing),
            "deletion_outgoing_cosine": tuple_cosine(deletion, outgoing),
            "deletion_attributed_fraction": attributed,
            "full_residual_cosine": tuple_cosine(
                incoming_residual, outgoing_residual),
        })
    return pd.DataFrame(rows)


def summarise(series):
    """Effect sizes for the predeclared complete Flag and diagnostics."""
    rows = []
    for component in ALL_COMPONENTS:
        group = series.loc[series["component"] == component]
        addition = group["addition_cosine"].to_numpy(dtype=float)
        addition = addition[np.isfinite(addition)]
        rows.append({
            "component": component,
            "n_origins": int(len(group)),
            "n_finite_addition": int(len(addition)),
            "mean_addition_cosine": float(np.mean(addition)),
            "median_addition_cosine": float(np.median(addition)),
            "q25_addition_cosine": float(np.quantile(addition, .25)),
            "q75_addition_cosine": float(np.quantile(addition, .75)),
            "positive_addition_fraction": float(np.mean(addition > 0)),
            "mean_addition_previous_speed": float(
                group["addition_previous_speed"].mean()),
            "mean_addition_next_speed": float(
                group["addition_next_speed"].mean()),
            "mean_full_cosine": float(group["full_cosine"].mean()),
            "mean_deletion_outgoing_cosine": float(
                group["deletion_outgoing_cosine"].mean()),
            "mean_deletion_attributed_fraction": float(
                group["deletion_attributed_fraction"].mean()),
            "mean_full_residual_cosine": float(
                group["full_residual_cosine"].mean()),
        })
    return pd.DataFrame(rows)


def _completed(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


def _append(path, frame):
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _null_indices(panel, kind, args, replicate):
    tag = 491 if kind == "calendar" else 492
    rng = np.random.default_rng(np.random.SeedSequence(
        [args.seed, tag, replicate]))
    if kind == "calendar":
        return block_permutation_indices(
            panel.shape[1], args.calendar_block_size, rng)
    return volatility_matched_block_indices(
        panel, args.volatility_block_size, args.volatility_bins, rng)


def run_null(panel, T, args, kind, stem):
    path = args.outdir / f"{stem}_{kind}_null.csv"
    existing = _completed(path)
    done = set(existing["replicate"].astype(int)) if len(existing) else set()
    for replicate in range(args.shuffles):
        if replicate in done:
            continue
        indices = _null_indices(panel, kind, args, replicate)
        starts, full, retained = flag_histories(
            panel[:, indices], T, args.step, args.horizon)
        summary = summarise(deletion_attribution_series(
            starts, full, retained, args.horizon, args.step))
        summary.insert(0, "replicate", replicate)
        summary.insert(1, "null", kind)
        _append(path, summary)
        print(f"    {kind} null {replicate + 1}/{args.shuffles}", flush=True)
    return _completed(path)


def attach_nulls(observed, nulls):
    out = observed.copy()
    for kind, null in nulls.items():
        means, q95s, pvalues = [], [], []
        for _, row in out.iterrows():
            values = null.loc[
                null["component"] == row["component"],
                "mean_addition_cosine"].to_numpy(dtype=float)
            means.append(float(np.mean(values)))
            q95s.append(float(np.quantile(values, .95)))
            pvalues.append(empirical_upper_p(
                row["mean_addition_cosine"], values))
        out[f"{kind}_null_mean"] = means
        out[f"{kind}_null_q95"] = q95s
        out[f"{kind}_p"] = pvalues
    return out


def main(argv=None):
    args = parse_args(argv)
    returns = pd.read_parquet(args.indir / f"{args.label}_returns.parquet")
    panel = to_panel(returns)
    N, days = panel.shape
    T = args.T or max(N, 250)
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.label}_T{T}_h{args.horizon}_step{args.step}"
    print(f"{args.label}: N={N}, days={days}, T={T}, h={args.horizon}, "
          f"step={args.step}, shuffles={args.shuffles}", flush=True)

    starts, full, retained = flag_histories(
        panel, T, args.step, args.horizon)
    series = deletion_attribution_series(
        starts, full, retained, args.horizon, args.step)
    date_positions = series["start"].to_numpy(dtype=int) + T - 1
    series.insert(0, "date", returns.index[date_positions].astype(str))
    series.to_csv(args.outdir / f"{stem}_series.csv", index=False)
    observed = summarise(series)

    kinds = (("calendar", "volatility") if args.null == "all"
             else (args.null,))
    nulls = {kind: run_null(panel, T, args, kind, stem) for kind in kinds}
    result = attach_nulls(observed, nulls)
    result.insert(0, "label", args.label)
    result.insert(1, "N", N)
    result.insert(2, "days", days)
    result.insert(3, "T", T)
    result.insert(4, "horizon", args.horizon)
    result.insert(5, "step", args.step)
    result.to_csv(args.outdir / f"{stem}_summary.csv", index=False)

    columns = ["component", "mean_full_cosine",
               "mean_deletion_attributed_fraction",
               "mean_full_residual_cosine", "mean_addition_cosine"]
    for kind in kinds:
        columns.extend([f"{kind}_null_mean", f"{kind}_p"])
    print(result[columns].to_string(index=False,
          float_format=lambda value: f"{value:.4f}"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
