"""Regime 4.7: is the information in Flag(N; 1, 3, 6) just as valid?

The existing Regimes 4.4--4.6 validate the cumulative top-three Grassmann
space.  This experiment reuses their exact full-history setup and asks whether
the additional nested flag information survives the same three gates:

1. direction persistence against a 21-day block-permuted-return null;
2. cross-asset coherence against independently shifted asset histories; and
3. persistence after attributing and removing the ERSE direction.

The flag is embedded through its nested projectors at dimensions 1, 3 and 6.
The primary ``flag_nested`` tangent is their inverse-dimension-weighted direct
sum.  Separate market [1], core [2:3], buffer [4:6], top-three and top-six
diagnostics ensure that the already-known top-three result cannot make the
additional flag information pass automatically.
"""
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.regime4_4_tangent import (block_permutation_indices,
                                       empirical_upper_p)
from scripts.regime4_5_coherence import (coherence_statistics, desynchronise,
                                         tangent_increments)
from scripts.regime4_6_erse import residualise_tangent
from src.data import standardise, to_correlation_panel, to_panel
from src.erse import erse
from src.flag import (DEFAULT_DIMS, component_logs, flag_component_bases,
                      flag_log, residualise_tuple, stack_nested_tangents,
                      tuple_cosine, tuple_inner)
from src.grassmann import tangent_cosine
from src.overlap import sample_covariance, spectral


INDIVIDUAL_COMPONENTS = (
    "market_1", "block_2_3", "block_4_6", "top_3", "top_6",
)
ALL_COMPONENTS = INDIVIDUAL_COMPONENTS + ("flag_nested",)
# These are the three predeclared tests needed to extend the already-confirmed
# top-three result into Flag(N;1,3,6).  The disjoint blocks explain a result but
# are not extra family-wise gates.
PRIMARY_FLAG_COMPONENTS = ("market_1", "top_6", "flag_nested")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="nikkei_full")
    p.add_argument("--indir", type=Path, default=Path("data/cache"))
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--T", type=int, default=None, help="default max(N, 250)")
    p.add_argument("--step", type=int, default=14)
    p.add_argument("--horizon", type=int, default=42)
    p.add_argument("--block-size", type=int, default=21)
    p.add_argument("--calendar-shuffles", type=int, default=99)
    p.add_argument("--coherence-shuffles", type=int, default=999)
    p.add_argument("--delta", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20260803)
    p.add_argument("--mode", choices=("raw", "standardised"),
                   default="standardised")
    return p.parse_args(argv)


def holm_adjust(pvalues):
    """Holm family-wise adjustment, retaining NaNs."""
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return adjusted
    order = finite[np.argsort(values[finite])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def window_flag_states(panel, T, step, delta, use_standardised):
    """Rolling sample and ERSE Flag(N;1,3,6) frames."""
    starts, frames, erse_frames, diagnostics = [], [], [], []
    q = DEFAULT_DIMS[-1]
    for start in range(0, panel.shape[1] - T + 1, step):
        window = panel[:, start:start + T]
        if use_standardised:
            window = standardise(window, window=1)
        corr_panel = to_correlation_panel(window)
        corr = sample_covariance(corr_panel)
        _, vectors = spectral(corr)
        corrected = erse(corr, delta)
        starts.append(start)
        frames.append(vectors[:, :q])
        erse_frames.append(corrected["corrected_vectors"][:, :q])
        diagnostics.append({
            "positive_correlation_fraction": corrected["positive_correlation_fraction"],
            "all_correlations_positive": corrected["all_correlations_positive"],
            "erse_rotations": len(corrected["rotations"]),
        })
    if not frames:
        raise ValueError(f"panel has {panel.shape[1]} days, shorter than T={T}")
    return (np.asarray(starts), np.asarray(frames), np.asarray(erse_frames),
            pd.DataFrame(diagnostics))


def _individual_triplet_logs(past, now, future, corrected):
    incoming = component_logs(now, past)
    outgoing = component_logs(now, future)
    erse_direction = component_logs(now, corrected)
    return incoming, outgoing, erse_direction


def flag_attribution_series(starts, frames, erse_frames, diagnostics,
                            horizon, step):
    """Long-format persistence and ERSE attribution for every flag component."""
    if horizon < step or horizon % step:
        raise ValueError(f"horizon={horizon} must be a positive multiple of step={step}")
    offset = horizon // step
    if len(frames) <= 2 * offset:
        raise ValueError("not enough rolling windows for one flag triple")

    rows = []
    for current in range(offset, len(frames) - offset):
        past, now, future = (frames[current - offset], frames[current],
                             frames[current + offset])
        corrected = erse_frames[current]
        past_logs, future_logs, erse_logs = _individual_triplet_logs(
            past, now, future, corrected)
        diag = diagnostics.iloc[current].to_dict()

        for component in INDIVIDUAL_COMPONENTS:
            incoming = -past_logs[component]
            outgoing = future_logs[component]
            direction = erse_logs[component]
            incoming_residual, _, _ = residualise_tangent(incoming, direction)
            outgoing_residual, attributed, residual_fraction = residualise_tangent(
                outgoing, direction)
            rows.append({
                "start": int(starts[current]),
                "past_start": int(starts[current - offset]),
                "future_start": int(starts[current + offset]),
                "component": component,
                "cosine": tangent_cosine(incoming, outgoing),
                "positive_cosine": tangent_cosine(incoming, outgoing) > 0,
                "incoming_speed": float(np.linalg.norm(incoming, ord="fro")),
                "outgoing_speed": float(np.linalg.norm(outgoing, ord="fro")),
                "erse_outgoing_cosine": tangent_cosine(direction, outgoing),
                "outgoing_erse_attributed_fraction": attributed,
                "outgoing_residual_energy_fraction": residual_fraction,
                "residual_cosine": tangent_cosine(incoming_residual,
                                                   outgoing_residual),
                **diag,
            })

        incoming_flag = tuple(-part for part in flag_log(now, past))
        outgoing_flag = flag_log(now, future)
        erse_flag = flag_log(now, corrected)
        incoming_residual, _, _ = residualise_tuple(incoming_flag, erse_flag)
        outgoing_residual, attributed, residual_fraction = residualise_tuple(
            outgoing_flag, erse_flag)
        cosine = tuple_cosine(incoming_flag, outgoing_flag)
        rows.append({
            "start": int(starts[current]),
            "past_start": int(starts[current - offset]),
            "future_start": int(starts[current + offset]),
            "component": "flag_nested",
            "cosine": cosine,
            "positive_cosine": cosine > 0,
            "incoming_speed": float(np.sqrt(tuple_inner(incoming_flag, incoming_flag))),
            "outgoing_speed": float(np.sqrt(tuple_inner(outgoing_flag, outgoing_flag))),
            "erse_outgoing_cosine": tuple_cosine(erse_flag, outgoing_flag),
            "outgoing_erse_attributed_fraction": attributed,
            "outgoing_residual_energy_fraction": residual_fraction,
            "residual_cosine": tuple_cosine(incoming_residual, outgoing_residual),
            **diag,
        })
    return pd.DataFrame(rows)


SUMMARY_STATISTICS = (
    "cosine", "residual_cosine", "erse_outgoing_cosine",
    "outgoing_erse_attributed_fraction", "outgoing_residual_energy_fraction",
    "incoming_speed", "outgoing_speed",
)


def summarise_attribution(series):
    """Mean, median and IQR for every flag-component statistic."""
    rows = []
    for component in ALL_COMPONENTS:
        group = series.loc[series["component"] == component]
        row = {"component": component, "n_triples": int(len(group))}
        for statistic in SUMMARY_STATISTICS:
            values = group[statistic].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"mean_{statistic}"] = float(np.mean(values)) if len(values) else np.nan
            row[f"median_{statistic}"] = float(np.median(values)) if len(values) else np.nan
            row[f"q25_{statistic}"] = float(np.quantile(values, .25)) if len(values) else np.nan
            row[f"q75_{statistic}"] = float(np.quantile(values, .75)) if len(values) else np.nan
        row["positive_fraction"] = float(group["positive_cosine"].mean())
        row["all_positive_window_fraction"] = float(
            group["all_correlations_positive"].astype(float).mean())
        row["mean_positive_correlation_fraction"] = float(
            group["positive_correlation_fraction"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def add_calendar_null(observed, null):
    """Attach matched-null summaries, raw p-values and Holm adjustments."""
    out = observed.copy()
    for statistic in ("mean_cosine", "mean_residual_cosine",
                      "mean_erse_outgoing_cosine"):
        means, q95, pvalues = [], [], []
        for _, row in out.iterrows():
            if len(null) and statistic in null:
                values = null.loc[
                    null["component"] == row["component"], statistic
                ].to_numpy(dtype=float)
            else:
                values = np.array([])
            means.append(float(np.mean(values)) if len(values) else np.nan)
            q95.append(float(np.quantile(values, .95)) if len(values) else np.nan)
            pvalues.append(empirical_upper_p(row[statistic], values))
        out[f"null_{statistic}_mean"] = means
        out[f"null_{statistic}_q95"] = q95
        out[f"p_upper_{statistic}"] = pvalues
        primary_mask = out["component"].isin(PRIMARY_FLAG_COMPONENTS).to_numpy()
        adjusted = np.full(len(out), np.nan)
        adjusted[primary_mask] = holm_adjust(
            np.asarray(pvalues, dtype=float)[primary_mask])
        out[f"p_holm_{statistic}"] = adjusted
    return out


def run_calendar_null(panel, T, step, horizon, block_size, shuffles, delta,
                      rng, use_standardised, progress_label=""):
    states = window_flag_states(panel, T, step, delta, use_standardised)
    observed_series = flag_attribution_series(*states, horizon, step)
    observed = summarise_attribution(observed_series)

    null_rows = []
    for replicate in range(shuffles):
        indices = block_permutation_indices(panel.shape[1], block_size, rng)
        shuffled = panel[:, indices]
        shuffled_states = window_flag_states(
            shuffled, T, step, delta, use_standardised)
        summary = summarise_attribution(flag_attribution_series(
            *shuffled_states, horizon, step))
        summary["replicate"] = replicate
        null_rows.append(summary)
        if shuffles >= 10 and ((replicate + 1) % 10 == 0
                               or replicate + 1 == shuffles):
            print(f"    {progress_label} calendar null {replicate + 1}/{shuffles}")
    null = (pd.concat(null_rows, ignore_index=True) if null_rows
            else pd.DataFrame(columns=["component", "replicate"]))
    return states, observed_series, add_calendar_null(observed, null), null


def coherence_inputs(frames, horizon, step):
    """Current bases and forward tangents for individual and nested components."""
    paths = OrderedDict((name, []) for name in INDIVIDUAL_COMPONENTS)
    for frame in frames:
        for name, basis in flag_component_bases(frame).items():
            paths[name].append(basis)
    paths = OrderedDict((name, np.asarray(path)) for name, path in paths.items())

    inputs = OrderedDict()
    for name, path in paths.items():
        bases, tangents = tangent_increments(path, horizon, step)
        inputs[name] = {"bases": (bases,), "tangents": (tangents,),
                        "stacked": tangents}

    nested_bases, nested_tangents = [], []
    for d in DEFAULT_DIMS:
        bases, tangents = tangent_increments(frames[:, :, :d], horizon, step)
        nested_bases.append(bases)
        nested_tangents.append(tangents)
    stacked = np.asarray([stack_nested_tangents(
        tuple(part[t] for part in nested_tangents))
        for t in range(len(nested_tangents[0]))])
    inputs["flag_nested"] = {
        "bases": tuple(nested_bases), "tangents": tuple(nested_tangents),
        "stacked": stacked,
    }
    return inputs


def desynchronise_nested(tangents, bases, rng):
    """One asset shift shared across levels, then restore each flag constraint."""
    outputs = [np.empty_like(part) for part in tangents]
    shifts = rng.integers(0, tangents[0].shape[0], size=tangents[0].shape[1])
    for level, part in enumerate(tangents):
        for asset, shift in enumerate(shifts):
            outputs[level][:, asset, :] = np.roll(part[:, asset, :], int(shift), axis=0)
        for t, U in enumerate(bases[level]):
            original_speed = np.linalg.norm(part[t], ord="fro")
            outputs[level][t] -= U @ (U.T @ outputs[level][t])
            speed = np.linalg.norm(outputs[level][t], ord="fro")
            if speed > 0:
                outputs[level][t] *= original_speed / speed
    return np.asarray([stack_nested_tangents(
        tuple(part[t] for part in outputs)) for t in range(len(outputs[0]))])


def run_coherence(frames, horizon, step, shuffles, rng):
    inputs = coherence_inputs(frames, horizon, step)
    observed_rows = []
    for component, values in inputs.items():
        row = coherence_statistics(values["stacked"])
        row["component"] = component
        observed_rows.append(row)
    observed = pd.DataFrame(observed_rows)

    null_rows = []
    for replicate in range(shuffles):
        for component, values in inputs.items():
            if component == "flag_nested":
                surrogate = desynchronise_nested(
                    values["tangents"], values["bases"], rng)
            else:
                surrogate = desynchronise(
                    values["tangents"][0], values["bases"][0], rng)
            row = coherence_statistics(surrogate)
            row.update({"component": component, "replicate": replicate})
            null_rows.append(row)
        if shuffles >= 10 and ((replicate + 1) % 100 == 0
                               or replicate + 1 == shuffles):
            print(f"    coherence null {replicate + 1}/{shuffles}")
    null = pd.DataFrame(null_rows)

    means, q95, pvalues = [], [], []
    for _, row in observed.iterrows():
        if len(null):
            values = null.loc[
                null["component"] == row["component"], "leading_share"
            ].to_numpy(dtype=float)
        else:
            values = np.array([])
        means.append(float(np.mean(values)) if len(values) else np.nan)
        q95.append(float(np.quantile(values, .95)) if len(values) else np.nan)
        pvalues.append(empirical_upper_p(row["leading_share"], values))
    observed["null_leading_share_mean"] = means
    observed["null_leading_share_q95"] = q95
    observed["p_upper_leading_share"] = pvalues
    primary_mask = observed["component"].isin(PRIMARY_FLAG_COMPONENTS).to_numpy()
    adjusted = np.full(len(observed), np.nan)
    adjusted[primary_mask] = holm_adjust(
        np.asarray(pvalues, dtype=float)[primary_mask])
    observed["p_holm_leading_share"] = adjusted
    return observed, null


def main(argv=None):
    a = parse_args(argv)
    if a.calendar_shuffles < 0 or a.coherence_shuffles < 0:
        raise SystemExit("shuffle counts must be non-negative")
    if not (0 <= a.delta <= 1):
        raise SystemExit("--delta must lie in [0,1]")
    returns = pd.read_parquet(a.indir / f"{a.label}_returns.parquet")
    panel = to_panel(returns)
    N, total = panel.shape
    T = a.T or max(N, 250)
    if N <= DEFAULT_DIMS[-1]:
        raise SystemExit(f"Flag(N;1,3,6) needs N>6, got N={N}")
    use_standardised = a.mode == "standardised"
    print(f"{a.label}: N={N}, {total} days, T={T}, Flag(N;1,3,6), "
          f"step={a.step}, h={a.horizon}, delta={a.delta}, "
          f"calendar={a.calendar_shuffles}, coherence={a.coherence_shuffles}, "
          f"mode={a.mode}")

    rng = np.random.default_rng(a.seed)
    states, series, summary, calendar_null = run_calendar_null(
        panel, T, a.step, a.horizon, a.block_size, a.calendar_shuffles,
        a.delta, rng, use_standardised, progress_label=a.mode)
    coherence, coherence_null = run_coherence(
        states[1], a.horizon, a.step, a.coherence_shuffles, rng)
    summary = summary.merge(coherence, on="component", how="left",
                            suffixes=("", "_coherence"))
    summary.insert(0, "label", a.label)
    summary.insert(1, "N", N)
    summary.insert(2, "days", total)
    summary.insert(3, "T", T)
    summary.insert(4, "step", a.step)
    summary.insert(5, "horizon", a.horizon)
    summary.insert(6, "delta", a.delta)
    summary.insert(7, "mode", a.mode)
    summary.insert(8, "calendar_shuffles", a.calendar_shuffles)
    summary.insert(9, "coherence_shuffles", a.coherence_shuffles)

    date_positions = series["start"].to_numpy(dtype=int) + T - 1
    series.insert(0, "date", returns.index[date_positions].astype(str))
    suffix = str(a.delta).replace(".", "p")
    stem = (f"{a.label}_flag_T{T}_dims1-3-6_h{a.horizon}_step{a.step}_"
            f"delta{suffix}_{a.mode}")
    a.outdir.mkdir(parents=True, exist_ok=True)
    series.to_csv(a.outdir / f"{stem}_series.csv", index=False)
    summary.to_csv(a.outdir / f"{stem}_summary.csv", index=False)
    calendar_null.to_csv(a.outdir / f"{stem}_calendar_null.csv", index=False)
    coherence_null.to_csv(a.outdir / f"{stem}_coherence_null.csv", index=False)

    print("\n  What is tested: is Flag(N;1,3,6) information just as valid?")
    print("  component       persistence (null, p/Holm)  coherence (null, p/Holm)  "
          "ERSE attr  residual (null, p/Holm)")
    for _, row in summary.iterrows():
        holm_cos = (f"{row['p_holm_mean_cosine']:.3f}"
                    if np.isfinite(row['p_holm_mean_cosine']) else "diag")
        holm_coh = (f"{row['p_holm_leading_share']:.3f}"
                    if np.isfinite(row['p_holm_leading_share']) else "diag")
        holm_res = (f"{row['p_holm_mean_residual_cosine']:.3f}"
                    if np.isfinite(row['p_holm_mean_residual_cosine']) else "diag")
        print(f"  {row['component']:<15} "
              f"{row['mean_cosine']:+.4f} ({row['null_mean_cosine_mean']:+.4f}, "
              f"{row['p_upper_mean_cosine']:.3f}/{holm_cos})  "
              f"{row['leading_share']:.1%} ({row['null_leading_share_mean']:.1%}, "
              f"{row['p_upper_leading_share']:.3f}/{holm_coh})  "
              f"{row['mean_outgoing_erse_attributed_fraction']:.2%}  "
              f"{row['mean_residual_cosine']:+.4f} "
              f"({row['null_mean_residual_cosine_mean']:+.4f}, "
              f"{row['p_upper_mean_residual_cosine']:.3f}/{holm_res})")
    print(f"\n  -> {a.outdir / f'{stem}_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
