"""Unattended runner for the alarm ladder: Alarm 2 window sweep, then Alarm 3.

    python run_alarms.py --quick     # ~5 min smoke test. RUN THIS FIRST.
    python run_alarms.py             # the real thing

Everything underneath is idempotent and checkpointed per replicate, so this
script is safe to interrupt and safe to re-run: completed work is skipped, not
recomputed, and no seed changes on resume.  A stage that fails is recorded and
the run continues, so one broken cell cannot cost a whole night.

Stages
    0  preflight     imports, cached panels, output paths
    1  alarm2 sp500  the (T, h) window sweep, both arms, full replication
    2  alarm2 others Nikkei/DAX/CAC at the matched corrected window
    3  alarm2 collect the observed / null-mean / null-sd curves
    4  alarm3 power   empirical minimum detectable excess, then the synthetic
                      detection floor in rotation-speed units

Rough cost at full replication on a modern desktop, threaded BLAS: stage 1
dominates at 1.5-3 h (seven cells x 99 replicates at N=357), stage 2 about
30-50 min, stage 4 about 40-90 min, stages 0 and 3 seconds.  Small-N cells are
an order of magnitude cheaper than S&P ones.

Note stage 1 re-runs the published T=357 cell into the sweep directory.  That is
about 25 minutes of deliberate duplication: the seeds are identical to the
published run, so it doubles as a reproducibility check on
``results/regime4_9``.  Pass ``--skip-anchor`` to drop it.
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SWEEP_DIR = ROOT / "results" / "alarm2_window_sweep"
POWER_DIR = ROOT / "results" / "alarm3_detection_power"
RUN_DIR = ROOT / "results" / "alarm_runs"
LOG_DIR = RUN_DIR / "logs"
PANELS = ("sp500_full", "nikkei_full", "dax_full", "cac40_full")
CORRECTED_T = 750
HORIZON = 42


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shuffles", type=int, default=99,
                        help="null replicates per cell (default 99)")
    parser.add_argument("--only", default="all",
                        help="comma-separated stage numbers, e.g. 0,4")
    parser.add_argument("--skip-anchor", action="store_true",
                        help="do not re-run the published T=357 cell")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: tiny replication, short synthetic "
                             "history, one small panel. Verifies every code "
                             "path in a few minutes.")
    return parser.parse_args(argv)


def _stamp():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def run(name, arguments, log_name):
    """Run one child process, streaming to console and to its own log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{log_name}.log"
    command = [sys.executable, *arguments]
    print(f"\n[{_stamp()}] >>> {name}\n      {' '.join(str(c) for c in command)}",
          flush=True)
    started = time.time()
    tail = []
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"\n===== {datetime.now(timezone.utc).isoformat()} "
                     f"{' '.join(str(c) for c in command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            errors="replace")
        for line in process.stdout:
            handle.write(line)
            tail.append(line.rstrip())
            del tail[:-40]
            print("      " + line.rstrip(), flush=True)
        process.wait()
    elapsed = time.time() - started
    ok = process.returncode == 0
    if not ok:
        print(f"[{_stamp()}] !!! {name} FAILED (exit {process.returncode}) "
              f"after {elapsed / 60:.1f} min. Last lines:", flush=True)
        for line in tail[-15:]:
            print("      | " + line, flush=True)
        print(f"      full log: {log_path}", flush=True)
    else:
        print(f"[{_stamp()}] <<< {name} ok in {elapsed / 60:.1f} min",
              flush=True)
    return {"stage": name, "ok": ok, "returncode": process.returncode,
            "minutes": round(elapsed / 60, 2), "log": str(log_path)}


# --------------------------------------------------------------------- stages

def stage_preflight(args):
    """Fail fast and cheaply rather than three hours in."""
    print(f"\n[{_stamp()}] >>> stage 0 preflight", flush=True)
    problems = []
    try:
        import numpy, scipy, pandas  # noqa: F401
        print(f"      numpy {numpy.__version__}  scipy {scipy.__version__}  "
              f"pandas {pandas.__version__}", flush=True)
    except ImportError as exc:
        problems.append(f"missing dependency: {exc}")
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError:
            problems.append("no parquet engine (pip install pyarrow)")
    for panel in PANELS:
        path = ROOT / "data" / "cache" / f"{panel}_returns.parquet"
        if not path.exists():
            problems.append(f"missing cached panel {path}")
    for script in ("scripts/alarm2_window_sweep.py",
                   "scripts/alarm2_null_chunk.py",
                   "scripts/alarm3_detection_power.py",
                   "scripts/regime4_9_deletion_attribution.py"):
        if not (ROOT / script).exists():
            problems.append(f"missing {script}")
    for directory in (SWEEP_DIR, POWER_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    if problems:
        print(f"[{_stamp()}] !!! preflight FAILED", flush=True)
        for problem in problems:
            print("      - " + problem, flush=True)
        return {"stage": "stage 0 preflight", "ok": False, "returncode": 1,
                "minutes": 0.0, "log": ""}
    print(f"      panels present, scripts present, outputs writable")
    print(f"[{_stamp()}] <<< stage 0 preflight ok", flush=True)
    return {"stage": "stage 0 preflight", "ok": True, "returncode": 0,
            "minutes": 0.0, "log": ""}


def stage_sp500_sweep(args):
    cells = "all"
    if args.quick:
        cells = "500:42"
    elif args.skip_anchor:
        cells = "500:42,750:42,1008:42,500:56,750:84,1008:112"
    return run("stage 1 alarm2 sp500 window sweep",
               ["scripts/alarm2_window_sweep.py", "--label", "sp500_full",
                "--mode", "full", "--shuffles", str(args.shuffles),
                "--cells", cells, "--outdir", str(SWEEP_DIR)],
               "stage1_alarm2_sp500")


def stage_other_panels(args):
    results = []
    panels = ("cac40_full",) if args.quick else PANELS[1:]
    for panel in panels:
        results.append(run(
            f"stage 2 alarm2 {panel} T={CORRECTED_T}",
            ["scripts/regime4_9_deletion_attribution.py", "--label", panel,
             "--T", str(CORRECTED_T), "--horizon", str(HORIZON),
             "--shuffles", str(args.shuffles), "--null", "volatility",
             "--outdir", str(SWEEP_DIR)],
            f"stage2_alarm2_{panel}"))
    return results


def stage_collect(args):
    return run("stage 3 alarm2 collect",
               ["scripts/alarm2_window_sweep.py", "--label", "sp500_full",
                "--mode", "collect", "--outdir", str(SWEEP_DIR),
                "--cells", "500:42" if args.quick else "all"],
               "stage3_alarm2_collect")


def stage_power(args):
    arguments = ["scripts/alarm3_detection_power.py", "--mode", "all",
                 "--outdir", str(POWER_DIR),
                 "--scan", str(ROOT / "results" / "regime4_9"), str(SWEEP_DIR)]
    if args.quick:
        arguments.append("--quick")
    return run("stage 4 alarm3 detection power", arguments,
               "stage4_alarm3_power")


STAGES = {
    "0": stage_preflight,
    "1": stage_sp500_sweep,
    "2": stage_other_panels,
    "3": stage_collect,
    "4": stage_power,
}


# -------------------------------------------------------------------- summary

def final_summary():
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78, flush=True)
    pd.set_option("display.width", 220)
    for title, path, columns in (
        ("Alarm 2 window sweep",
         SWEEP_DIR / "sp500_full_sweep_collected.csv",
         ["arm", "T", "horizon", "q", "deletion_fraction",
          "observed_addition_cosine", "null_mean", "null_sd", "excess", "z",
          "p", "n_null"]),
        ("Alarm 3a empirical minimum detectable excess",
         POWER_DIR / "empirical_mde.csv",
         ["cell", "n_null", "null_mean", "null_sd", "mde_q95_excess",
          "observed", "excess", "z", "powered_ratio"]),
        ("Alarm 3b synthetic detection floor",
         POWER_DIR / "synthetic_mde.csv",
         ["tag", "N", "T", "q", "null_mean", "null_sd",
          "detection_threshold", "mde_omega", "mde_horizon_degrees",
          "mde_addition_speed"]),
    ):
        print(f"\n--- {title}")
        if not path.exists():
            print(f"    not produced ({path.name} absent)")
            continue
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, OSError):
            print(f"    {path.name} unreadable or empty")
            continue
        available = [c for c in columns if c in frame.columns]
        print(frame[available].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))


def main(argv=None):
    args = parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    wanted = (list(STAGES) if args.only == "all"
              else [token.strip() for token in args.only.split(",")])
    unknown = [token for token in wanted if token not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; known: {list(STAGES)}")
    if args.quick:
        args.shuffles = min(args.shuffles, 6)
        print("QUICK smoke test: 6 null replicates, one S&P cell, one small "
              "panel, short synthetic history.\nNothing here is quotable -- it "
              "only proves every code path runs.", flush=True)

    started = time.time()
    print(f"[{_stamp()}] alarm ladder starting; shuffles={args.shuffles}; "
          f"stages={','.join(wanted)}", flush=True)
    records = []
    for token in wanted:
        outcome = STAGES[token](args)
        records.extend(outcome if isinstance(outcome, list) else [outcome])
        if token == "0" and not records[-1]["ok"]:
            print("\npreflight failed; stopping before any long stage.",
                  flush=True)
            break

    status = pd.DataFrame(records)
    status.to_csv(RUN_DIR / "status.csv", index=False)
    final_summary()

    print("\n" + "=" * 78)
    print(f"stages run in {(time.time() - started) / 60:.1f} min")
    print(status[["stage", "ok", "minutes"]].to_string(index=False))
    failed = status.loc[~status["ok"], "stage"].tolist()
    if failed:
        print(f"\nFAILED: {len(failed)} stage(s) -> {failed}")
        print(f"logs in {LOG_DIR}")
        print("Everything is resumable: fix the cause and re-run the same "
              "command; completed replicates are skipped.")
    else:
        print("\nall stages ok")
    print(f"status: {RUN_DIR / 'status.csv'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
