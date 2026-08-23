"""
Plot in-simulation accuracy from EM4 ANALYTICAL_DIFF output.

run_em4_accuracy.sh evolves the dipole initial data for several derivative
schemes (E4, E6, JTT6, ...) with EM4 compiled `-DEM4_COMPUTE_ANALYTICAL=ON`,
which writes <prefix>_ANALYTICAL_DIFF.csv (columns: Timestep, Time, then per
output variable {VAR}_DIFF_{MIN,MAX,L2,RMSE,NRMSE,MAE,L2_INT}). The runner
renames each to <scheme>_diff.csv.

This reproduces the paper's in-simulation error figures (Ex_error / Bx_error):
the L2 norm of the difference between the numerical and analytic fields vs time,
one curve per scheme. It is a DEMONSTRATION that the compact operator works in a
live evolution and tracks the explicit schemes -- the deeper convergence /
constraint / long-time-stability study belongs to the companion paper.

Usage:
    python scripts/em4_accuracy_plot.py [results_dir]
        [--vars U_E0,U_B0] [--metric DIFF_L2]
        [--out-dir scripts/em4_accuracy_plots]
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict

FNAME_RE = re.compile(r"(?P<scheme>.+)_diff\.csv$")


def read_runs(results_dir):
    """{scheme: list-of-row-dicts} from <scheme>_diff.csv files."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_diff.csv"))):
        m = FNAME_RE.search(os.path.basename(path))
        if not m:
            continue
        rows = []
        with open(path, newline="") as f:
            for raw in csv.DictReader(f):
                rows.append({k.strip(): v for k, v in raw.items()
                             if k is not None})
        if rows:
            runs[m.group("scheme")] = rows
    return runs


def series(rows, col, sqrt=False, volume=None):
    """(times, values) for a column, dropping non-numeric/zero-noise rows.

    With sqrt=True the value is sqrt'd -- used for the volume-integrated column
    DIFF_L2_INT, which EM4 stores as the integral of diff^2 over the mesh
    (calculateL2FullMeshIntegration), i.e. integral diff^2 dV.

      - sqrt only            -> (integral diff^2 dV)^(1/2)  [volume-weighted L2,
                                unnormalized; fine for comparing schemes since
                                the domain volume is the same for all]
      - sqrt + volume=V      -> (integral diff^2 dV / V)^(1/2)  [matches the
                                rms_vol convention BSSN/CCZ4 report: pass the
                                domain volume V = Lx*Ly*Lz]
    """
    ts, vs = [], []
    for r in rows:
        t = r.get("Time")
        v = r.get(col)
        if t is None or v is None:
            continue
        try:
            tf, vf = float(t), float(v)
        except ValueError:
            continue
        if not (math.isfinite(tf) and math.isfinite(vf)):
            continue
        if sqrt:
            if vf < 0:
                continue
            if volume:
                vf = vf / volume
            vf = math.sqrt(vf)
        ts.append(tf)
        vs.append(vf)
    return ts, vs


def available_vars(runs, metric):
    """variables that actually have a {var}_{metric} column in the data."""
    cols = set()
    for rows in runs.values():
        if rows:
            cols |= set(rows[0].keys())
    suffix = "_" + metric
    return sorted({c[:-len(suffix)] for c in cols if c.endswith(suffix)})


def plot_var(runs, var, metric, out_dir, sqrt=False, volume=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plot", file=sys.stderr)
        return
    col = f"{var}_{metric}"
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    plotted = 0
    for scheme in sorted(runs):
        ts, vs = series(runs[scheme], col, sqrt=sqrt, volume=volume)
        # keep positive values for the log axis
        pts = [(t, v) for t, v in zip(ts, vs) if v > 0]
        if len(pts) < 2:
            continue
        ts2, vs2 = zip(*pts)
        ax.semilogy(ts2, vs2, linewidth=1.4, label=scheme)
        plotted += 1
    if not plotted:
        plt.close(fig)
        print(f"  (no data for {col})", file=sys.stderr)
        return
    norm_label = ("volume-weighted L2" if sqrt else metric.replace("DIFF_", ""))
    ax.set_xlabel("time")
    ax.set_ylabel(f"{var} {norm_label} error")
    ax.set_title(f"In-simulation error: {var}"
                 + (" (mesh-integrated)" if sqrt else ""))
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{var}_error.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def print_summary(runs, varlist, metric, sqrt=False, volume=None):
    label = f"sqrt({metric})" if sqrt else metric
    print(f"\nFinal-time {label} per scheme (volume-weighted)"
          if sqrt else f"\nFinal-time {label} per scheme")
    head = f"{'scheme':<10}" + "".join(f"{v:>16}" for v in varlist)
    print(head)
    print("-" * len(head))
    for scheme in sorted(runs):
        rows = runs[scheme]
        line = f"{scheme:<10}"
        for v in varlist:
            ts, vs = series(rows, f"{v}_{metric}", sqrt=sqrt, volume=volume)
            line += f"{(vs[-1] if vs else float('nan')):>16.4e}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?",
                    default=os.path.join(os.path.dirname(__file__),
                                         "em4_accuracy_results"))
    ap.add_argument("--vars", default="U_E0,U_B0",
                    help="comma list of variables to plot (default U_E0,U_B0)")
    # default to the volume-integrated (mesh-aware) norm: on an AMR mesh the
    # plain per-node DIFF_L2 weights fine and coarse cells equally, which biases
    # comparisons across schemes/resolutions whose meshes differ. DIFF_L2_INT is
    # EM4's integral of diff^2 over element volumes (shared dendrolib
    # calculateL2FullMeshIntegration); we sqrt it to a proper L2 norm.
    ap.add_argument("--metric", default="DIFF_L2_INT",
                    help="error column suffix (DIFF_L2_INT [volume-aware, "
                         "default], DIFF_L2, DIFF_RMSE, ...)")
    # EM4 now writes DIFF_L2_INT as the finished volume-averaged norm
    # rms_vol = sqrt(int diff^2 dV / V) directly (solverCtx.cpp, BSSN/CCZ4
    # convention), so by default we plot it as-is. Use --raw-integrated only
    # for older EM4 builds whose DIFF_L2_INT held the raw integral diff^2 dV.
    ap.add_argument("--raw-integrated", action="store_true",
                    help="treat DIFF_L2_INT as the raw integral diff^2 dV "
                         "(pre-rms_vol EM4 builds): sqrt it, optionally "
                         "dividing by --volume V first")
    ap.add_argument("--volume", type=float, default=None,
                    help="with --raw-integrated: domain volume V=Lx*Ly*Lz to "
                         "divide by, giving rms_vol = sqrt(int diff^2 dV / V)")
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(__file__),
                                         "em4_accuracy_plots"))
    args = ap.parse_args()
    sqrt = args.raw_integrated and args.metric.endswith("L2_INT")

    runs = read_runs(args.results_dir)
    if not runs:
        print(f"No *_diff.csv files in {args.results_dir}", file=sys.stderr)
        return 1

    want = [v.strip() for v in args.vars.split(",") if v.strip()]
    have = available_vars(runs, args.metric)
    varlist = [v for v in want if v in have]
    if not varlist:
        print(f"None of {want} have a _{args.metric} column. "
              f"Available: {have}", file=sys.stderr)
        return 1

    print_summary(runs, varlist, args.metric, sqrt=sqrt, volume=args.volume)
    for v in varlist:
        plot_var(runs, v, args.metric, args.out_dir, sqrt=sqrt, volume=args.volume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
