"""
Parse + plot EM4 strong-scaling runs produced by run_em4_scaling.sh.

solverScalingTest writes one tab-separated file per run, `solverCtx_WS_<npes>.txt`
(header + a data row), with per-phase wall times (min/mean/max across ranks).
run_em4_scaling.sh renames each to `<scheme>_np<N>.txt` so runs of different
derivative schemes at the same rank count don't collide.

This reads all such files, builds a tidy CSV, and makes the figures that carry
the methods-paper performance argument: compact FD is slower per call but, at
matched accuracy, uses a narrower stencil -> less ghost-exchange communication.
The communication cost is read directly as the gap between the "with comm" and
"compute only" timers:
    comm_unzip = unzip_wcomm - unzip
    comm_zip   = zip_wcomm   - zip
    comm_total = comm_unzip + comm_zip

Figures:
  1. strong_scaling.png  -- evolve time/step vs ranks (per scheme) + ideal 1/p
  2. comm_cost.png       -- comm_total vs ranks (per scheme)
  3. comm_fraction.png   -- comm_total / evolve vs ranks (per scheme)

Usage:
    python scripts/em4_scaling_plot.py [results_dir]
        [--out-dir scripts/em4_scaling_plots] [--csv scripts/em4_scaling.csv]
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict

# columns we lift out of the (45-col) solverScalingTest row
NUM_COLS = ["act_npes", "maxdepth", "numOcts", "dof_cg", "dof_uz",
            "evolve_mean", "unzip_wcomm_mean", "unzip_mean",
            "zip_wcomm_mean", "zip_mean", "rhs_mean", "rhs_blk_mean"]

FNAME_RE = re.compile(r"(?P<scheme>.+)_np(?P<np>\d+)\.txt$")


def read_table(path):
    """Parse a solverScalingTest dump: a banner, then one or more repeated
    `act_npes\\t...` header lines each followed by a data row. Returns a list
    of {col: float} dicts (one per data row), tolerating the leading banner
    and duplicate header lines that the binary interleaves."""
    header = None
    out = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue  # banner / separator lines have no tabs
            fields = line.split("\t")
            if fields[0] == "act_npes":
                header = fields  # (re)capture; trailing '' from trailing tab ok
                continue
            if header is None:
                continue  # data before any header -> ignore
            rec = {}
            for k, v in zip(header, fields):
                if not k:
                    continue
                try:
                    rec[k] = float(v)
                except ValueError:
                    pass
            if all(c in rec for c in NUM_COLS):
                out.append(rec)
    return out


def parse_dir(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_np*.txt"))):
        m = FNAME_RE.search(os.path.basename(path))
        if not m:
            continue
        scheme = m.group("scheme")
        recs = read_table(path)
        if not recs:
            continue
        # average the (per-step) data rows into one point per (scheme, np)
        rec = {"scheme": scheme}
        for c in NUM_COLS:
            rec[c] = sum(r[c] for r in recs) / len(recs)
        rec["np"] = int(round(rec["act_npes"]))
        rec["comm_unzip"] = rec["unzip_wcomm_mean"] - rec["unzip_mean"]
        rec["comm_zip"] = rec["zip_wcomm_mean"] - rec["zip_mean"]
        rec["comm_total"] = rec["comm_unzip"] + rec["comm_zip"]
        rec["comm_frac"] = (rec["comm_total"] / rec["evolve_mean"]
                            if rec["evolve_mean"] > 0 else float("nan"))
        rows.append(rec)
    return rows


def write_csv(rows, path):
    fields = ["scheme", "np", "maxdepth", "numOcts", "dof_uz",
              "evolve_mean", "rhs_mean", "unzip_mean", "unzip_wcomm_mean",
              "zip_mean", "zip_wcomm_mean",
              "comm_unzip", "comm_zip", "comm_total", "comm_frac"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["scheme"], r["np"])):
            w.writerow(r)
    print(f"Wrote {path}")


def by_scheme(rows, ycol):
    """{scheme: ([np...], [y...])} sorted by np, for a given y column."""
    g = defaultdict(list)
    for r in rows:
        g[r["scheme"]].append((r["np"], r[ycol]))
    out = {}
    for s, pts in g.items():
        pts.sort()
        xs, ys = zip(*pts)
        out[s] = (list(xs), list(ys))
    return out


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib unavailable; skipping plots", file=sys.stderr)
        return None


def plot_strong(rows, out_dir):
    plt = _plt()
    if not plt:
        return
    g = by_scheme(rows, "evolve_mean")
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for s, (xs, ys) in sorted(g.items()):
        ax.loglog(xs, ys, marker="o", linewidth=1.4, label=s)
    # ideal strong-scaling reference off the first scheme's first point
    any_s = sorted(g)[0]
    x0, y0 = g[any_s][0][0], g[any_s][1][0]
    xs_ref = sorted({x for _, (xx, _) in g.items() for x in xx})
    ax.loglog(xs_ref, [y0 * x0 / x for x in xs_ref], "k--",
              linewidth=0.9, label="ideal (1/p)")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("evolve time / step  (s, mean over ranks)")
    ax.set_title("EM4 strong scaling")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "strong_scaling.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_y(rows, ycol, ylabel, title, fname, out_dir, logy=True):
    plt = _plt()
    if not plt:
        return
    g = by_scheme(rows, ycol)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for s, (xs, ys) in sorted(g.items()):
        (ax.loglog if logy else ax.semilogx)(
            xs, ys, marker="s", linewidth=1.4, label=s)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, fname)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def print_summary(rows):
    print(f"\n{'scheme':<10}{'np':>6}{'evolve_mean':>14}"
          f"{'comm_total':>13}{'comm_frac':>11}")
    print("-" * 54)
    for r in sorted(rows, key=lambda r: (r["scheme"], r["np"])):
        print(f"{r['scheme']:<10}{r['np']:>6}{r['evolve_mean']:>14.4e}"
              f"{r['comm_total']:>13.4e}{r['comm_frac']:>11.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?",
                    default=os.path.join(os.path.dirname(__file__),
                                         "em4_scaling_results"))
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(__file__),
                                         "em4_scaling_plots"))
    ap.add_argument("--csv",
                    default=os.path.join(os.path.dirname(__file__),
                                         "em4_scaling.csv"))
    args = ap.parse_args()

    rows = parse_dir(args.results_dir)
    if not rows:
        print(f"No *_np*.txt result files in {args.results_dir}",
              file=sys.stderr)
        return 1
    write_csv(rows, args.csv)
    print_summary(rows)
    plot_strong(rows, args.out_dir)
    # linear y: comm = unzip_wcomm - unzip can be ~0 or slightly negative at
    # tiny rank counts (no real ghost exchange); only meaningful at scale.
    plot_y(rows, "comm_total",
           "communication time / step  (s)",
           "Ghost-exchange communication cost (unzip_wcomm - unzip + zip...)",
           "comm_cost.png", args.out_dir, logy=False)
    plot_y(rows, "comm_frac",
           "comm fraction of evolve",
           "Communication as a fraction of evolve time",
           "comm_fraction.png", args.out_dir, logy=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
