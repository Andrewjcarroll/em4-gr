#!/usr/bin/env python3
"""Parse [ifc] interface-norm blocks from EM4 convergence-sweep logs and
print a fixed-structure convergence table with observed orders.

For each (scheme, k) log the LAST [ifc] block is used; the runs are set up
(CFL halved per k, output freq 4^k) so those blocks sit at the same physical
time, which is cross-checked here. Global rms is rebuilt from the per-bin
rms/counts; d1 is the interface-adjacent bin, d>=4 the interior bin.
"""

import math
import re
import sys
from pathlib import Path

BIN_RE = re.compile(r"(d1|d2|d3|d>=4|none) ([0-9.e+-]+) ([0-9.e+-]+) \((\d+)\)")
HDR_RE = re.compile(r"\[ifc\] step (\d+) t ([0-9.e+-]+) elems (\d+)")


def last_block(log_path):
    """Return (t, elems, {qty: {bin: (rms, count)}}) for the final [ifc] block."""
    hdr, rows = None, {}
    cur_hdr, cur_rows = None, {}
    for line in open(log_path, errors="replace"):
        if not line.startswith("[ifc]"):
            continue
        m = HDR_RE.match(line)
        if m:
            if cur_hdr is not None and cur_rows:
                hdr, rows = cur_hdr, cur_rows
            cur_hdr, cur_rows = (int(m[1]), float(m[2]), int(m[3])), {}
            continue
        m = re.match(r"\[ifc\] (\S+)\s+all\s+\|", line)
        if m:
            cur_rows[m[1]] = {b: (float(r), int(c))
                              for b, r, _mx, c in BIN_RE.findall(line)}
    if cur_hdr is not None and cur_rows:
        hdr, rows = cur_hdr, cur_rows
    if hdr is None:
        raise SystemExit(f"no [ifc] block in {log_path}")
    return hdr, rows


def combine(bins, keys):
    s2 = sum(r * r * c for b, (r, c) in bins.items() if b in keys)
    n = sum(c for b, (r, c) in bins.items() if b in keys)
    return math.sqrt(s2 / n) if n else float("nan")


def main():
    argv = sys.argv[1:]
    kmax = 2
    if "--kmax" in argv:
        i = argv.index("--kmax")
        kmax = int(argv[i + 1])
        del argv[i:i + 2]
    outdir = Path(argv[0])
    schemes = [a.split(":")[0] for a in argv[1:]]

    for scheme in schemes:
        data = {}
        for k in range(kmax + 1):
            log = outdir / f"{scheme}_k{k}.log"
            if not log.exists():
                continue
            (step, t, elems), rows = last_block(log)
            data[k] = (t, elems, rows)
        if not data:
            print(f"{scheme}: no logs found")
            continue
        times = {round(v[0], 6) for v in data.values()}
        note = "" if len(times) == 1 else f"  !! times differ: {sorted(times)}"
        print(f"\n=== {scheme} (t = {sorted(times)[0]}){note}")
        print(f"{'k':>2} {'elems':>8} | "
              f"{'|dE| global':>12} {'ord':>5} | {'|dE| interior':>13} {'ord':>5} | "
              f"{'|dE| d1':>12} {'ord':>5}")
        prev = None
        for k in sorted(data):
            _t, elems, rows = data[k]
            de = rows.get("|dE|", {})
            g = combine(de, {"d1", "d2", "d3", "d>=4", "none"})
            ii = combine(de, {"d>=4", "none"})
            d1 = combine(de, {"d1"})
            if prev:
                og, oi, o1 = (math.log2(p / c) if c > 0 and p > 0 else float("nan")
                              for p, c in zip(prev, (g, ii, d1)))
                ords = (f"{og:5.2f}", f"{oi:5.2f}", f"{o1:5.2f}")
            else:
                ords = ("    -",) * 3
            print(f"{k:>2} {elems:>8} | {g:12.4e} {ords[0]} | "
                  f"{ii:13.4e} {ords[1]} | {d1:12.4e} {ords[2]}")
            prev = (g, ii, d1)


if __name__ == "__main__":
    main()
