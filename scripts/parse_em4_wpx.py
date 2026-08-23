#!/usr/bin/env python3
"""Summarise EM4 wpx A/B logs: interface-local error (d1 vs d>=4 control),
global error vs analytic, element counts, and wall time per step.

usage: parse_em4_wpx.py LOG [LOG ...]
"""
import re
import sys
import os

IFC_HDR = re.compile(r"\[ifc\] step (\d+) t ([\d.eE+-]+)(?: elems (\d+) wall ([\d.]+))?")
IFC_ROW = re.compile(r"\[ifc\] (\S+)\s+(lvl\s+(\d+)|all)\s+\|(.*)$")
CELL = re.compile(r"\s*(d1|d2|d3|d>=4|none) ([\d.eE+-]+) ([\d.eE+-]+) \((\d+)\)")
ELEM = re.compile(r"(?:old mesh|new mesh|number of elements|Elements|elements)\s*[:=]?\s*(\d+)", re.I)


def parse(path):
    steps = {}  # step -> {qty -> {('lvl',l) or 'all' -> {bin: (rms,max,cnt)}}}
    cur = None
    elems = []
    steptimes = []
    remesh = []
    with open(path, errors="replace") as f:
        for line in f:
            m = IFC_HDR.search(line)
            if m:
                cur = (int(m.group(1)), float(m.group(2)))
                steps.setdefault(cur, {})
                if m.group(3):
                    steps[cur]["_elems"] = int(m.group(3))
                    steps[cur]["_wall"] = float(m.group(4))
                continue
            m = IFC_ROW.search(line)
            if m and cur is not None:
                qty = m.group(1)
                key = "all" if m.group(2) == "all" else int(m.group(3))
                cells = {}
                for c in CELL.finditer(m.group(4)):
                    cells[c.group(1)] = (float(c.group(2)), float(c.group(3)), int(c.group(4)))
                steps[cur].setdefault(qty, {})[key] = cells
                continue
            if "Remesh triggered" in line:
                remesh.append(line.strip())
            mm = re.search(r"old mesh\s*:\s*(\d+)\s*new mesh\s*:\s*(\d+)", line)
            if mm:
                elems.append((int(mm.group(1)), int(mm.group(2))))
            mt = re.search(r"\[ETS\].*?step.*?(\d+).*?(?:time|wall|elapsed)[^\d]*([\d.]+)", line)
    return steps, elems, remesh


def show(path):
    steps, elems, remesh = parse(path)
    print(f"=== {os.path.basename(path)}")
    if elems:
        print("  remesh (old->new elements, zip nodes line pairs):", elems[:12], "..." if len(elems) > 12 else "")
    keys = sorted(steps)
    if not keys:
        print("  no [ifc] lines")
        return
    print("  step     t     |   |dE| d1 rms / max   |  |dE| d>=4 rms / max  | excess |  divE d1 rms / max  | divE d>=4 rms | elems  wall(s)")
    for k in keys:
        d = steps[k]
        de = d.get("|dE|", {}).get("all", {})
        dv = d.get("divE", {}).get("all", {})
        d1 = de.get("d1", (float("nan"),) * 3)
        d4 = de.get("d>=4", (float("nan"),) * 3)
        v1 = dv.get("d1", (float("nan"),) * 3)
        v4 = dv.get("d>=4", (float("nan"),) * 3)
        ex = d1[0] / d4[0] if d4[0] else float("nan")
        el = d.get("_elems", 0); wl = d.get("_wall", 0.0)
        print(f"  {k[0]:5d} {k[1]:7.3f} | {d1[0]:.3e} / {d1[1]:.3e} | {d4[0]:.3e} / {d4[1]:.3e} | {ex:6.2f} | {v1[0]:.3e} / {v1[1]:.3e} | {v4[0]:.3e} | {el:6d} {wl:8.1f}")
    # per-level at last step
    k = keys[-1]
    print(f"  per-level |dE| at step {k[0]} (d1 rms/max | d>=4 rms/max | counts):")
    for lvl, cells in sorted(x for x in steps[k].get("|dE|", {}).items() if x[0] != "all"):
        d1 = cells.get("d1")
        d4 = cells.get("d>=4")
        s1 = f"{d1[0]:.3e}/{d1[1]:.3e} ({d1[2]})" if d1 else "-"
        s4 = f"{d4[0]:.3e}/{d4[1]:.3e} ({d4[2]})" if d4 else "-"
        print(f"    lvl {lvl:2d}: d1 {s1:32s} | d>=4 {s4}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        show(p)
