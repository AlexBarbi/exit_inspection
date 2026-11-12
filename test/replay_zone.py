#!/usr/bin/env python3
"""
replay_zone.py (test tool)

Evaluates again the inputs exit_zone_checker.py saved with debug_dump:=true (zone_debug/exit_<id>.npz), with the
current exit_inspection/zone.py and optional parameter changes, and prints the verdict and the grid (door at the
bottom, u left to right): '.' free, '#' object, 'h' hidden behind an object, 'p' person, 'b' behind a person,
' ' unseen.

Usage: replay_zone.py zone_debug/exit_0.npz [--set w_min=0.8 object_depth=0.3] [--no-grid]
"""

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exit_inspection import zone as zn  # noqa: E402

CHARS = {zn.UNKNOWN: " ", zn.FREE: ".", zn.OBJECT: "#", zn.HIDDEN: "h", zn.PERSON: "p", zn.BEHIND: "b"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--no-grid", action="store_true")
    args = ap.parse_args()
    d = np.load(args.npz, allow_pickle=False)
    params = zn.ZoneParams()
    for kv in args.set:
        k, _, v = kv.partition("=")
        setattr(params, k, type(getattr(params, k))(v))
    zone = zn.Zone.from_exit(yaml.safe_load(str(d["exit"])), params)
    pts, fid = d["points"].astype(float), d["fid"]
    kinds = d["kinds"] if "kinds" in d else ["depth"] * len(d["stamps"])
    frames = [zn.Frame(float(t), s, pts[fid == i], str(k))
              for i, (t, s, k) in enumerate(zip(d["stamps"], d["sensors"], kinds))]
    r = zn.evaluate(zone, frames, d["persons"])
    for k, v in r.summary().items():
        print(f"{k}: {v}")
    if not args.no_grid:
        print("+" + "-" * r.state.shape[0] + "+  far edge")
        for j in range(r.state.shape[1] - 1, -1, -1):
            print("|" + "".join(CHARS[c] for c in r.state[:, j]) + "|")
        print("+" + "-" * r.state.shape[0] + "+  door")


if __name__ == "__main__":
    main()
