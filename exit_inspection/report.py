"""
Inspection pass results (sub-project 3 of the emergency exit inspection): choosing the verdict of a stop, and the
report (report.yaml + one self-contained report.html with a top-down map).
"""

import html
import os
import shutil
from collections import Counter
from typing import List, Optional, Sequence

import numpy as np
import yaml

# worse first: a tie between two verdicts goes to the worse one
RANK = {"BLOCKED": 4, "WARNING": 3, "UNVERIFIED": 2, "CLEAR": 1}
COLORS = {"CLEAR": "#1a9850", "WARNING": "#e6a100", "BLOCKED": "#d7301f", "UNVERIFIED": "#8c8c8c"}


NO_CAMERA = "camera did not see the zone"


def has_data(evaluation: dict) -> bool:
    """False for the checker's "camera did not see the zone" (a gap in the data, not a judgement of the zone)."""
    return not str(evaluation.get("reason", "")).startswith(NO_CAMERA)


def pick_verdict(evaluations: Sequence[dict]) -> Optional[dict]:
    """The latest evaluation with the most frequent verdict (ties: the worse verdict), among those with camera data
    when there are any; None without evaluations."""
    if not evaluations:
        return None
    evaluations = [e for e in evaluations if has_data(e)] or list(evaluations)
    counts = Counter(e["verdict"] for e in evaluations)
    verdict = max(counts, key=lambda v: (counts[v], RANK.get(v, 0)))
    chosen = dict([e for e in evaluations if e["verdict"] == verdict][-1])
    chosen["votes"] = f"{counts[verdict]}/{len(evaluations)}"
    return chosen


def best_attempt(attempts: Sequence[dict]) -> Optional[dict]:
    """The first attempt with a verdict other than UNVERIFIED, else the last one."""
    for a in attempts:
        if a.get("verdict") not in (None, "UNVERIFIED"):
            return a
    return attempts[-1] if attempts else None


def summary(exits: Sequence[dict]) -> dict:
    counts = Counter(e.get("verdict", "UNVERIFIED") for e in exits)
    return {v: counts.get(v, 0) for v in ("CLEAR", "WARNING", "BLOCKED", "UNVERIFIED")}


def map_cells(xyz: Optional[np.ndarray], floor_z: float, low: float = 0.15, high: float = 1.8,
              resolution: float = 0.1) -> np.ndarray:
    """(N, 2) corners of the occupied map cells (points low..high above the floor) for the drawing."""
    if xyz is None or len(xyz) == 0:
        return np.zeros((0, 2))
    band = xyz[(xyz[:, 2] >= floor_z + low) & (xyz[:, 2] <= floor_z + high), :2]
    return np.unique(np.floor(band / resolution).astype(np.int64), axis=0) * resolution


def _svg(mission: dict, cells: np.ndarray, resolution: float) -> str:
    pts = [np.array(e["door_floor_center"][:2]) for e in mission["exits"]]
    pts += [np.array(p) for p in mission.get("path", [])]
    if len(cells):
        pts += [cells.min(0), cells.max(0) + resolution]
    if not pts:
        return ""
    lo, hi = np.min(pts, axis=0) - 2.0, np.max(pts, axis=0) + 2.0
    w, h = hi - lo
    out = [f'<svg viewBox="{lo[0]:.2f} {-hi[1]:.2f} {w:.2f} {h:.2f}" xmlns="http://www.w3.org/2000/svg" '
           f'class="map"><g transform="scale(1,-1)">']
    if len(cells):
        d = "".join(f"M{x:.1f} {y:.1f}h{resolution}v{resolution}h-{resolution}z" for x, y in cells)
        out.append(f'<path d="{d}" class="wall"/>')
    path = mission.get("path", [])
    if len(path) > 1:
        out.append('<polyline class="path" points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in path) + '"/>')
    for e in mission["exits"]:
        c = np.array(e["door_floor_center"][:2])
        n = np.array(e["normal"][:2], float)
        n /= np.linalg.norm(n)
        t = np.array([-n[1], n[0]])
        half = e.get("door_width", 1.2) / 2
        color = COLORS.get(e.get("verdict"), COLORS["UNVERIFIED"])
        zone_half, depth = half + mission.get("params", {}).get("margin", 0.5), mission.get("params", {}).get(
            "depth", 3.0)
        corners = [c - zone_half * t, c - zone_half * t + depth * n, c + zone_half * t + depth * n, c + zone_half * t]
        out.append(f'<polygon points="{" ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in corners)}" fill="{color}" '
                   f'fill-opacity="0.25" stroke="{color}" stroke-width="0.05"/>')
        a, b = c - half * t, c + half * t
        out.append(f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke="{color}" '
                   f'stroke-width="0.25"/>')
        vp = e.get("viewpoint")
        if vp:
            tip = (vp["x"] + 0.6 * np.cos(vp["yaw"]), vp["y"] + 0.6 * np.sin(vp["yaw"]))
            out.append(f'<circle cx="{vp["x"]:.2f}" cy="{vp["y"]:.2f}" r="0.18" class="vp"/>'
                       f'<line x1="{vp["x"]:.2f}" y1="{vp["y"]:.2f}" x2="{tip[0]:.2f}" y2="{tip[1]:.2f}" '
                       f'class="vp-dir"/>')
        label = c + (depth + 0.6) * n
        out.append(f'<text x="{label[0]:.2f}" y="{-label[1]:.2f}" transform="scale(1,-1)" class="label" '
                   f'fill="{color}">{html.escape(str(e["id"]))}</text>')
    out.append("</g></svg>")
    return "".join(out)


def render_html(mission: dict, cells: np.ndarray, resolution: float = 0.1) -> str:
    s = mission.get("summary") or summary(mission["exits"])
    badges = "".join(f'<span class="badge" style="background:{COLORS[v]}">{v} {n}</span>' for v, n in s.items())
    rows = []
    for e in mission["exits"]:
        v = e.get("verdict", "UNVERIFIED")
        attempts = e.get("attempts", [])
        reached = sum(1 for a in attempts if a.get("reached"))
        cells_ = [str(e["id"]), html.escape(str(e.get("cls", ""))),
                  f'<b style="color:{COLORS.get(v, "#888")}">{v}</b>', html.escape(str(e.get("reason", ""))),
                  f'{e["coverage"]:.0%}' if e.get("coverage") is not None else "",
                  f'{e["object_area"]:.2f}' if e.get("object_area") is not None else "",
                  f'{e["corridor_width"]:.2f}' if e.get("corridor_width") is not None else "",
                  str(e.get("people", "")), f"{reached}/{len(attempts)}",
                  f'{e["duration"]:.0f}' if e.get("duration") is not None else ""]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells_) + "</tr>")
    head = ["exit", "class", "verdict", "reason", "seen", "objects m²", "corridor m", "people", "stops reached",
            "time s"]
    meta = [f"started {html.escape(str(mission.get('started', '')))}",
            f"finished {html.escape(str(mission.get('finished', '')))}",
            f"exits {html.escape(str(mission.get('exits_file', '')))}",
            f"planner {html.escape(str(mission.get('planner', '')))}"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Exit inspection report</title>
<style>
:root {{ --bg:#ffffff; --fg:#1f2328; --muted:#59636e; --line:#d1d9e0; --wall:#3d444d; --path:#0969da; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --line:#3d444d;
  --wall:#b7bdc8; --path:#4493f8; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; margin:0 auto; max-width:1100px;
  padding:16px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} .meta {{ color:var(--muted); font-size:13px; }}
.badge {{ color:#fff; border-radius:12px; padding:2px 10px; margin-right:6px; font-weight:600; font-size:13px; }}
.map {{ width:100%; max-height:70vh; border:1px solid var(--line); border-radius:6px; margin:12px 0; }}
.wall {{ fill:var(--wall); }} .path {{ fill:none; stroke:var(--path); stroke-width:0.06; }}
.vp {{ fill:var(--path); }} .vp-dir {{ stroke:var(--path); stroke-width:0.08; }}
.label {{ font-size:0.7px; font-weight:700; }}
.table-wrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th, td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }}
th {{ color:var(--muted); font-weight:600; }}
</style></head><body>
<h1>Exit inspection report</h1>
<div class="meta">{" · ".join(meta)}</div>
<p>{badges}</p>
{_svg(mission, cells, resolution)}
<div class="table-wrap"><table><thead><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
</body></html>
"""


def write_report(directory: str, stamp: str, mission: dict, map_xyz: Optional[np.ndarray] = None,
                 floor_z: float = 0.0) -> dict:
    """Writes <directory>/reports/<stamp>/report.yaml and report.html and copies the YAML to
    <directory>/exit_report.yaml. Returns the paths."""
    mission = dict(mission, summary=summary(mission["exits"]))
    folder = os.path.join(directory, "reports", stamp)
    os.makedirs(folder, exist_ok=True)
    yaml_path, html_path = os.path.join(folder, "report.yaml"), os.path.join(folder, "report.html")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(mission, f, sort_keys=False, default_flow_style=None)
    with open(html_path, "w") as f:
        f.write(render_html(mission, map_cells(map_xyz, floor_z)))
    latest = os.path.join(directory, "exit_report.yaml")
    shutil.copyfile(yaml_path, latest + ".tmp")
    os.replace(latest + ".tmp", latest)
    return {"yaml": yaml_path, "html": html_path, "latest": latest}


def exit_entry(exit_: dict, attempts: List[dict], duration: float) -> dict:
    """One exit of the report from its attempts (each with viewpoint, reached, verdict fields)."""
    best = best_attempt(attempts) or {}
    keys = ("verdict", "reason", "coverage", "object_area", "corridor_width", "corridor_width_optimistic", "people",
            "behind_person_area", "frames", "votes")
    out = {"id": exit_.get("id"), "cls": exit_.get("cls", ""), "name": exit_.get("name", ""),
           "door_floor_center": exit_.get("door_floor_center"), "normal": exit_.get("normal"),
           "door_width": exit_.get("door_width", 1.2), "viewpoint": best.get("viewpoint")}
    out.update({k: best.get(k) for k in keys})
    if out["verdict"] is None:
        out["verdict"], out["reason"] = "UNVERIFIED", best.get("reason") or "not inspected"
    out["attempts"] = list(attempts)
    out["duration"] = round(float(duration), 1)
    return out
