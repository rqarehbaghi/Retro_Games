#!/usr/bin/env python3
"""Aggregate the sweep into the published result: distributions, no thresholds.

    python tools/progression/report.py                 # writes results.json/.csv
    python tools/progression/report.py --select        # also applies the state rule

Reports median, IQR, min, max, how many trajectories were CENSORED by the
placement cap, and the count of states -- and publishes every raw trajectory.
It deliberately reports no pass/fail line: the old "100+ lines" figure was a
diagnostic goal from when the agent cleared a handful of lines, it is retired,
and it must not reappear as a target, threshold or benchmark.
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.progression import ids, select                        # noqa: E402

OUT = os.path.join(ROOT, "progression_out")
CACHE = os.path.join(OUT, "cache")


def quantile(values, q):
    """Linear-interpolated quantile, so IQR does not depend on numpy here."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (pos - low))


def load_results(placement_cap=None):
    """Cached trajectories, filtered to ONE placement cap.

    The cache legitimately holds runs at other caps -- the probe that rejected
    500 is still there -- and pooling them would compare checkpoints under
    different budgets, which is the whole thing this design guards against.
    Anything at another cap is excluded and counted."""
    rows, excluded = [], 0
    if not os.path.isdir(CACHE):
        return rows, excluded
    for name in sorted(os.listdir(CACHE)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(CACHE, name)) as fh:
            row = json.load(fh)
        if placement_cap is not None and int(row.get("placement_cap", -1)) != int(placement_cap):
            excluded += 1
            continue
        rows.append(row)
    return rows, excluded


def checkpoint_label(row):
    name = row["checkpoint"]
    if name == "control":
        return "control"
    digits = "".join(c for c in name if c.isdigit())
    return "%dk" % (int(digits) // 1000) if digits else name


def summarise(rows, field="delta_lines"):
    """Per checkpoint: the distribution, and how much of it is censored."""
    groups = {}
    for row in rows:
        groups.setdefault(checkpoint_label(row), []).append(row)
    out = {}
    for label, group in groups.items():
        values = [r[field] for r in group]
        capped = sum(1 for r in group if r["end_reason"] == "placement_cap")
        out[label] = {
            "states": len(group),
            "median": quantile(values, 0.5),
            "q1": quantile(values, 0.25),
            "q3": quantile(values, 0.75),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "capped_trajectories": capped,
            "steps": group[0].get("steps"),
            "placement_cap": group[0]["placement_cap"],
        }
        out[label]["iqr"] = (None if out[label]["q1"] is None
                             else round(out[label]["q3"] - out[label]["q1"], 2))
    return out


def order_labels(labels):
    def key(label):
        if label == "control":
            return (0, 0)
        digits = "".join(c for c in label if c.isdigit())
        return (1, int(digits) if digits else 0)
    return sorted(labels, key=key)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--field", default="delta_lines", choices=("delta_lines", "placements"))
    p.add_argument("--select", action="store_true", help="apply the state-selection rule")
    a = p.parse_args()

    manifest = json.load(open(os.path.join(OUT, "suite_manifest.json")))
    cap = int(manifest["placement_cap"])
    rows, excluded = load_results(cap)
    if not rows:
        sys.exit("no results at cap %d in %s -- run the evaluator first" % (cap, CACHE))
    if excluded:
        print("(%d cached trajectories at another placement cap were excluded)" % excluded)
    summary = summarise(rows, a.field)

    print("%-8s %6s %8s %8s %8s %8s %8s %8s" %
          ("ckpt", "states", "median", "q1", "q3", "min", "max", "capped"))
    for label in order_labels(summary):
        s = summary[label]
        print("%-8s %6d %8.1f %8.1f %8.1f %8d %8d %8d" %
              (label, s["states"], s["median"], s["q1"], s["q3"], s["min"], s["max"],
               s["capped_trajectories"]))

    payload = {
        "field": a.field,
        "suite_manifest_sha256": ids.sha256_text(
            open(os.path.join(OUT, "suite_manifest.json")).read()),
        "placement_cap": manifest["placement_cap"],
        "states": len(manifest["states"]),
        "summary": summary,
        "trajectories": sorted(rows, key=lambda r: (checkpoint_label(r), r["state"])),
        "reporting": ("distributions only: median, IQR, min, max and the number of "
                      "trajectories censored by the placement cap. No threshold is "
                      "reported; the retired 100-line figure is not a target."),
    }
    if a.select:
        control = {r["state"]: r for r in rows if r["checkpoint"] == "control"}
        if not control:
            sys.exit("the control has not been evaluated: it must run before states are chosen")
        payload["selection"] = select.select(control)
        print("\nselected from the control only: median state %s, hard state %s"
              % (payload["selection"]["median_state"], payload["selection"]["hard_state"]))
        print("  rule: %s" % payload["selection"]["rule"])

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["checkpoint", "state", "placements", "delta_lines", "end_reason",
                    "holes", "height", "placement_cap"])
        for r in payload["trajectories"]:
            w.writerow([checkpoint_label(r), r["state"], r["placements"], r["delta_lines"],
                        r["end_reason"], r["holes"], r["height"], r["placement_cap"]])
    print("\nwrote %s and results.csv (%d trajectories)"
          % (os.path.join(OUT, "results.json"), len(rows)))


if __name__ == "__main__":
    main()
