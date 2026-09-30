#!/usr/bin/env python
"""Parameter sweep: how sensitive is the range to each design variable?

Sweeps one parameter at a time around the baseline aircraft and prints a
table. This is the cheapest way to see which number on the drawing actually
matters -- and it is where a point-mass model earns its keep.

Usage::

    python examples/demo_param_scan.py
    python examples/demo_param_scan.py --plot sweep.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flighttrajsim import AircraftParams, AircraftTrajSim, EngineParams, MissionProfile  # noqa: E402
from flighttrajsim.utils import export_sweep_csv  # noqa: E402

# name, start, stop, number of points, unit label
SCANS = [
    ("Take-off mass", "m0", 9000.0, 17000.0, 9, "kg"),
    ("Usable fuel", "fuel_mass", 1500.0, 6000.0, 10, "kg"),
    ("Zero-lift drag", "cd0", 0.018, 0.045, 10, "-"),
    ("Induced drag factor", "k", 0.030, 0.075, 10, "-"),
    ("Sea-level thrust", "thrust_sl", 24000.0, 60000.0, 9, "N"),
    ("TSFC", "tsfc", 1.1e-5, 2.6e-5, 7, "kg/(N s)"),
    ("Cruise altitude", "h_cruise", 5000.0, 14000.0, 10, "m"),
    ("Cruise Mach", "cruise_mach", 0.55, 0.86, 10, "-"),
]

AC_FIELDS = {"m0", "fuel_mass", "S", "cd0", "k", "cl_max", "mach_crit"}
EN_FIELDS = {"thrust_sl", "tsfc", "lapse_exponent"}
MS_FIELDS = {"h_cruise", "cruise_mach", "climb_cas", "descent_throttle"}


def linspace(start: float, stop: float, count: int):
    if count < 2:
        return [start]
    return [start + (stop - start) * i / (count - 1) for i in range(count)]


def build(param: str, value: float):
    """Return (aircraft, engine, mission) with ``param`` overridden."""
    if param in AC_FIELDS:
        return AircraftParams().updated(**{param: value}), EngineParams(), MissionProfile()
    if param in EN_FIELDS:
        return AircraftParams(), EngineParams().updated(**{param: value}), MissionProfile()
    return AircraftParams(), EngineParams(), MissionProfile().updated(**{param: value})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", help="save the sweep figure to this image file")
    parser.add_argument("--csv", default="sweep.csv", help="sweep CSV output")
    args = parser.parse_args(argv)

    baseline = AircraftTrajSim().run_simulation()
    print("baseline range: %.0f km\n" % baseline.range_km)

    all_rows = []
    for label, param, start, stop, count, unit in SCANS:
        print("%-20s (%s)" % (label, param))
        print("  %14s %10s %10s %10s %8s" % ("value", "range[km]", "delta[%]",
                                             "endur[min]", "ok"))
        print("  " + "-" * 56)
        rows = []
        for value in linspace(start, stop, count):
            aircraft, engine, mission = build(param, value)
            result = AircraftTrajSim(aircraft, engine, mission).run_simulation()
            delta = 100.0 * (result.range_km - baseline.range_km) / baseline.range_km
            print("  %14.5g %10.0f %+10.1f %10.0f %8s"
                  % (value, result.range_km, delta, result.endurance_min,
                     "yes" if result.completed else "NO"))
            row = {
                "scan": label, "param": param, "value": value, "unit": unit,
                "range_km": result.range_km, "delta_pct": delta,
                "endurance_min": result.endurance_min,
                "fuel_burned_kg": result.fuel_burned_kg,
                "completed": float(bool(result.completed)),
            }
            rows.append(row)
            all_rows.append(row)
        print()

    path = export_sweep_csv(all_rows, args.csv)
    print("sweep written to %s" % path)

    if args.plot:
        from flighttrajsim import plot_sweep
        plot_rows = [r for r in all_rows if r["param"] == "cd0"]
        plot_sweep(plot_rows, "value", "range_km", show=False, save_path=args.plot)
        print("figure written to %s" % args.plot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
