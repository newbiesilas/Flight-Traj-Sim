#!/usr/bin/env python
"""Baseline climb / cruise / descent mission.

Runs the default aircraft, prints the mission summary and the per-phase
breakdown, cross-checks the cruise segment against the closed-form Breguet
range and writes the trajectory to CSV.

Usage::

    python examples/demo_climb_cruise.py
    python examples/demo_climb_cruise.py --plot mission.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flighttrajsim import AircraftTrajSim  # noqa: E402
from flighttrajsim.utils import export_result_csv, format_summary  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", help="save the mission figure to this image file")
    parser.add_argument("--csv", default="mission.csv", help="trajectory CSV output")
    args = parser.parse_args(argv)

    sim = AircraftTrajSim()
    result = sim.run_simulation()

    print(format_summary(result))
    print()

    print("%-9s %8s %9s %10s %11s %10s" % ("phase", "time[min]", "dH[m]", "dx[km]",
                                           "fuel[kg]", "TAS[m/s]"))
    print("-" * 62)
    for name, i0, i1 in result.phases:
        print("%-9s %8.1f %9.0f %10.1f %11.1f %10.1f" % (
            name,
            (result.t[i1 - 1] - result.t[i0]) / 60.0,
            result.h[i1 - 1] - result.h[i0],
            (result.x[i1 - 1] - result.x[i0]) / 1000.0,
            result.m[i0] - result.m[i1 - 1],
            result.v[i1 - 1],
        ))

    # Independent analytic check on the cruise segment.
    cruise = result.phase_slice("cruise")
    if cruise is not None:
        i0, i1 = cruise
        integrated = (result.x[i1 - 1] - result.x[i0]) / 1000.0
        analytic = sim.breguet_range_estimate(
            mass_start=result.m[i0], mass_end=result.m[i1 - 1]
        ) / 1000.0
        print()
        print("cruise segment, integrated      : %8.1f km" % integrated)
        print("cruise segment, Breguet (exact) : %8.1f km" % analytic)
        print("relative difference             : %8.3f %%"
              % (100.0 * (integrated - analytic) / analytic))

    cl_opt, ld_max = sim.best_glide()
    print()
    print("polar: (L/D)max = %.2f at CL = %.3f" % (ld_max, cl_opt))

    path = export_result_csv(result, args.csv)
    print("trajectory written to %s" % path)

    if args.plot:
        from flighttrajsim import plot_trajectory
        plot_trajectory(result, show=False, save_path=args.plot)
        print("figure written to %s" % args.plot)

    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
