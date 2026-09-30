"""Command line interface.

::

    flighttrajsim run --mass 13000 --altitude 11000 --mach 0.78 --csv out.csv
    flighttrajsim sweep --param m0 --values 10000 12000 14000
    flighttrajsim breguet --mass 12000
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .core import AircraftTrajSim
from .params import AircraftParams, EngineParams, MissionProfile
from .plot import plot_sweep, plot_trajectory
from .utils import export_result_csv, export_sweep_csv, format_summary, format_sweep_table

_AC_FIELDS = set(AircraftParams().__dataclass_fields__)
_EN_FIELDS = set(EngineParams().__dataclass_fields__)
_MS_FIELDS = set(MissionProfile().__dataclass_fields__)


def _split(changes: dict):
    ac = {k: v for k, v in changes.items() if k in _AC_FIELDS}
    en = {k: v for k, v in changes.items() if k in _EN_FIELDS}
    ms = {k: v for k, v in changes.items() if k in _MS_FIELDS}
    return ac, en, ms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flighttrajsim",
        description="Point-mass aircraft trajectory simulation toolkit",
    )
    parser.add_argument("-V", "--version", action="version", version="Flight-Traj-Sim 1.0.0")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="simulate a single mission")
    run.add_argument("--mass", type=float, help="take-off mass [kg]")
    run.add_argument("--fuel", type=float, help="usable fuel mass [kg]")
    run.add_argument("--thrust", type=float, help="sea-level static thrust [N]")
    run.add_argument("--tsfc", type=float, help="thrust specific fuel consumption [kg/(N s)]")
    run.add_argument("--cd0", type=float, help="zero-lift drag coefficient [-]")
    run.add_argument("--k", type=float, dest="k_induced", help="induced drag factor [-]")
    run.add_argument("--area", type=float, dest="S", help="wing area [m^2]")
    run.add_argument("--altitude", type=float, dest="h_cruise", help="cruise altitude [m]")
    run.add_argument("--mach", type=float, dest="cruise_mach", help="cruise Mach number [-]")
    run.add_argument("--cas", type=float, dest="climb_cas", help="climb calibrated airspeed [m/s]")
    run.add_argument("--csv", help="write the trajectory to this CSV file")
    run.add_argument("--plot", help="write the mission figure to this image file")

    sweep = sub.add_parser("sweep", help="sweep one parameter")
    sweep.add_argument("--param", default="m0", help="parameter name, e.g. m0, thrust_sl, cd0")
    sweep.add_argument(
        "--values", nargs="+", type=float, required=True, help="values to evaluate"
    )
    sweep.add_argument("--altitude", type=float, dest="h_cruise", default=None)
    sweep.add_argument("--csv", help="write sweep rows to this CSV file")
    sweep.add_argument("--plot", help="write the sweep figure to this image file")

    sub.add_parser("breguet", help="closed-form Breguet cruise range for comparison")

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    overrides = {
        k: getattr(args, k)
        for k in (
            "mass", "fuel", "thrust", "tsfc", "cd0", "k_induced", "S",
            "h_cruise", "cruise_mach", "climb_cas",
        )
        if getattr(args, k, None) is not None
    }
    overrides = {"m0" if k == "mass" else k: v for k, v in overrides.items()}
    overrides = {"fuel_mass" if k == "fuel" else k: v for k, v in overrides.items()}
    overrides = {"thrust_sl" if k == "thrust" else k: v for k, v in overrides.items()}
    ac_over, en_over, ms_over = _split(overrides)

    sim = AircraftTrajSim(
        aircraft=AircraftParams().updated(**ac_over),
        engine=EngineParams().updated(**en_over),
        mission=MissionProfile().updated(**ms_over),
    )
    try:
        result = sim.run_simulation()
    except (RuntimeError, ValueError) as exc:
        print("simulation failed: %s" % exc, file=sys.stderr)
        return 2

    print(format_summary(result))
    if args.csv:
        print("CSV  -> %s" % export_result_csv(result, args.csv))
    if args.plot:
        plot_trajectory(result, show=False, save_path=args.plot)
        print("plot -> %s" % args.plot)

    if not result.completed:
        print("warning: mission incomplete (%s)" % result.message, file=sys.stderr)
        return 1
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    mission = MissionProfile()
    if args.h_cruise is not None:
        mission = mission.updated(h_cruise=args.h_cruise)

    sim = AircraftTrajSim(mission=mission)
    try:
        rows = sim.sweep(args.param, args.values, mission=mission)
    except ValueError as exc:
        print("sweep failed: %s" % exc, file=sys.stderr)
        return 2

    print(format_sweep_table(rows, args.param))
    if args.csv:
        print("CSV  -> %s" % export_sweep_csv(rows, args.csv))
    if args.plot:
        plot_sweep(rows, args.param, show=False, save_path=args.plot)
        print("plot -> %s" % args.plot)
    return 0


def cmd_breguet(args: argparse.Namespace) -> int:
    """Print the closed-form Breguet cruise range next to the integrated one."""
    sim = AircraftTrajSim()
    result = sim.run_simulation()
    cruise = result.phase_slice("cruise")
    integrated = 0.0
    analytic = float("nan")
    if cruise is not None:
        i0, i1 = cruise
        integrated = (result.x[i1 - 1] - result.x[i0]) / 1000.0
        analytic = sim.breguet_range_estimate(
            mass_start=result.m[i0], mass_end=result.m[i1 - 1]
        ) / 1000.0
    print("Breguet cruise range (closed form) : %8.1f km" % analytic)
    print("Simulated cruise segment           : %8.1f km" % integrated)
    if integrated > 0:
        print("Relative difference                : %8.3f %%"
              % (100.0 * (integrated - analytic) / analytic))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "sweep":
        return cmd_sweep(args)
    if args.command == "breguet":
        return cmd_breguet(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
