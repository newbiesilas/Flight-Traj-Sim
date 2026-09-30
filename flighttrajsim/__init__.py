"""Flight-Traj-Sim - point-mass aircraft trajectory simulation toolkit.

Solves the climb / cruise / descent mission of a fixed-wing aircraft with a
point-mass ODE model, a parabolic drag polar, an ISA atmosphere and a turbofan
thrust / fuel-flow model, and provides parameter sweeps for sensitivity
analysis.

Typical use::

    from flighttrajsim import AircraftTrajSim, plot_trajectory

    sim = AircraftTrajSim()
    result = sim.run_simulation()
    print(result.summary())
    plot_trajectory(result)
"""

__version__ = "1.0.0"

from .aero import (
    aero_state,
    breguet_range,
    drag_coefficient,
    dynamic_pressure,
    lift_coefficient,
    max_lift_to_drag,
    wave_drag_coefficient,
)
from .atmosphere import (
    density,
    density_ratio,
    equivalent_airspeed,
    geopotential_altitude,
    mach,
    pressure,
    scheduled_speed,
    speed_of_sound,
    temperature,
)
from .core import CLIMB, CRUISE, DESCENT, AircraftTrajSim, Instant, TrajectoryResult
from .engine import available_thrust, cruise_equilibrium_speed, fuel_flow, thrust_lapse
from .integrate import PhaseSolution, integrate_phase, rk4_step
from .params import AircraftParams, EngineParams, MissionProfile
from .utils import (
    export_result_csv,
    export_sweep_csv,
    format_summary,
    format_sweep_table,
)

# Plotting is optional: matplotlib is imported lazily inside each helper so
# that the solver, the tests and the packaged GUI never need it.
from . import plot as plot  # noqa: E402  (namespace module, no hard import)

build_sweep_figure = plot.build_sweep_figure
build_mission_profile_figure = plot.build_mission_profile_figure
build_trajectory_figure = plot.build_trajectory_figure
plot_sweep = plot.plot_sweep
plot_trajectory = plot.plot_trajectory

__all__ = [
    "__version__",
    # core
    "AircraftTrajSim",
    "TrajectoryResult",
    "CLIMB",
    "CRUISE",
    "DESCENT",
    # params
    "AircraftParams",
    "EngineParams",
    "MissionProfile",
    # physics
    "density",
    "density_ratio",
    "pressure",
    "temperature",
    "speed_of_sound",
    "mach",
    "geopotential_altitude",
    "equivalent_airspeed",
    "scheduled_speed",
    "aero_state",
    "drag_coefficient",
    "wave_drag_coefficient",
    "dynamic_pressure",
    "lift_coefficient",
    "max_lift_to_drag",
    "breguet_range",
    "available_thrust",
    "thrust_lapse",
    "fuel_flow",
    "cruise_equilibrium_speed",
    # numerics
    "rk4_step",
    "integrate_phase",
    "PhaseSolution",
    # output
    "plot_trajectory",
    "plot_sweep",
    "build_trajectory_figure",
    "build_mission_profile_figure",
    "build_sweep_figure",
    "export_result_csv",
    "export_sweep_csv",
    "format_summary",
    "format_sweep_table",
]
