"""Aerodynamic model: parabolic drag polar with compressibility correction."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .atmosphere import G0


@dataclass
class AeroState:
    """Instantaneous aerodynamic state.

    Attributes:
        cl: Lift coefficient [-].
        cd: Total drag coefficient [-].
        lift: Lift force [N].
        drag: Drag force [N].
        ld: Lift-to-drag ratio [-].
        mach: Mach number [-].
        stalled: True when the required CL exceeds ``cl_max``.
    """

    cl: float
    cd: float
    lift: float
    drag: float
    ld: float
    mach: float
    stalled: bool


def lift_coefficient(mass: float, gamma: float, q: float, s: float) -> float:
    """Lift coefficient required for quasi-steady flight.

    ``L = m g cos(gamma)`` so ``CL = L / (q S)``.
    """
    lift = mass * G0 * math.cos(gamma)
    return lift / (q * s) if q > 0.0 else float("inf")


def wave_drag_coefficient(mach: float, mach_crit: float) -> float:
    """Compressibility (wave) drag increment above the critical Mach number.

    A mild cubic rise ``12 (M - M_crit)^3``; zero below ``mach_crit``.
    """
    if mach <= mach_crit:
        return 0.0
    return 12.0 * (mach - mach_crit) ** 3


def drag_coefficient(
    cl: float, cd0: float, k: float, mach: float = 0.0, mach_crit: float = 0.82
) -> float:
    """Total drag coefficient ``CD = CD0 + k CL^2 + CD_wave``."""
    return cd0 + k * cl * cl + wave_drag_coefficient(mach, mach_crit)


def dynamic_pressure(rho: float, v: float) -> float:
    """Dynamic pressure ``q = 0.5 rho V^2`` [Pa]."""
    return 0.5 * rho * v * v


def aero_state(
    mass: float,
    gamma: float,
    v: float,
    rho: float,
    s: float,
    cd0: float,
    k: float,
    cl_max: float,
    mach_crit: float,
    mach: float = 0.0,
) -> AeroState:
    """Assemble the full aerodynamic state at one trajectory point."""
    q = dynamic_pressure(rho, v)
    cl = lift_coefficient(mass, gamma, q, s)
    cd = drag_coefficient(cl, cd0, k, mach, mach_crit)
    lift = q * s * cl
    drag = q * s * cd
    ld = lift / drag if drag > 0 else float("inf")
    return AeroState(
        cl=cl, cd=cd, lift=lift, drag=drag, ld=ld, mach=mach, stalled=cl > cl_max
    )


def max_lift_to_drag(cd0: float, k: float) -> tuple[float, float]:
    """Best glide ratio and the CL at which it occurs.

    For a parabolic polar ``(L/D)_max = 1 / (2 sqrt(CD0 k))`` at
    ``CL = sqrt(CD0 / k)``. Used as a sanity check on the polar.
    """
    cl_opt = math.sqrt(cd0 / k)
    return cl_opt, 1.0 / (2.0 * math.sqrt(cd0 * k))


def breguet_range(v: float, ld: float, tsfc: float, m0: float, m1: float) -> float:
    """Breguet range for a jet aircraft at constant speed and L/D [m].

    ``R = V (L/D) / (TSFC g) ln(m0 / m1)``

    Used in the test-suite as an independent analytical cross-check of the
    numerically integrated cruise segment.
    """
    return v * ld / (tsfc * G0) * math.log(m0 / m1)
