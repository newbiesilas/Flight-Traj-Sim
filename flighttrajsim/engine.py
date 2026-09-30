"""Turbofan engine model: altitude thrust lapse and fuel consumption."""

from __future__ import annotations

from .params import EngineParams


def thrust_lapse(sigma: float, exponent: float = 0.7) -> float:
    """Thrust lapse factor ``sigma ** exponent``.

    ``sigma`` is the density ratio ``rho / rho0``. An exponent near 0.7 is the
    classical approximation for high-bypass turbofans; 1.0 would correspond to
    a pure turbojet.
    """
    if sigma <= 0.0:
        return 0.0
    return sigma ** exponent


def available_thrust(engine: EngineParams, sigma: float, throttle: float) -> float:
    """Installed thrust [N] at the given density ratio and throttle setting."""
    return engine.thrust_sl * thrust_lapse(sigma, engine.lapse_exponent) * throttle


def fuel_flow(thrust: float, tsfc: float) -> float:
    """Fuel mass flow [kg/s] for the delivered thrust [N].

    ``mdot = TSFC * T`` with TSFC in kg/(N s). Thrust is clipped at zero so
    that windmilling / zero-thrust segments do not produce negative flow.
    """
    return tsfc * max(thrust, 0.0)


def cruise_equilibrium_speed(
    mass: float,
    rho: float,
    s: float,
    cd0: float,
    k: float,
    thrust: float,
) -> tuple[float, float] | None:
    """True airspeeds where thrust equals drag in level flight [m/s].

    Solves ``A V^4 - T V^2 + B = 0`` with ``A = 0.5 rho S CD0`` and
    ``B = k W^2 / (0.5 rho S)``, i.e. the two roots of the classic
    thrust-required curve.

    Returns:
        ``(v_slow, v_fast)`` in m/s, or ``None`` when the available thrust is
        below the minimum drag required to sustain level flight.
    """
    from .atmosphere import G0

    weight = mass * G0
    a = 0.5 * rho * s * cd0
    b = k * weight * weight / (0.5 * rho * s)
    discriminant = thrust * thrust - 4.0 * a * b
    if discriminant < 0.0 or a <= 0.0:
        return None  # thrust insufficient: no level-flight equilibrium
    root = discriminant ** 0.5
    v2_slow = (thrust - root) / (2.0 * a)
    v2_fast = (thrust + root) / (2.0 * a)
    return v2_slow ** 0.5, v2_fast ** 0.5
