"""Parameter containers for the aircraft, engine and mission.

Everything is expressed in **SI base units** (kg, m, s, N, K). Keeping a
single unit system throughout removes the classic source of simulation bugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Any


@dataclass
class AircraftParams:
    """Aircraft geometry, aerodynamics and mass properties.

    Defaults describe a ~12 t twin-turbofan transport in the class of a
    regional jet. They are engineering-plausible, not a specific type.

    Attributes:
        m0: Take-off mass [kg].
        fuel_mass: Usable fuel mass [kg].
        reserve_fuel: Fuel that must remain at the end of cruise [kg].
        S: Wing reference area [m^2].
        cd0: Zero-lift drag coefficient [-].
        k: Induced drag factor, such that ``CD = cd0 + k * CL^2`` [-].
        cl_max: Maximum usable lift coefficient (stall limit) [-].
        mach_crit: Critical Mach number; above it wave drag is added [-].
    """

    m0: float = 12000.0
    fuel_mass: float = 3000.0
    reserve_fuel: float = 200.0
    S: float = 24.0
    cd0: float = 0.025
    k: float = 0.048
    cl_max: float = 1.4
    mach_crit: float = 0.82

    @property
    def dry_mass(self) -> float:
        """Operational empty mass plus payload [kg]."""
        return self.m0 - self.fuel_mass

    @property
    def min_mass(self) -> float:
        """Mass at which the fuel is considered exhausted (dry + reserve)."""
        return self.dry_mass + self.reserve_fuel

    def validate(self) -> None:
        """Raise ``ValueError`` on physically impossible parameters."""
        if self.m0 <= 0:
            raise ValueError("m0 must be positive")
        if not 0 < self.fuel_mass < self.m0:
            raise ValueError("fuel_mass must be positive and smaller than m0")
        if self.reserve_fuel < 0 or self.reserve_fuel >= self.fuel_mass:
            raise ValueError("reserve_fuel must be in [0, fuel_mass)")
        if self.S <= 0:
            raise ValueError("S must be positive")
        if self.cd0 <= 0:
            raise ValueError("cd0 must be positive")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.cl_max <= 0:
            raise ValueError("cl_max must be positive")

    def updated(self, **changes: float) -> "AircraftParams":
        """Return a copy with ``changes`` applied (functional update)."""
        return replace(self, **changes)


@dataclass
class EngineParams:
    """Turbofan engine model parameters.

    Attributes:
        thrust_sl: Maximum sea-level static thrust of the whole engine set [N].
            39 kN against a 12 t airframe gives a thrust-to-weight ratio of
            0.33, squarely in the regional-jet / midsize-business-jet class.
        tsfc: Thrust specific fuel consumption [kg/(N s)].
            1.6e-5 corresponds to roughly 0.57 lb/(lbf h), typical for a
            modern high-bypass turbofan at cruise.
        lapse_exponent: Altitude lapse exponent, ``T = T_sl * sigma**n``.
            ``n = 0.9`` sits between the classical turbofan value (0.7) and
            the pure-turbojet limit (1.0) and matches cruise thrust fractions
            of 0.6-0.7 at the tropopause.
    """

    thrust_sl: float = 39000.0
    tsfc: float = 1.6e-5
    lapse_exponent: float = 0.9

    def validate(self) -> None:
        if self.thrust_sl <= 0:
            raise ValueError("thrust_sl must be positive")
        if self.tsfc <= 0:
            raise ValueError("tsfc must be positive")
        if not 0 < self.lapse_exponent <= 1.5:
            raise ValueError("lapse_exponent must be in (0, 1.5]")

    def updated(self, **changes: float) -> "EngineParams":
        return replace(self, **changes)


@dataclass
class MissionProfile:
    """Definition of the climb / cruise / descent mission.

    The mission is flown the way a real transport is flown: a **speed
    schedule** in the vertical phases (constant calibrated airspeed, then
    constant Mach above the crossover altitude) and a **constant Mach
    level cruise** held by the autothrottle. The flight-path angle is not
    prescribed in the vertical phases; it follows from the specific excess
    power ``sin(gamma) = (T - D) / W``.

    Attributes:
        h_start: Start altitude [m].
        h_cruise: Cruise (level-off) altitude [m].
        h_end: End altitude of the descent [m].
        climb_cas: Calibrated airspeed held during the climb [m/s].
        cruise_mach: Mach number held during the cruise [-].
        descent_cas: Calibrated airspeed held during the descent [m/s].
        max_climb_gamma_deg: Ceiling on the climb flight-path angle [deg];
            the excess thrust near the ground is far larger than the airframe
            rating, so the climb angle is capped there.
        climb_min_gamma_deg: Floor on the climb flight-path angle [deg]. A
            negative value lets an aircraft that has passed its ceiling sink
            while holding its scheduled speed, which is what really happens.
        min_descent_gamma_deg: Floor on the descent flight-path angle [deg].
        climb_throttle: Throttle setting in climb, fraction of max [-].
        descent_throttle: Throttle setting in descent (idle-like) [-].
        stall_patience_s: A climb that gains no altitude for this long is
            declared stalled and the mission is abandoned [s].
        dt: Fixed integration step [s].
        t_max: Hard time limit per phase [s] (guards against no-event runs).
        n_points: Maximum number of stored samples per phase [-].
    """

    h_start: float = 100.0
    h_cruise: float = 11000.0
    h_end: float = 500.0
    climb_cas: float = 150.0
    cruise_mach: float = 0.78
    descent_cas: float = 150.0
    max_climb_gamma_deg: float = 8.0
    climb_min_gamma_deg: float = -10.0
    min_descent_gamma_deg: float = -20.0
    climb_throttle: float = 1.0
    descent_throttle: float = 0.15
    stall_patience_s: float = 600.0
    dt: float = 2.0
    t_max: float = 40000.0
    n_points: int = 400

    def validate(self) -> None:
        if not 0 <= self.h_start < self.h_cruise:
            raise ValueError("require 0 <= h_start < h_cruise")
        if not 0 <= self.h_end < self.h_cruise:
            raise ValueError("require 0 <= h_end < h_cruise")
        if self.climb_cas <= 0:
            raise ValueError("climb_cas must be positive")
        if self.descent_cas <= 0:
            raise ValueError("descent_cas must be positive")
        if not 0.05 < self.cruise_mach < 0.95:
            raise ValueError("cruise_mach must be in (0.05, 0.95)")
        if not 0 < self.max_climb_gamma_deg <= 30:
            raise ValueError("max_climb_gamma_deg must be in (0, 30]")
        if not -30 <= self.min_descent_gamma_deg < 0:
            raise ValueError("min_descent_gamma_deg must be in [-30, 0)")
        for name in ("climb_throttle", "descent_throttle"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError("%s must be in [0, 1]" % name)
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.t_max <= 0:
            raise ValueError("t_max must be positive")
        if self.n_points < 2:
            raise ValueError("n_points must be at least 2")

    def updated(self, **changes: float) -> "MissionProfile":
        return replace(self, **changes)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)
