"""Mission solver: climb / cruise / descent integration of the point-mass model.

Formulation
-----------
The aircraft is treated as a point mass in coordinated, zero-wind flight.
Two equivalent descriptions are used, one per family of mission phases.

**Vertical phases (climb, descent).** The speed is *prescribed* by a standard
schedule -- constant calibrated airspeed, then constant Mach above the
crossover altitude -- and the flight-path angle follows from the *specific
excess power*::

    sin(gamma) = (T - D) / W
    dh/dt = V sin(gamma)
    dx/dt = V cos(gamma)
    dm/dt = -TSFC * T

This is the classical "steady climb" approximation: the autothrottle and the
autopilot hold the scheduled speed, so every Newton of excess thrust is
converted into altitude (or, when negative, into a descent). Prescribing the
speed instead of the angle is what makes the model robust -- a fixed climb
angle of a few degrees is simply not achievable in the thin air at the
tropopause, and a model that insists on it produces an aircraft that
decelerates into a stall.

**Cruise.** Level flight at constant altitude and constant Mach number, with
the throttle trimmed so that ``T = D`` exactly::

    dx/dt = V
    dm/dt = -TSFC * D

Because the mass (and therefore the drag) falls slowly, this is Breguet's
equation integrated numerically rather than in closed form -- which is why
:meth:`AircraftTrajSim.breguet_range_estimate` reproduces the simulated
cruise distance to within a fraction of a percent.

The three phases are joined by terminal events (level-off, fuel reserve,
descent floor) located by linear interpolation of the last integration step.

Everything is pure standard-library Python. The JavaScript build in ``web/``
uses the identical scheme, so the two implementations agree to machine
precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .aero import (
    breguet_range,
    drag_coefficient,
    dynamic_pressure,
    max_lift_to_drag,
)
from .atmosphere import (
    G0,
    density,
    density_ratio,
    mach as mach_of,
    scheduled_speed,
    speed_of_sound,
)
from .engine import available_thrust, cruise_equilibrium_speed, fuel_flow
from .integrate import PhaseSolution, integrate_phase, thin
from .params import AircraftParams, EngineParams, MissionProfile

CLIMB = "climb"
CRUISE = "cruise"
DESCENT = "descent"

PHASE_ORDER = (CLIMB, CRUISE, DESCENT)


@dataclass
class Instant:
    """Aerodynamic and propulsive state at one point of the trajectory."""

    h: float
    v: float
    mass: float
    gamma: float = 0.0
    mach: float = 0.0
    cl: float = 0.0
    cd: float = 0.0
    lift: float = 0.0
    drag: float = 0.0
    thrust: float = 0.0
    thrust_available: float = 0.0
    throttle: float = 0.0
    ld: float = 0.0
    fuel_flow: float = 0.0
    stalled: bool = False
    angle_limited: bool = False

    @property
    def excess_power_ratio(self) -> float:
        """``(T_max - D) / W`` -- the specific excess power available."""
        weight = self.mass * G0
        return (self.thrust_available - self.drag) / weight if weight > 0 else 0.0

    @property
    def climb_rate(self) -> float:
        """Rate of climb ``dh/dt = V sin(gamma)`` [m/s]."""
        return self.v * math.sin(self.gamma)


@dataclass
class TrajectoryResult:
    """Outcome of a mission simulation.

    All histories are plain Python lists of floats, indexed identically:
    ``result.altitude[i]`` is the altitude at ``result.t[i]``.
    """

    t: List[float] = field(default_factory=list)
    h: List[float] = field(default_factory=list)
    v: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    m: List[float] = field(default_factory=list)
    phases: List[Tuple[str, int, int]] = field(default_factory=list)
    completed: bool = True
    message: str = ""

    # Diagnostics filled by the solver -----------------------------------
    cruise_throttle_start: float = float("nan")
    cruise_throttle_end: float = float("nan")
    max_cl: float = float("nan")
    stalled: bool = False
    ceiling_m: float = float("nan")

    # --- views for backward compatibility --------------------------------
    @property
    def y(self) -> List[List[float]]:
        """State history as ``[altitude, velocity, range, mass]``."""
        return [self.h, self.v, self.x, self.m]

    @property
    def altitude(self) -> List[float]:
        return self.h

    @property
    def velocity(self) -> List[float]:
        return self.v

    @property
    def range(self) -> List[float]:
        return self.x

    @property
    def mass(self) -> List[float]:
        return self.m

    @property
    def mach(self) -> List[float]:
        return [mach_of(h, v) for h, v in zip(self.h, self.v)]

    # --- scalar metrics --------------------------------------------------
    @property
    def endurance_s(self) -> float:
        return self.t[-1] if self.t else 0.0

    @property
    def endurance_min(self) -> float:
        return self.endurance_s / 60.0

    @property
    def range_m(self) -> float:
        return self.x[-1] if self.x else 0.0

    @property
    def range_km(self) -> float:
        return self.range_m / 1000.0

    @property
    def fuel_burned_kg(self) -> float:
        return (self.m[0] - self.m[-1]) if self.m else 0.0

    def phase_slice(self, name: str) -> Optional[Tuple[int, int]]:
        """Index range ``(i0, i1)`` of a named phase, or ``None``."""
        for pname, i0, i1 in self.phases:
            if pname == name:
                return i0, i1
        return None

    def summary(self) -> Dict[str, float]:
        """Aggregate mission metrics as plain floats."""
        cruise = self.phase_slice(CRUISE)
        cruise_km = 0.0
        if cruise is not None:
            cruise_km = (self.x[cruise[1] - 1] - self.x[cruise[0]]) / 1000.0
        return {
            "range_km": self.range_km,
            "cruise_range_km": cruise_km,
            "endurance_s": self.endurance_s,
            "endurance_min": self.endurance_min,
            "fuel_burned_kg": self.fuel_burned_kg,
            "final_altitude_m": self.h[-1] if self.h else 0.0,
            "final_velocity_mps": self.v[-1] if self.v else 0.0,
            "final_mass_kg": self.m[-1] if self.m else 0.0,
            "max_altitude_m": max(self.h) if self.h else 0.0,
            "cruise_throttle_start": self.cruise_throttle_start,
            "cruise_throttle_end": self.cruise_throttle_end,
            "max_cl": self.max_cl,
            "completed": bool(self.completed),
        }

    def rows(self) -> List[Dict[str, float]]:
        """One dictionary per sample -- convenient for CSV export."""
        out = []
        names = {name: (i0, i1) for name, i0, i1 in self.phases}
        for i in range(len(self.t)):
            phase = next(
                (n for n, (a, b) in names.items() if a <= i < b), ""
            )
            out.append(
                {
                    "time_s": self.t[i],
                    "altitude_m": self.h[i],
                    "tas_mps": self.v[i],
                    "mach": mach_of(self.h[i], self.v[i]),
                    "range_m": self.x[i],
                    "mass_kg": self.m[i],
                    "phase": phase,
                }
            )
        return out


class AircraftTrajSim:
    """Point-mass trajectory simulator.

    Example:
        >>> sim = AircraftTrajSim()
        >>> result = sim.run_simulation()
        >>> 4000 < result.range_km < 6000
        True
    """

    def __init__(
        self,
        aircraft: Optional[AircraftParams] = None,
        engine: Optional[EngineParams] = None,
        mission: Optional[MissionProfile] = None,
    ) -> None:
        self.aircraft = aircraft or AircraftParams()
        self.engine = engine or EngineParams()
        self.mission = mission or MissionProfile()
        self.validate()

    def validate(self) -> None:
        """Validate every parameter group."""
        self.aircraft.validate()
        self.engine.validate()
        self.mission.validate()

    # ------------------------------------------------------------------
    # Instantaneous state
    # ------------------------------------------------------------------
    @staticmethod
    def _aero_instant(
        h: float, v: float, mass: float, throttle_cmd: float,
        ac: AircraftParams, en: EngineParams,
    ) -> Instant:
        """Aerodynamics and thrust at one point, for a given throttle."""
        rho = density(h)
        q = dynamic_pressure(rho, v)
        weight = mass * G0
        sigma = density_ratio(h)
        thrust = available_thrust(en, sigma, throttle_cmd)
        num = mach_of(h, v)
        cl = (weight / (q * ac.S)) if q > 0 else float("inf")
        cd = drag_coefficient(cl, ac.cd0, ac.k, num, ac.mach_crit)
        drag = q * ac.S * cd
        return Instant(
            h=h, v=v, mass=mass, mach=num, cl=cl, cd=cd,
            lift=q * ac.S * cl, drag=drag, thrust=thrust,
            thrust_available=thrust, throttle=throttle_cmd,
            ld=(cl / cd) if cd > 0 else float("inf"),
            fuel_flow=fuel_flow(thrust, en.tsfc),
            stalled=cl > ac.cl_max,
        )

    def steady_flight_path_angle(
        self,
        h: float, v: float, mass: float, throttle: float,
        gamma_min: float, gamma_max: float,
        ac: Optional[AircraftParams] = None,
        en: Optional[EngineParams] = None,
    ) -> Instant:
        """Quasi-steady flight-path angle from the specific excess power.

        Solves ``sin(gamma) = (T - D(gamma)) / W`` by fixed-point iteration --
        the drag depends on ``gamma`` only through ``L = W cos(gamma)``, so
        four passes are far below the floating-point noise floor -- and clamps
        the result to ``[gamma_min, gamma_max]`` (airframe / procedure limits).

        The **delivered** thrust is then ``min(T_available, D + W sin(gamma))``.
        That distinction matters: when the angle is limited the engines are no
        longer at the commanded setting, and metering fuel at the commanded
        thrust would overstate the burn (and, at the ceiling, let the aircraft
        burn fuel it does not have).
        """
        ac = ac or self.aircraft
        en = en or self.engine
        gamma = 0.0
        state = self._aero_instant(h, v, mass, throttle, ac, en)
        t_available = state.thrust
        for _ in range(4):
            lift = mass * G0 * math.cos(gamma)
            q = dynamic_pressure(density(h), v)
            cl = (lift / (q * ac.S)) if q > 0 else float("inf")
            cd = drag_coefficient(cl, ac.cd0, ac.k, state.mach, ac.mach_crit)
            drag = q * ac.S * cd
            weight = mass * G0
            s_free = (t_available - drag) / weight if weight > 0 else 0.0
            s = min(max(s_free, math.sin(gamma_min)), math.sin(gamma_max))
            gamma = math.asin(min(max(s, -0.999999), 0.999999))
            state.cl = cl
            state.cd = cd
            state.drag = drag
            state.lift = lift
            state.ld = (cl / cd) if cd > 0 else float("inf")
            state.stalled = cl > ac.cl_max
        state.gamma = gamma
        delivered = min(t_available, state.drag + mass * G0 * math.sin(gamma))
        state.thrust = max(delivered, 0.0)
        state.fuel_flow = fuel_flow(state.thrust, en.tsfc)
        state.angle_limited = abs(state.thrust - t_available) > 1e-9
        return state

    def cruise_point(
        self,
        mass: float,
        ac: Optional[AircraftParams] = None,
        en: Optional[EngineParams] = None,
        mission: Optional[MissionProfile] = None,
    ) -> Instant:
        """Trimmed level-cruise state: throttle adjusted so that ``T = D``."""
        ac = ac or self.aircraft
        en = en or self.engine
        ms = mission or self.mission
        h = ms.h_cruise
        v = ms.cruise_mach * speed_of_sound(h)
        state = self._aero_instant(h, v, mass, 1.0, ac, en)
        if state.thrust > 0:
            throttle = state.drag / state.thrust
        else:
            throttle = float("inf")
        state.throttle = throttle
        state.thrust = state.drag
        state.fuel_flow = fuel_flow(state.drag, en.tsfc)
        return state

    # ------------------------------------------------------------------
    # Phase right-hand sides
    # ------------------------------------------------------------------
    def _make_vertical_rhs(
        self, ac: AircraftParams, en: EngineParams, ms: MissionProfile,
        cas: float, throttle: float, gamma_min: float, gamma_max: float,
    ) -> Callable[[Sequence[float]], List[float]]:
        """RHS for a phase with a prescribed speed schedule; state ``[h, x, m]``."""

        def rhs(y: Sequence[float]) -> List[float]:
            h, _x, m = y
            h = max(h, 0.0)
            m = max(m, 1.0)
            v = scheduled_speed(h, cas, ms.cruise_mach)
            state = self.steady_flight_path_angle(
                h, v, m, throttle, gamma_min, gamma_max, ac, en
            )
            return [
                v * math.sin(state.gamma),
                v * math.cos(state.gamma),
                -state.fuel_flow,
            ]

        return rhs

    @staticmethod
    def _make_guard(ac: AircraftParams, ms: MissionProfile, patience: float,
                    rising: bool, phase: str = ""):
        """Physical guards: fuel exhaustion and a climb that stops climbing.

        Without these an infeasible mission -- an under-powered aircraft, say
        -- would quietly fly level at its ceiling until the time limit,
        burning several tankfuls of fuel it does not have.
        """

        class Guard:
            def __init__(self):
                self.fuel_out = False
                self.stalled = False
                self._prev = None
                self._idle = 0.0

            def check(self, y: Sequence[float]) -> bool:
                m = y[2] if len(y) == 3 else y[1]
                if m <= ac.dry_mass:
                    self.fuel_out = True
                    return True
                if phase == CLIMB:
                    h = y[0]
                    if self._prev is not None and h - self._prev < 0.02 * ms.dt:
                        self._idle += ms.dt
                    else:
                        self._idle = 0.0
                    self._prev = h
                    if self._idle >= patience:
                        self.stalled = True
                        return True
                return False

        return Guard()

    def _make_cruise_rhs(
        self, ac: AircraftParams, en: EngineParams, ms: MissionProfile,
    ) -> Callable[[Sequence[float]], List[float]]:
        """RHS for the level cruise; state ``[x, m]`` at fixed altitude."""
        h = ms.h_cruise
        v = ms.cruise_mach * speed_of_sound(h)

        def rhs(y: Sequence[float]) -> List[float]:
            _x, m = y
            m = max(m, 1.0)
            state = self.cruise_point(m, ac, en, ms)
            return [v, -state.fuel_flow]

        return rhs

    # ------------------------------------------------------------------
    # Mission
    # ------------------------------------------------------------------
    def run_simulation(
        self,
        aircraft: Optional[AircraftParams] = None,
        engine: Optional[EngineParams] = None,
        mission: Optional[MissionProfile] = None,
    ) -> TrajectoryResult:
        """Integrate the full climb-cruise-descent mission.

        Returns:
            A :class:`TrajectoryResult`. ``completed`` is False when a phase
            could not reach its terminal condition within ``t_max`` -- an
            under-powered climb that stalls at its ceiling, for instance.
            The partial trajectory is always returned so that parameter
            sweeps degrade gracefully instead of raising.
        """
        ac = aircraft or self.aircraft
        en = engine or self.engine
        ms = mission or self.mission
        ac.validate()
        en.validate()
        ms.validate()

        n = max(int(ms.n_points), 2)
        dt = ms.dt
        result = TrajectoryResult()

        def record(sol: PhaseSolution, offset: float, name: str, speed_of,
                   skip_first: bool = False) -> float:
            """Append a phase to the result and return the new time offset."""
            i0 = len(result.t)
            for i, (ti, yi) in enumerate(zip(sol.t, sol.y)):
                if skip_first and i == 0:
                    continue  # duplicated junction sample
                result.t.append(ti + offset)
                result.h.append(speed_of[0](yi))
                result.v.append(speed_of[1](yi))
                result.x.append(yi[1] if len(yi) == 3 else yi[0])
                result.m.append(yi[2] if len(yi) == 3 else yi[1])
            result.phases.append((name, i0, len(result.t)))
            return offset + sol.t_end

        # ---- climb -----------------------------------------------------
        climb_rhs = self._make_vertical_rhs(
            ac, en, ms, ms.climb_cas, ms.climb_throttle,
            math.radians(ms.climb_min_gamma_deg), math.radians(ms.max_climb_gamma_deg),
        )
        guard = self._make_guard(
            ac, ms, ms.stall_patience_s, rising=True, phase=CLIMB
        )
        climb = integrate_phase(
            climb_rhs, [ms.h_start, 0.0, ac.m0], dt, ms.t_max,
            watch=lambda y: y[0] - ms.h_cruise, rising=True, abort=guard.check,
        )
        climb = thin(climb, n)
        offset = record(
            climb, 0.0, CLIMB,
            (lambda s: s[0], lambda s: scheduled_speed(s[0], ms.climb_cas, ms.cruise_mach)),
        )
        if not climb.hit:
            result.completed = False
            result.ceiling_m = max(s[0] for s in climb.y)
            if guard.fuel_out:
                result.message = (
                    "climb ran out of fuel at %d m, below the cruise altitude "
                    "of %d m" % (int(result.ceiling_m), int(ms.h_cruise))
                )
            elif guard.stalled:
                result.message = (
                    "climb stalled at %d m: the excess thrust is exhausted, so "
                    "the cruise altitude of %d m is out of reach"
                    % (int(result.ceiling_m), int(ms.h_cruise))
                )
            else:
                result.message = (
                    "climb stopped at %d m; the cruise altitude of %d m was "
                    "never reached" % (int(result.ceiling_m), int(ms.h_cruise))
                )
            self._finalise(result, ac)
            return result
        y_now = climb.y_end

        # ---- cruise ----------------------------------------------------
        cruise_rhs = self._make_cruise_rhs(ac, en, ms)
        start = self.cruise_point(y_now[2], ac, en, ms)
        end = self.cruise_point(ac.min_mass, ac, en, ms)
        result.cruise_throttle_start = start.throttle
        result.cruise_throttle_end = end.throttle
        if start.throttle > 1.0:
            result.completed = False
            result.message = (
                "cruise requires a throttle of %.2f at %.0f kg: the installed "
                "thrust cannot hold M=%.2f at %d m"
                % (start.throttle, y_now[2], ms.cruise_mach, int(ms.h_cruise))
            )
        cruise = integrate_phase(
            cruise_rhs, [y_now[1], y_now[2]], dt, ms.t_max,
            watch=lambda y: y[1] - ac.min_mass, rising=False,
        )
        cruise = thin(cruise, n)
        offset = record(
            cruise, offset, CRUISE,
            (lambda s: ms.h_cruise, lambda s: ms.cruise_mach * speed_of_sound(ms.h_cruise)),
            skip_first=True,
        )
        if not cruise.hit:
            result.completed = False
            result.message = (
                "cruise ended on the time limit of %d s before the fuel "
                "reserve of %d kg was reached" % (int(ms.t_max), int(ac.reserve_fuel))
            )
            self._finalise(result, ac)
            return result
        y_now = [y_now[0], cruise.y_end[0], cruise.y_end[1]]

        # ---- descent ---------------------------------------------------
        descent_rhs = self._make_vertical_rhs(
            ac, en, ms, ms.descent_cas, ms.descent_throttle,
            math.radians(ms.min_descent_gamma_deg), 0.0,
        )
        descent_guard = self._make_guard(
            ac, ms, ms.stall_patience_s, rising=False, phase=DESCENT
        )
        descent = integrate_phase(
            descent_rhs, [ms.h_cruise, y_now[1], y_now[2]], dt, ms.t_max,
            watch=lambda y: y[0] - ms.h_end, rising=False,
            abort=descent_guard.check,
        )
        descent = thin(descent, n)
        record(
            descent, offset, DESCENT,
            (lambda s: s[0], lambda s: scheduled_speed(s[0], ms.descent_cas, ms.cruise_mach)),
            skip_first=True,
        )
        if not descent.hit:
            result.completed = False
            result.message = (
                "descent did not reach %d m within the time limit of %d s; "
                "reduce the descent throttle or increase the descent speed"
                % (int(ms.h_end), int(ms.t_max))
            )

        self._finalise(result, ac)
        return result

    @staticmethod
    def _finalise(result: TrajectoryResult, ac: AircraftParams) -> None:
        """Fill post-run diagnostics."""
        if result.h:
            result.max_cl = max(
                mass * G0 / (dynamic_pressure(density(h), v) * ac.S)
                for h, v, mass in zip(result.h, result.v, result.m)
            )
            result.stalled = result.max_cl > ac.cl_max
        if not result.completed and not result.message:
            result.message = "mission incomplete"

    # ------------------------------------------------------------------
    # Sweeps
    # ------------------------------------------------------------------
    def _split_variation(self, changes: Dict[str, float]) -> Tuple[AircraftParams, EngineParams]:
        """Route variation keys to the aircraft or the engine dataclass."""
        ac_fields = set(self.aircraft.__dataclass_fields__)
        en_fields = set(self.engine.__dataclass_fields__)
        ac_over = {k: v for k, v in changes.items() if k in ac_fields}
        en_over = {k: v for k, v in changes.items() if k in en_fields}
        unknown = set(changes) - ac_fields - en_fields
        if unknown:
            raise ValueError("unknown parameter(s): %s" % ", ".join(sorted(unknown)))
        return self.aircraft.updated(**ac_over), self.engine.updated(**en_over)

    def parameter_sweep(
        self, variations: Sequence[Dict[str, float]], mission: Optional[MissionProfile] = None
    ) -> List[Dict[str, float]]:
        """Run one mission per parameter variation.

        Args:
            variations: sequence of dicts such as ``[{"m0": 11000}, {"cd0": 0.03}]``.
            mission: optional mission override shared by all runs.

        Returns:
            One row per run containing the varied parameters plus the mission
            metrics. Runs that abort still produce a row (with ``completed``
            False) so that plotting never breaks on a bad corner of the design
            space.
        """
        rows: List[Dict[str, float]] = []
        for changes in variations:
            ac, en = self._split_variation(changes)
            result = self.run_simulation(aircraft=ac, engine=en, mission=mission)
            row: Dict[str, float] = dict(changes)
            row.update(
                {
                    "mass": ac.m0,
                    "range": result.range_m,
                    "time": result.endurance_s,
                    "range_km": result.range_km,
                    "endurance_s": result.endurance_s,
                    "endurance_min": result.endurance_min,
                    "fuel_burned_kg": result.fuel_burned_kg,
                    "cruise_throttle": result.cruise_throttle_start,
                    "completed": float(bool(result.completed)),
                }
            )
            rows.append(row)
        return rows

    def sweep(
        self, param: str, values: Sequence[float], mission: Optional[MissionProfile] = None
    ) -> List[Dict[str, float]]:
        """Sweep a single named parameter over ``values``."""
        return self.parameter_sweep([{param: float(v)} for v in values], mission=mission)

    def parameter_sweep_mass(
        self, mass_list: Sequence[float], mission: Optional[MissionProfile] = None
    ) -> List[Dict[str, float]]:
        """Sweep take-off mass (kept from the original API)."""
        return self.sweep("m0", mass_list, mission=mission)

    # ------------------------------------------------------------------
    # Analytic cross-checks
    # ------------------------------------------------------------------
    def cruise_equilibrium(
        self, mission: Optional[MissionProfile] = None, mass: Optional[float] = None
    ) -> Optional[Tuple[float, float]]:
        """Level-flight equilibrium speeds at the cruise altitude [m/s].

        Returns ``(v_slow, v_fast)`` -- the two roots of the thrust-required
        curve -- or ``None`` when the throttle cannot overcome minimum drag.
        """
        ms = mission or self.mission
        ac, en = self.aircraft, self.engine
        rho = density(ms.h_cruise)
        sigma = density_ratio(ms.h_cruise)
        thrust = available_thrust(en, sigma, 1.0)
        return cruise_equilibrium_speed(
            mass or ac.m0, rho, ac.S, ac.cd0, ac.k, thrust
        )

    def breguet_range_estimate(
        self,
        mission: Optional[MissionProfile] = None,
        mass_start: Optional[float] = None,
        mass_end: Optional[float] = None,
    ) -> float:
        """Closed-form Breguet cruise range [m] for the same conditions.

        The simulated cruise segment should match this to well under one
        percent; the test suite asserts that agreement.
        """
        ms = mission or self.mission
        ac, en = self.aircraft, self.engine
        m0 = mass_start if mass_start is not None else ac.m0
        m1 = mass_end if mass_end is not None else ac.min_mass
        v = ms.cruise_mach * speed_of_sound(ms.h_cruise)
        # The L/D drifts slowly with mass; evaluate it at the geometric mean.
        m_mean = math.sqrt(max(m0 * m1, 1.0))
        state = self.cruise_point(m_mean, ac, en, ms)
        return breguet_range(v, state.ld, en.tsfc, m0, m1)

    def best_glide(self) -> Tuple[float, float]:
        """Best lift-to-drag ratio and the CL where it occurs."""
        ac = self.aircraft
        return max_lift_to_drag(ac.cd0, ac.k)
