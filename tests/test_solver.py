"""The mission solver itself: structure, physics and analytic cross-checks."""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flighttrajsim import (  # noqa: E402
    AircraftParams,
    AircraftTrajSim,
    EngineParams,
    MissionProfile,
)
from flighttrajsim.atmosphere import (  # noqa: E402
    G0,
    density,
    density_ratio,
    speed_of_sound,
)
from flighttrajsim.core import CLIMB, CRUISE, DESCENT  # noqa: E402
from flighttrajsim.integrate import integrate_phase, rk4_step  # noqa: E402


@pytest.fixture(scope="module")
def result():
    return AircraftTrajSim().run_simulation()


@pytest.fixture(scope="module")
def sim():
    return AircraftTrajSim()


# --------------------------------------------------------------- structure --
def test_three_phases_in_order(result):
    assert [name for name, _i0, _i1 in result.phases] == [CLIMB, CRUISE, DESCENT]


def test_phase_index_ranges_tile_the_history(result):
    total = len(result.t)
    assert result.phases[0][1] == 0
    assert result.phases[-1][2] == total
    for (_n0, _a0, b0), (_n1, a1, _b1) in zip(result.phases, result.phases[1:]):
        assert a1 == b0


def test_histories_have_consistent_lengths(result):
    lengths = {len(result.t), len(result.h), len(result.v), len(result.x), len(result.m)}
    assert len(lengths) == 1


def test_state_views_alias_the_arrays(result):
    assert result.altitude is result.h
    assert result.y[0] is result.h
    assert result.y[3] is result.m


# ----------------------------------------------------------------- physics --
def test_mission_completes(result):
    assert result.completed is True, result.message


def test_range_is_physically_plausible(result):
    assert 500.0 < result.range_km < 12000.0


def test_endurance_is_physically_plausible(result):
    assert 30.0 < result.endurance_min < 900.0


def test_fuel_burn_matches_the_usable_fuel(result):
    usable = AircraftParams().fuel_mass - AircraftParams().reserve_fuel
    assert abs(result.fuel_burned_kg - usable) < 150.0


def test_descends_to_the_end_altitude(result):
    assert result.h[-1] == pytest.approx(MissionProfile().h_end, abs=50.0)


def test_climb_levels_off_at_the_cruise_altitude(result):
    climb = result.phase_slice(CLIMB)
    assert result.h[climb[1] - 1] == pytest.approx(MissionProfile().h_cruise, abs=5.0)


def test_cruise_is_level(result):
    cruise = result.phase_slice(CRUISE)
    segment = result.h[cruise[0]:cruise[1]]
    assert max(segment) - min(segment) < 1e-6


def test_altitude_stays_positive(result):
    assert min(result.h) >= 0.0


def test_mass_decreases_monotonically(result):
    assert all(a >= b for a, b in zip(result.m, result.m[1:]))


def test_range_increases_monotonically(result):
    assert all(a <= b for a, b in zip(result.x, result.x[1:]))


def test_never_stalls(result):
    assert result.max_cl < AircraftParams().cl_max


def test_cruise_throttle_is_inside_the_envelope(result):
    assert 0.0 < result.cruise_throttle_start < 1.0


def test_cruise_mach_is_held(result):
    cruise = result.phase_slice(CRUISE)
    target = MissionProfile().cruise_mach * speed_of_sound(MissionProfile().h_cruise)
    for v in result.v[cruise[0]:cruise[1]]:
        assert v == pytest.approx(target, rel=1e-9)


def test_climb_holds_the_scheduled_speed(result):
    """Constant CAS off the deck, constant Mach by the top of climb.

    TAS rises through the CAS-limited segment and then eases back slightly
    above the crossover altitude, because the speed of sound falls with
    temperature all the way to the tropopause.
    """
    ms = MissionProfile()
    climb = result.phase_slice(CLIMB)
    airspeed = result.v[climb[0]:climb[1]]
    assert airspeed[0] == pytest.approx(
        ms.climb_cas / math.sqrt(density_ratio(ms.h_start)), rel=1e-9
    )
    assert airspeed[-1] == pytest.approx(
        ms.cruise_mach * speed_of_sound(ms.h_cruise), rel=1e-9
    )
    assert max(airspeed) >= airspeed[-1] - 1e-9
    assert max(airspeed) / airspeed[0] > 1.2


def test_climb_angle_stays_within_the_airframe_limit(sim):
    """Near the ground the excess thrust exceeds what the schedule allows."""
    state = sim.steady_flight_path_angle(
        h=100.0, v=150.0, mass=12000.0, throttle=1.0,
        gamma_min=math.radians(-10.0), gamma_max=math.radians(8.0),
    )
    assert math.degrees(state.gamma) == pytest.approx(8.0, abs=1e-6)
    assert state.angle_limited is True
    # and the metered thrust reflects the limited angle, not full throttle
    assert state.thrust < state.thrust_available


def test_specific_excess_power_drives_the_climb_angle(sim):
    """When the angle is not limited, sin(gamma) == (T - D) / W exactly."""
    state = sim.steady_flight_path_angle(
        h=9000.0, v=220.0, mass=11000.0, throttle=1.0,
        gamma_min=math.radians(-10.0), gamma_max=math.radians(8.0),
    )
    if not state.angle_limited:
        assert math.sin(state.gamma) == pytest.approx(state.excess_power_ratio, rel=1e-6)


def test_cruise_point_trims_thrust_to_drag(sim):
    point = sim.cruise_point(11000.0)
    assert point.thrust == pytest.approx(point.drag, rel=1e-12)
    assert 0.0 < point.throttle < 1.0


# ------------------------------------------------------- analytic agreement --
def test_cruise_segment_matches_breguet(sim, result):
    """The integrated cruise must reproduce the closed-form Breguet range."""
    cruise = result.phase_slice(CRUISE)
    integrated = (result.x[cruise[1] - 1] - result.x[cruise[0]]) / 1000.0
    analytic = sim.breguet_range_estimate(
        mass_start=result.m[cruise[0]], mass_end=result.m[cruise[1] - 1]
    ) / 1000.0
    assert integrated == pytest.approx(analytic, rel=0.01)


def test_cruise_equilibrium_brackets_the_cruise_speed(sim, result):
    """The flown cruise speed must sit on the fast (stable) branch."""
    cruise = result.phase_slice(CRUISE)
    flown = result.v[cruise[0]]
    equilibrium = sim.cruise_equilibrium(mass=result.m[cruise[0]])
    assert equilibrium is not None
    _slow, fast = equilibrium
    assert flown <= fast * 1.05


# ------------------------------------------------------------- sensitivity --
def test_lighter_aircraft_fly_further():
    light = AircraftTrajSim(aircraft=AircraftParams(m0=10000.0)).run_simulation()
    heavy = AircraftTrajSim(aircraft=AircraftParams(m0=15000.0)).run_simulation()
    assert light.range_km > heavy.range_km


@pytest.mark.parametrize("cd0", [0.030, 0.040, 0.050])
def test_more_parasite_drag_costs_range(cd0):
    base = AircraftTrajSim().run_simulation()
    worse = AircraftTrajSim(aircraft=AircraftParams(cd0=cd0)).run_simulation()
    assert worse.range_km < base.range_km


def test_more_induced_drag_costs_range():
    base = AircraftTrajSim().run_simulation()
    worse = AircraftTrajSim(aircraft=AircraftParams(k=0.070)).run_simulation()
    assert worse.range_km < base.range_km


def test_higher_tsfc_costs_range():
    base = AircraftTrajSim().run_simulation()
    worse = AircraftTrajSim(engine=EngineParams(tsfc=2.4e-5)).run_simulation()
    assert worse.range_km < base.range_km


def test_more_fuel_buys_range():
    base = AircraftTrajSim().run_simulation()
    more = AircraftTrajSim(aircraft=AircraftParams(fuel_mass=4500.0)).run_simulation()
    assert more.range_km > base.range_km


def test_a_lower_cruise_altitude_costs_range():
    base = AircraftTrajSim().run_simulation()
    lower = AircraftTrajSim(
        mission=MissionProfile(h_cruise=6000.0)
    ).run_simulation()
    assert lower.range_km < base.range_km


def test_range_is_optimal_near_the_best_lift_to_drag_wing_area():
    """At fixed speed and altitude the range peaks at CL = sqrt(CD0/k)."""
    areas = list(range(12, 47, 2))
    ranges = [
        AircraftTrajSim(aircraft=AircraftParams(S=float(s))).run_simulation().range_km
        for s in areas
    ]
    best = areas[ranges.index(max(ranges))]
    cl_opt = math.sqrt(AircraftParams().cd0 / AircraftParams().k)
    v = MissionProfile().cruise_mach * speed_of_sound(MissionProfile().h_cruise)
    q = 0.5 * density(MissionProfile().h_cruise) * v * v
    theory = AircraftParams().m0 * G0 / (q * cl_opt)
    assert abs(best - theory) / theory < 0.30


# ----------------------------------------------------- graceful degradation --
def test_under_powered_aircraft_is_reported_not_raised():
    sim = AircraftTrajSim(engine=EngineParams(thrust_sl=11000.0))
    result = sim.run_simulation()
    assert result.completed is False
    assert result.message
    assert len(result.t) > 1


def test_an_unreachable_cruise_altitude_is_reported():
    sim = AircraftTrajSim(mission=MissionProfile(h_cruise=20000.0))
    result = sim.run_simulation()
    assert result.completed is False
    assert "climb" in result.message.lower()


def test_a_stalled_climb_never_burns_more_fuel_than_it_carries():
    """Guard against the classic bug: flying at the ceiling until t_max."""
    sim = AircraftTrajSim(aircraft=AircraftParams(cd0=0.12))
    result = sim.run_simulation()
    assert result.fuel_burned_kg <= AircraftParams().fuel_mass + 1.0


def test_sweep_rows_survive_infeasible_corners():
    sim = AircraftTrajSim()
    rows = sim.sweep("thrust_sl", [11000.0, 22000.0, 39000.0, 60000.0])
    assert len(rows) == 4
    assert any(not row["completed"] for row in rows)
    assert all(math.isfinite(row["range_km"]) for row in rows)


def test_parameter_sweep_rejects_unknown_keys():
    with pytest.raises(ValueError):
        AircraftTrajSim().parameter_sweep([{"wingspan": 30.0}])


# ------------------------------------------------------------- validation --
@pytest.mark.parametrize(
    "changes",
    [
        {"m0": -1.0},
        {"fuel_mass": 0.0},
        {"reserve_fuel": 5000.0},
        {"S": 0.0},
        {"cd0": -0.1},
    ],
)
def test_invalid_aircraft_parameters_raise(changes):
    with pytest.raises(ValueError):
        AircraftParams().updated(**changes).validate()


def test_invalid_mission_raises():
    with pytest.raises(ValueError):
        MissionProfile(h_cruise=-100.0).validate()
    with pytest.raises(ValueError):
        MissionProfile(cruise_mach=1.4).validate()


# ---------------------------------------------------------------- numerics --
def test_rk4_integrates_exponentials_exactly_to_fourth_order():
    """RK4 reproduces e^t to ~1e-6 with a 0.1 s step."""
    y = [1.0]
    dt, steps = 0.1, 10
    for _ in range(steps):
        y = rk4_step(lambda s: [s[0]], y, dt)
    assert y[0] == pytest.approx(math.exp(1.0), rel=1e-6)


def test_integrate_phase_lands_on_the_event():
    sol = integrate_phase(
        lambda y: [1.0], [0.0], dt=0.25, t_max=100.0,
        watch=lambda y: y[0] - 3.0, rising=True,
    )
    assert sol.hit is True
    assert sol.y_end[0] == pytest.approx(3.0, abs=1e-9)
    assert sol.t_end == pytest.approx(3.0, abs=1e-9)


def test_integrate_phase_reports_a_missed_event():
    sol = integrate_phase(
        lambda y: [0.0], [0.0], dt=1.0, t_max=5.0,
        watch=lambda y: y[0] - 3.0, rising=True,
    )
    assert sol.hit is False
    assert sol.message


def test_integrate_phase_honours_the_abort_guard():
    sol = integrate_phase(
        lambda y: [1.0], [0.0], dt=1.0, t_max=100.0,
        watch=lambda y: y[0] - 50.0, rising=True,
        abort=lambda y: y[0] >= 3.0,
    )
    assert sol.hit is False
    assert sol.y_end[0] <= 4.0
