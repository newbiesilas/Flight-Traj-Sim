"""Aerodynamic model: the parabolic polar and its closed-form properties."""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flighttrajsim.aero import (  # noqa: E402
    aero_state,
    breguet_range,
    drag_coefficient,
    dynamic_pressure,
    lift_coefficient,
    max_lift_to_drag,
    wave_drag_coefficient,
)
from flighttrajsim.atmosphere import G0, density  # noqa: E402

CD0 = 0.025
K = 0.048
S = 24.0


def test_dynamic_pressure():
    assert dynamic_pressure(1.225, 100.0) == pytest.approx(6125.0)


def test_lift_coefficient_level_flight():
    q = dynamic_pressure(density(11000.0), 230.0)
    cl = lift_coefficient(10000.0, 0.0, q, S)
    assert cl == pytest.approx(10000.0 * G0 / (q * S), rel=1e-12)


def test_lift_coefficient_uses_cosine_in_a_climb():
    q = dynamic_pressure(density(5000.0), 180.0)
    level = lift_coefficient(10000.0, 0.0, q, S)
    climbing = lift_coefficient(10000.0, math.radians(10.0), q, S)
    assert climbing < level
    assert climbing == pytest.approx(level * math.cos(math.radians(10.0)), rel=1e-12)


def test_drag_polar_is_parabolic():
    for cl in (0.2, 0.5, 0.8):
        cd = drag_coefficient(cl, CD0, K, 0.0, 0.82)
        assert cd == pytest.approx(CD0 + K * cl * cl, rel=1e-12)


def test_wave_drag_is_zero_below_the_critical_mach():
    assert wave_drag_coefficient(0.80, 0.82) == 0.0
    assert wave_drag_coefficient(0.82, 0.82) == 0.0


def test_wave_drag_rises_above_the_critical_mach():
    assert wave_drag_coefficient(0.90, 0.82) > 0.0
    assert wave_drag_coefficient(0.95, 0.82) > wave_drag_coefficient(0.90, 0.82)


def test_best_glide_ratio_matches_the_closed_form():
    cl_opt, ld_max = max_lift_to_drag(CD0, K)
    assert cl_opt == pytest.approx(math.sqrt(CD0 / K), rel=1e-12)
    assert ld_max == pytest.approx(1.0 / (2.0 * math.sqrt(CD0 * K)), rel=1e-12)


def test_best_glide_ratio_is_reproduced_by_the_polar():
    """(L/D)max really does occur at CL = sqrt(CD0/k)."""
    cl_opt, ld_max = max_lift_to_drag(CD0, K)
    rho = density(11000.0)
    for cl in [cl_opt * f for f in (0.6, 0.8, 1.0, 1.2, 1.4)]:
        if cl <= 0:
            continue
        # Pick the speed that produces this CL for a fixed weight.
        weight = 10000.0 * G0
        v = math.sqrt(2.0 * weight / (rho * S * cl))
        state = aero_state(
            mass=10000.0, gamma=0.0, v=v, rho=rho, s=S, cd0=CD0, k=K,
            cl_max=1.4, mach_crit=0.82, mach=0.0,
        )
        assert state.ld <= ld_max + 1e-9
    # and the optimum point reaches it
    v_opt = math.sqrt(2.0 * 10000.0 * G0 / (rho * S * cl_opt))
    best = aero_state(
        mass=10000.0, gamma=0.0, v=v_opt, rho=rho, s=S, cd0=CD0, k=K,
        cl_max=1.4, mach_crit=0.82, mach=0.0,
    )
    assert best.ld == pytest.approx(ld_max, rel=1e-6)


def test_aero_state_flags_stall():
    rho = density(11000.0)
    slow = aero_state(
        mass=15000.0, gamma=0.0, v=90.0, rho=rho, s=S, cd0=CD0, k=K,
        cl_max=1.0, mach_crit=0.82, mach=0.0,
    )
    assert slow.stalled is True


def test_breguet_range_scales_logarithmically_with_fuel():
    v, ld, tsfc = 230.0, 14.0, 1.6e-5
    r_one = breguet_range(v, ld, tsfc, 12000.0, 11000.0)
    r_two = breguet_range(v, ld, tsfc, 12000.0, 10000.0)
    ratio = math.log(12000.0 / 10000.0) / math.log(12000.0 / 11000.0)
    assert r_two / r_one == pytest.approx(ratio, rel=1e-9)


def test_breguet_range_zero_fuel_is_zero():
    assert breguet_range(230.0, 14.0, 1.6e-5, 10000.0, 10000.0) == pytest.approx(0.0)
