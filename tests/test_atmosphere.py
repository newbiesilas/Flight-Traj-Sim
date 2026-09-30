"""International Standard Atmosphere against the tabulated ICAO values."""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flighttrajsim.atmosphere import (  # noqa: E402
    R_EARTH,
    density,
    density_ratio,
    geopotential_altitude,
    pressure,
    scheduled_speed,
    speed_of_sound,
    temperature,
)

# (geopotential altitude m, T K, p Pa, rho kg/m^3)
# ICAO Doc 7488 / US Standard Atmosphere 1976. The tables are tabulated
# against *geopotential* altitude, so the tests convert before comparing --
# which exercises the geometric/geopotential conversion at the same time.
ISA_TABLE = [
    (0, 288.15, 101325.0, 1.22500),
    (1000, 281.65, 89874.6, 1.11164),
    (2000, 275.15, 79495.2, 1.00649),
    (5000, 255.65, 54019.9, 0.73612),
    (8000, 236.15, 35599.8, 0.52517),
    (11000, 216.65, 22632.0, 0.36392),
    (15000, 216.65, 12044.6, 0.19367),
    (20000, 216.65, 5474.9, 0.08803),
]


def geometric(h_geopotential: float) -> float:
    """Inverse of :func:`geopotential_altitude`."""
    return R_EARTH * h_geopotential / (R_EARTH - h_geopotential)


@pytest.mark.parametrize("H,t_expected,_p,_rho", ISA_TABLE)
def test_temperature(H, t_expected, _p, _rho):
    assert temperature(geometric(H)) == pytest.approx(t_expected, abs=0.35)


@pytest.mark.parametrize("H,_t,p_expected,_rho", ISA_TABLE)
def test_pressure(H, _t, p_expected, _rho):
    assert pressure(geometric(H)) == pytest.approx(p_expected, rel=5e-3)


@pytest.mark.parametrize("H,_t,_p,rho_expected", ISA_TABLE)
def test_density(H, _t, _p, rho_expected):
    assert density(geometric(H)) == pytest.approx(rho_expected, rel=5e-3)


def test_density_ratio_at_sea_level():
    assert density_ratio(0.0) == pytest.approx(1.0, rel=1e-3)


def test_speed_of_sound_standard_values():
    assert speed_of_sound(0.0) == pytest.approx(340.29, abs=0.3)
    assert speed_of_sound(11000.0) == pytest.approx(295.07, abs=0.3)


def test_isothermal_above_the_tropopause():
    assert temperature(12000.0) == pytest.approx(temperature(16000.0), abs=1e-9)
    assert temperature(12000.0) == pytest.approx(216.65, abs=1e-9)
    assert pressure(12000.0) < pressure(11000.0)


def test_geopotential_round_trip():
    for h in (0.0, 1000.0, 11000.0, 20000.0):
        assert geopotential_altitude(geometric(h)) == pytest.approx(h, rel=1e-9)


def test_density_monotonically_decreases():
    values = [density(h) for h in range(0, 20000, 250)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_geopotential_is_below_geometric():
    assert geopotential_altitude(15000.0) < 15000.0
    assert geopotential_altitude(0.0) == 0.0


def test_scheduled_speed_crossover():
    """Constant CAS below the crossover altitude, constant Mach above it."""
    cas, mach = 150.0, 0.78
    low = scheduled_speed(1000.0, cas, mach)
    high = scheduled_speed(11000.0, cas, mach)
    # Below the crossover the CAS limit rules, so TAS grows with altitude ...
    assert low == pytest.approx(cas / math.sqrt(density_ratio(1000.0)), rel=1e-12)
    assert low > cas
    # ... and above it the Mach limit takes over.
    assert high == pytest.approx(mach * speed_of_sound(11000.0), rel=1e-12)
    assert high == pytest.approx(0.78 * 295.07, rel=1e-3)


def test_scheduled_speed_is_the_lower_of_the_two_limits():
    cas, mach = 150.0, 0.78
    for h in range(0, 15000, 500):
        v = scheduled_speed(float(h), cas, mach)
        assert v <= cas / math.sqrt(density_ratio(float(h))) + 1e-9
        assert v <= mach * speed_of_sound(float(h)) + 1e-9
