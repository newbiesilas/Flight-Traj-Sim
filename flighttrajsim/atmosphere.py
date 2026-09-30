"""International Standard Atmosphere (ISA) model.

Implements the troposphere / lower stratosphere layers of ICAO Doc 7488 and
the US Standard Atmosphere 1976::

    0 – 11 km   : linear temperature lapse, dT/dH = -6.5 K/km
    11 – 20 km  : isothermal, T = 216.65 K

Both the layer boundaries and the published tables are expressed in
**geopotential** altitude ``H = r h / (r + h)``, which is what the ISA uses
internally; the public functions here take *geometric* altitude ``h`` and do
the conversion, so ``density(11019.0)`` returns the tabulated 11 km value.
Ignoring the conversion shifts the density by ~0.3 % at the tropopause and
~1 % at 20 km -- small, but visible against the standard tables.
"""

from __future__ import annotations

import math

# --- ISA constants -------------------------------------------------------
G0 = 9.80665          # standard gravity, m/s^2
R_AIR = 287.05287     # specific gas constant for dry air, J/(kg K)
GAMMA_AIR = 1.4       # ratio of specific heats

T0 = 288.15           # sea-level temperature, K
P0 = 101325.0         # sea-level pressure, Pa
RHO0 = 1.225          # sea-level density, kg/m^3

LAPSE = -0.0065       # tropospheric temperature lapse rate, K/m
H_TROP = 11000.0      # tropopause, **geopotential** altitude, m
T_TROP = 216.65       # temperature at tropopause, K

R_EARTH = 6356766.0   # effective Earth radius for geopotential conversion, m

# Derived constants
_EXP_TROP = G0 / (R_AIR * -LAPSE)          # pressure exponent in the troposphere
P_TROP = P0 * (T_TROP / T0) ** _EXP_TROP   # pressure at the tropopause, Pa
RHO_TROP = P_TROP / (R_AIR * T_TROP)       # density at the tropopause, kg/m^3

# Scale height of the isothermal layer, R T / g0 = 6341.6 m
_H_ISO = R_AIR * T_TROP / G0


def geopotential_altitude(h: float) -> float:
    """Convert geometric altitude ``h`` [m] to geopotential altitude ``H`` [m].

    ``H = r * h / (r + h)`` with ``r`` the effective Earth radius. The
    correction is small (35 m at 15 km) but visible in the density tables.
    """
    return R_EARTH * h / (R_EARTH + h)


def temperature(h: float) -> float:
    """ISA temperature [K] at geometric altitude ``h`` [m]."""
    return temperature_at_geopotential(geopotential_altitude(h))


def pressure(h: float) -> float:
    """ISA pressure [Pa] at geometric altitude ``h`` [m]."""
    H = geopotential_altitude(h)
    if H < H_TROP:
        return P0 * (temperature_at_geopotential(H) / T0) ** _EXP_TROP
    return P_TROP * math.exp(-(H - H_TROP) / _H_ISO)


def temperature_at_geopotential(H: float) -> float:
    """ISA temperature [K] at **geopotential** altitude ``H`` [m]."""
    return T0 + LAPSE * H if H < H_TROP else T_TROP


def density(h: float) -> float:
    """ISA density [kg/m^3] at geometric altitude ``h`` [m]."""
    return pressure(h) / (R_AIR * temperature(h))


def density_ratio(h: float) -> float:
    """Density ratio ``rho / rho0`` (``sigma``), used for thrust lapse."""
    return density(h) / RHO0


def speed_of_sound(h: float) -> float:
    """Speed of sound [m/s] at geometric altitude ``h`` [m]."""
    return math.sqrt(GAMMA_AIR * R_AIR * temperature(h))


def mach(h: float, v: float) -> float:
    """Mach number for true airspeed ``v`` [m/s] at altitude ``h`` [m]."""
    return v / speed_of_sound(h)


def equivalent_airspeed(h: float, v: float) -> float:
    """Equivalent (calibrated) airspeed [m/s] for a true airspeed ``v``.

    ``V_cas ~ V * sqrt(rho / rho0)`` (compressibility ignored, valid for the
    subsonic/low-transonic range this simulator targets).
    """
    return v * math.sqrt(max(density_ratio(h), 1e-9))


def scheduled_speed(h: float, cas: float, mach_cruise: float) -> float:
    """True airspeed [m/s] of the standard climb / descent schedule.

    Real transport aircraft climb at a constant *indicated* speed and then
    hold a constant Mach number once the two limits cross (the "crossover
    altitude", typically 8-9 km). This function reproduces that schedule::

        V(h) = min( CAS / sqrt(sigma) ,  M_cr * a(h) )
    """
    sigma = max(density_ratio(h), 1e-9)
    return min(cas / math.sqrt(sigma), mach_cruise * speed_of_sound(h))
