"""Minimal fixed-step Runge-Kutta integrator with terminal events.

The whole simulator is deliberately **dependency free**: no NumPy, no SciPy.
That keeps the Python package and the JavaScript build byte-for-byte
equivalent (both use the very same RK4 scheme), makes the packaged desktop
application small, and lets the test suite run anywhere.

The terminal event is located by a linear interpolation of the watched
quantity between the last two steps, which is second-order accurate for a
smoothly crossing function and is far more robust than a bisection loop on a
noisy right-hand side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

State = List[float]
RHS = Callable[[State], State]


@dataclass
class PhaseSolution:
    """Result of integrating one mission phase.

    Attributes:
        t: Sample times measured from the start of the phase [s].
        y: State history, ``y[i]`` is the state at ``t[i]``.
        hit: True when the terminal event fired.
        message: Human readable note, empty on success.
    """

    t: List[float] = field(default_factory=list)
    y: List[State] = field(default_factory=list)
    hit: bool = False
    message: str = ""

    @property
    def t_end(self) -> float:
        """Time of the last sample [s]."""
        return self.t[-1] if self.t else 0.0

    @property
    def y_end(self) -> State:
        """State at the last sample."""
        return list(self.y[-1]) if self.y else []


def rk4_step(rhs: RHS, y: Sequence[float], dt: float) -> State:
    """One classical fourth-order Runge-Kutta step."""
    y0 = list(y)
    k1 = rhs(y0)
    k2 = rhs([v + 0.5 * dt * k for v, k in zip(y0, k1)])
    k3 = rhs([v + 0.5 * dt * k for v, k in zip(y0, k2)])
    k4 = rhs([v + dt * k for v, k in zip(y0, k3)])
    return [
        v + dt / 6.0 * (a + 2.0 * b + 2.0 * c + d)
        for v, a, b, c, d in zip(y0, k1, k2, k3, k4)
    ]


def integrate_phase(
    rhs: RHS,
    y0: Sequence[float],
    dt: float,
    t_max: float,
    watch: Callable[[State], float],
    rising: bool,
    abort: Optional[Callable[[State], bool]] = None,
) -> PhaseSolution:
    """Integrate ``dy/dt = rhs(y)`` until ``watch(y)`` crosses zero.

    Args:
        rhs: Right-hand side, takes and returns a list of floats.
        y0: Initial state.
        dt: Fixed step [s].
        t_max: Hard stop; the phase is abandoned after this long [s].
        watch: Quantity whose zero crossing ends the phase. It must be
            strictly negative (``rising=True``) or strictly positive
            (``rising=False``) at ``y0``.
        rising: True when ``watch`` crosses zero from below.
        abort: Optional predicate. When it returns True the phase is
            abandoned *without* success -- used for physical guards such as
            fuel exhaustion or a climb that has stopped making altitude.

    Returns:
        A :class:`PhaseSolution`. ``hit`` is False when ``t_max`` was reached
        first, the state became non-finite, or ``abort`` fired; the partial
        history is still returned so that parameter sweeps degrade
        gracefully instead of raising.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    y = [float(v) for v in y0]
    t = 0.0
    sol = PhaseSolution(t=[0.0], y=[list(y)])
    n_max = int(math.ceil(t_max / dt))

    for _ in range(n_max):
        prev = list(y)
        nxt = rk4_step(rhs, y, dt)
        t_next = t + dt

        s0 = watch(prev)
        s1 = watch(nxt)
        crossed = s1 >= 0.0 if rising else s1 <= 0.0

        if crossed:
            denom = s1 - s0
            frac = 0.0 if abs(denom) < 1e-15 else -s0 / denom
            frac = min(max(frac, 0.0), 1.0)
            y_end = [p + frac * (n - p) for p, n in zip(prev, nxt)]
            sol.t.append(t + frac * dt)
            sol.y.append(y_end)
            sol.hit = True
            return sol

        y = nxt
        t = t_next
        sol.t.append(t)
        sol.y.append(list(y))

        if not all(math.isfinite(v) for v in y):
            sol.message = "state became non-finite at t = %.1f s" % t
            return sol
        if abort is not None and abort(y):
            sol.message = "phase abandoned by the guard at t = %.1f s" % t
            return sol

    sol.message = "time limit of %.0f s reached before the event" % t_max
    return sol


def thin(solution: PhaseSolution, max_points: int) -> PhaseSolution:
    """Downsample a phase to at most ``max_points`` samples.

    The first and last samples (the terminal event) are always preserved, so
    phase boundaries stay exact while the stored history stays small enough
    for interactive plotting.
    """
    n = len(solution.t)
    if n <= max_points or max_points < 2:
        return solution
    step = (n - 1) / (max_points - 1.0)
    idx = sorted({int(round(i * step)) for i in range(max_points)})
    return PhaseSolution(
        t=[solution.t[i] for i in idx],
        y=[solution.y[i] for i in idx],
        hit=solution.hit,
        message=solution.message,
    )
