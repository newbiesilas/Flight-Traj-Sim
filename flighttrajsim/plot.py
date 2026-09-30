"""Optional matplotlib plotting helpers.

``matplotlib`` is an **optional** dependency: the core simulator, the test
suite and the packaged desktop application run without it. Figure
construction is separated from display so that the desktop GUI can grab a
bare ``Figure`` object, while :func:`plot_trajectory` handles the show/save
decision for scripts.
"""

from __future__ import annotations

from typing import Optional, Sequence

MISSING_MATPLOTLIB = (
    "matplotlib is required for plotting; install it with "
    "'pip install flighttrajsim[plot]' or 'pip install matplotlib'."
)


def _plt(backend: Optional[str] = None):
    """Import matplotlib.pyplot lazily with an actionable error message."""
    try:
        import matplotlib
        if backend:
            matplotlib.use(backend)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(MISSING_MATPLOTLIB) from exc
    return matplotlib, plt


def _phase_colors() -> dict:
    return {"climb": "#2ca02c", "cruise": "#1f77b4", "descent": "#d62728"}


def _as_list(values) -> list:
    return list(values)


def _scaled(values, factor: float) -> list:
    return [v * factor for v in values]


def build_trajectory_figure(result):
    """Create the standard 2x2 mission figure (altitude, speed, mass, range)."""
    matplotlib, plt = _plt()

    t = _as_list(result.t)
    h = _as_list(result.altitude)
    v = _as_list(result.velocity)
    x = _as_list(result.range)
    m = _as_list(result.mass)

    time_min = [ti / 60.0 for ti in t]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    colors = _phase_colors()
    phases = list(getattr(result, "phases", []))

    def mark_phases(ax):
        for name, i0, i1 in phases:
            if i1 > len(t) or i1 <= i0:
                continue
            ax.axvspan(
                time_min[i0], time_min[i1 - 1],
                color=colors.get(name, "#999999"), alpha=0.07,
            )
        for name, i0, i1 in phases:
            if 0 < i0 < len(t):
                ax.axvline(time_min[i0], color="#888888", linestyle="--", linewidth=0.8)
                ax.text(
                    time_min[i0], ax.get_ylim()[1], " " + name, fontsize=8,
                    color="#555555", va="top",
                )

    ax = axes[0, 0]
    ax.plot(time_min, _scaled(h, 1e-3), color="#1f77b4", linewidth=1.6)
    mark_phases(ax)
    ax.set_ylabel("Altitude [km]")
    ax.grid(alpha=0.3)
    ax.set_title("Altitude")

    ax = axes[0, 1]
    ax.plot(time_min, v, color="#ff7f0e", linewidth=1.6, label="TAS")
    ax.set_ylabel("True airspeed [m/s]")
    ax.grid(alpha=0.3)
    ax.set_title("Airspeed")
    try:
        mach = _as_list(result.mach)
        twin = ax.twinx()
        twin.plot(time_min, mach, color="#7f7f7f", linewidth=1.0, linestyle=":", label="Mach")
        twin.set_ylabel("Mach")
    except Exception:  # pragma: no cover - cosmetic twin axis
        pass

    ax = axes[1, 0]
    ax.plot(time_min, _scaled(m, 1e-3), color="#9467bd", linewidth=1.6)
    mark_phases(ax)
    ax.set_ylabel("Mass [t]")
    ax.set_xlabel("Time [min]")
    ax.grid(alpha=0.3)
    ax.set_title("Mass")

    ax = axes[1, 1]
    ax.plot(time_min, _scaled(x, 1e-3), color="#2ca02c", linewidth=1.6)
    mark_phases(ax)
    ax.set_ylabel("Ground distance [km]")
    ax.set_xlabel("Time [min]")
    ax.grid(alpha=0.3)
    ax.set_title("Range")

    fig.tight_layout()
    return fig


def build_mission_profile_figure(result):
    """Altitude-versus-range profile (the classic mission diagram)."""
    matplotlib, plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        _scaled(_as_list(result.range), 1e-3),
        _scaled(_as_list(result.altitude), 1e-3),
        color="#17a2b8", linewidth=1.8,
    )
    ax.set_xlabel("Range [km]")
    ax.set_ylabel("Altitude [km]")
    ax.grid(alpha=0.3)
    ax.set_title("Mission profile")
    fig.tight_layout()
    return fig


def build_sweep_figure(
    rows: Sequence[dict], x_key: str, y_key: str = "range_km", y_label: Optional[str] = None
):
    """Create a line plot of a swept parameter against a result metric."""
    matplotlib, plt = _plt()

    xs = [r[x_key] for r in rows]
    ys = [r.get(y_key) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", color="#1f77b4", linewidth=1.6, markersize=4)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_label or y_key)
    ax.grid(alpha=0.3)
    ax.set_title("%s vs %s" % (y_key, x_key))
    fig.tight_layout()
    return fig


def plot_trajectory(result, show: bool = True, save_path: Optional[str] = None) -> None:
    """Plot a mission trajectory; display it and/or write it to a file.

    Args:
        result: :class:`TrajectoryResult`.
        show: call ``plt.show()``.
        save_path: optional output image path (png/pdf/svg).
    """
    matplotlib, plt = _plt("Agg" if (not show and save_path) else None)

    fig = build_trajectory_figure(result)
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_sweep(
    rows: Sequence[dict],
    x_key: str,
    y_key: str = "range_km",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Plot parameter-sweep results."""
    matplotlib, plt = _plt("Agg" if (not show and save_path) else None)

    fig = build_sweep_figure(rows, x_key, y_key)
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
