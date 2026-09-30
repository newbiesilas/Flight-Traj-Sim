"""I/O helpers: CSV export and human-readable summaries."""

from __future__ import annotations

import csv
import os
from typing import Dict, Optional, Sequence


def export_result_csv(result, filename: str = "sim_result.csv") -> str:
    """Write the full trajectory to CSV.

    Columns: ``time_s, altitude_m, tas_mps, mach, range_m, mass_kg, phase``.
    Returns the absolute path written.
    """
    directory = os.path.dirname(os.path.abspath(filename))
    if directory:
        os.makedirs(directory, exist_ok=True)

    rows = result.rows() if hasattr(result, "rows") else []
    with open(filename, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["time_s", "altitude_m", "tas_mps", "mach", "range_m", "mass_kg", "phase"]
        )
        for row in rows:
            writer.writerow(
                [
                    "%.3f" % row["time_s"],
                    "%.2f" % row["altitude_m"],
                    "%.4f" % row["tas_mps"],
                    "%.5f" % row["mach"],
                    "%.2f" % row["range_m"],
                    "%.3f" % row["mass_kg"],
                    row["phase"],
                ]
            )
    return os.path.abspath(filename)


def export_sweep_csv(rows: Sequence[Dict[str, float]], filename: str = "sweep_result.csv") -> str:
    """Write parameter-sweep rows to CSV (union of all keys as header)."""
    directory = os.path.dirname(os.path.abspath(filename))
    if directory:
        os.makedirs(directory, exist_ok=True)

    header = []
    for row in rows:
        for key in row:
            if key not in header:
                header.append(key)

    with open(filename, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return os.path.abspath(filename)


def format_summary(result, title: str = "Mission summary") -> str:
    """Render a compact multi-line mission summary."""
    s = result.summary() if hasattr(result, "summary") else {}
    lines = [title, "-" * len(title)]
    lines.append("Range            : %8.1f km" % (s.get("range_km", float("nan"))))
    lines.append("  of which cruise: %8.1f km" % (s.get("cruise_range_km", float("nan"))))
    lines.append("Endurance        : %8.1f min" % (s.get("endurance_min", float("nan"))))
    lines.append("Fuel burned      : %8.1f kg" % (s.get("fuel_burned_kg", float("nan"))))
    lines.append("Final altitude   : %8.1f m" % (s.get("final_altitude_m", float("nan"))))
    lines.append("Final TAS        : %8.1f m/s" % (s.get("final_velocity_mps", float("nan"))))
    lines.append("Final mass       : %8.1f kg" % (s.get("final_mass_kg", float("nan"))))
    lines.append("Cruise throttle  : %8.3f (start of cruise)" % (s.get("cruise_throttle_start", float("nan"))))
    lines.append("Peak CL          : %8.3f" % (s.get("max_cl", float("nan"))))
    lines.append("Completed        : %s" % s.get("completed", True))
    if getattr(result, "message", ""):
        lines.append("Note             : %s" % result.message)
    return "\n".join(lines)


def format_sweep_table(rows: Sequence[Dict[str, float]], x_key: str) -> str:
    """Render a fixed-width table of sweep results."""
    header = "%-12s %12s %12s %12s" % (x_key, "range_km", "endurance_min", "fuel_kg")
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            "%-12.4g %12.2f %12.2f %12.1f"
            % (
                row.get(x_key, float("nan")),
                row.get("range_km", float("nan")),
                row.get("endurance_s", float("nan")) / 60.0,
                row.get("fuel_burned_kg", float("nan")),
            )
        )
    return "\n".join(lines)
