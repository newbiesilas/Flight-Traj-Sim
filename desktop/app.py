"""Flight-Traj-Sim desktop application.

A native Qt front-end over the ``flighttrajsim`` solver. Every number shown
here is produced by the same pure-Python model that backs the command line
tool and the browser build: the GUI owns no physics of its own.

Run it with::

    python -m desktop.app           # from the repository root
    flighttrajsim-gui               # after pip install
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Sequence

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Make the package importable when launched from the repository root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from flighttrajsim import (  # noqa: E402
    AircraftParams,
    AircraftTrajSim,
    EngineParams,
    MissionProfile,
)
from flighttrajsim.core import CLIMB, CRUISE, DESCENT  # noqa: E402
from flighttrajsim.utils import export_result_csv, export_sweep_csv  # noqa: E402

from .chart import ChartWidget  # noqa: E402

APP_NAME = "Flight-Traj-Sim"
APP_VERSION = "1.0.0"

PHASE_COLORS = {CLIMB: "#2ca02c", CRUISE: "#1f77b4", DESCENT: "#d62728"}
SERIES_COLORS = {
    "altitude": "#1f77b4",
    "airspeed": "#ff7f0e",
    "mass": "#9467bd",
    "range": "#2ca02c",
    "profile": "#17a2b8",
    "sweep": "#1f6feb",
}


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------
class ParameterSlider(QWidget):
    """Labelled slider with a live value readout.

    ``scale`` converts the displayed value to the internal SI value, which
    keeps awkward magnitudes (TSFC in units of 1e-5) readable.
    """

    valueChanged = Signal()

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int = 0,
        suffix: str = "",
        scale: float = 1.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.decimals = decimals
        self.suffix = suffix
        self.scale = scale

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, int(round((maximum - minimum) / step)))
        self.slider.setValue(int(round((value - minimum) / step)))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(max(1, self.slider.maximum() // 20))
        self._minimum = minimum
        self._step = step

        self.readout = QLabel()
        self.readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.readout.setStyleSheet("color:#1f6feb; font-weight:600;")
        self.readout.setMinimumWidth(78)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(QLabel(label))
        top.addStretch(1)
        top.addWidget(self.readout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(2)
        layout.addLayout(top)
        layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self._on_changed)
        self._refresh()

    def _on_changed(self) -> None:
        self._refresh()
        self.valueChanged.emit()

    def _refresh(self) -> None:
        self.readout.setText("%.*f%s" % (self.decimals, self.value(), self.suffix))

    def value(self) -> float:
        """Displayed value."""
        return self._minimum + self.slider.value() * self._step

    def scaled_value(self) -> float:
        """Internal (SI) value."""
        return self.value() * self.scale

    def set_value(self, value: float) -> None:
        self.slider.setValue(int(round((value - self._minimum) / self._step)))


class MetricTile(QFrame):
    """One key figure in the summary strip."""

    def __init__(self, caption: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "MetricTile { background:#fbfcfd; border:1px solid #dfe3e8; border-radius:6px; }"
        )
        caption_label = QLabel(caption.upper())
        caption_label.setStyleSheet("color:#66707d; font-size:10px; letter-spacing:0.5px;")
        self.value_label = QLabel("--")
        self.value_label.setStyleSheet("font-size:17px; font-weight:600;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(1)
        layout.addWidget(caption_label)
        layout.addWidget(self.value_label)

    def set_value(self, text: str) -> None:
        self.value_label.setText(text)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    """Interactive mission simulator."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("%s %s" % (APP_NAME, APP_VERSION))
        self.resize(1280, 860)

        self.sim = AircraftTrajSim()
        self.result = None
        self.params: Dict[str, float] = {}
        self.sliders: Dict[str, ParameterSlider] = {}
        self.metrics: Dict[str, MetricTile] = {}
        self.sweep_rows: List[Dict[str, float]] = []

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self.run_simulation)

        self._build_ui()
        self.run_simulation()
        self.run_sweep()

    # -- construction -----------------------------------------------------
    def _build_ui(self) -> None:
        self._build_menu()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_results())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 940])
        self.setCentralWidget(splitter)

    def _build_menu(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        export_csv = QAction("Export trajectory as &CSV...", self)
        export_csv.setShortcut(QKeySequence.Save)
        export_csv.triggered.connect(self.export_csv)
        file_menu.addAction(export_csv)

        export_sweep = QAction("Export &sweep as CSV...", self)
        export_sweep.triggered.connect(self.export_sweep_csv)
        file_menu.addAction(export_sweep)

        file_menu.addSeparator()
        save_plot = QAction("Save &charts as images...", self)
        save_plot.triggered.connect(self.save_charts)
        file_menu.addAction(save_plot)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = menu.addMenu("&Help")
        about = QAction("&About", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        title = QLabel("Parameters")
        title.setStyleSheet("font-size:13px; font-weight:600; color:#66707d;")
        layout.addWidget(title)

        layout.addWidget(self._group("Aircraft", [
            ("m0", "Take-off mass", 6000, 20000, 12000, 100, 0, " kg"),
            ("fuel_mass", "Usable fuel", 500, 8000, 3000, 50, 0, " kg"),
            ("S", "Wing area", 12, 60, 24, 0.5, 1, " m2"),
        ]))
        layout.addWidget(self._group("Aerodynamics", [
            ("cd0", "Zero-lift drag CD0", 0.015, 0.060, 0.025, 0.001, 3, ""),
            ("k", "Induced drag factor k", 0.020, 0.090, 0.048, 0.001, 3, ""),
        ]))
        layout.addWidget(self._group("Engine", [
            ("thrust_sl", "Sea-level thrust", 12000, 80000, 39000, 500, 0, " N"),
            ("tsfc", "TSFC", 0.8, 3.0, 1.6, 0.05, 2, "e-5", 1e-5),
        ]))
        layout.addWidget(self._group("Mission", [
            ("h_cruise", "Cruise altitude", 3000, 15000, 11000, 100, 0, " m"),
            ("cruise_mach", "Cruise Mach", 0.45, 0.88, 0.78, 0.01, 2, ""),
            ("climb_cas", "Climb CAS", 90, 220, 150, 5, 0, " m/s"),
            ("descent_throttle", "Descent throttle", 0.0, 0.60, 0.15, 0.01, 2, ""),
        ]))

        buttons = QHBoxLayout()
        reset = QPushButton("Reset")
        reset.clicked.connect(self.reset_parameters)
        run_now = QPushButton("Run")
        run_now.setDefault(True)
        run_now.clicked.connect(self.run_simulation)
        buttons.addWidget(reset)
        buttons.addWidget(run_now)
        layout.addLayout(buttons)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color:#7a5b12;")
        self.status_label.hide()
        layout.addWidget(self.status_label)

        layout.addStretch(1)
        note = QLabel(
            "Sliders re-run the mission as you drag them. The solver is the same "
            "pure-Python model used by the CLI and the web build."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#66707d; font-size:11px;")
        layout.addWidget(note)
        return panel

    def _group(self, title: str, specs: Sequence[tuple]) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        for spec in specs:
            key, label, lo, hi, default, step, decimals, suffix = spec[:8]
            scale = spec[8] if len(spec) > 8 else 1.0
            slider = ParameterSlider(
                label, lo, hi, default, step, decimals, suffix, scale
            )
            slider.valueChanged.connect(self._schedule_run)
            self.sliders[key] = slider
            layout.addWidget(slider)
        return box

    def _build_results(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 12, 12, 12)
        layout.setSpacing(8)

        strip = QGridLayout()
        strip.setSpacing(7)
        for column, key in enumerate(
            ["Range", "Cruise part", "Endurance", "Fuel burned",
             "Cruise throttle", "Cruise TAS", "Peak CL", "Top altitude"]
        ):
            tile = MetricTile(key)
            self.metrics[key] = tile
            strip.addWidget(tile, 0, column)
        layout.addLayout(strip)

        tabs = QTabWidget()

        mission_tab = QWidget()
        mission_layout = QGridLayout(mission_tab)
        mission_layout.setSpacing(8)
        self.charts = {
            "altitude": ChartWidget(),
            "airspeed": ChartWidget(),
            "mass": ChartWidget(),
            "range": ChartWidget(),
            "profile": ChartWidget(),
        }
        mission_layout.addWidget(self.charts["altitude"], 0, 0)
        mission_layout.addWidget(self.charts["airspeed"], 0, 1)
        mission_layout.addWidget(self.charts["mass"], 1, 0)
        mission_layout.addWidget(self.charts["range"], 1, 1)
        mission_layout.addWidget(self.charts["profile"], 2, 0, 1, 2)
        mission_layout.setRowStretch(0, 1)
        mission_layout.setRowStretch(1, 1)
        mission_layout.setRowStretch(2, 1)
        tabs.addTab(mission_tab, "Trajectory")

        sweep_tab = QWidget()
        sweep_layout = QVBoxLayout(sweep_tab)
        sweep_layout.setSpacing(8)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Vary"))
        self.sweep_param = QComboBox()
        self.sweep_param.addItem("Take-off mass", "m0")
        self.sweep_param.addItem("Zero-lift drag CD0", "cd0")
        self.sweep_param.addItem("Induced drag factor k", "k")
        self.sweep_param.addItem("Sea-level thrust", "thrust_sl")
        self.sweep_param.addItem("TSFC", "tsfc")
        self.sweep_param.addItem("Cruise altitude", "h_cruise")
        self.sweep_param.addItem("Cruise Mach", "cruise_mach")
        controls.addWidget(self.sweep_param, 1)

        self.sweep_points = QDoubleSpinBox()
        self.sweep_points.setRange(3, 40)
        self.sweep_points.setValue(12)
        self.sweep_points.setDecimals(0)
        self.sweep_points.setSingleStep(1)
        self.sweep_points.setPrefix("points: ")
        controls.addWidget(self.sweep_points)

        run_sweep = QPushButton("Run sweep")
        run_sweep.clicked.connect(self.run_sweep)
        controls.addWidget(run_sweep)
        sweep_layout.addLayout(controls)

        self.sweep_chart = ChartWidget()
        sweep_layout.addWidget(self.sweep_chart, 2)

        self.sweep_table = QTableWidget()
        self.sweep_table.setAlternatingRowColors(True)
        self.sweep_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.sweep_table.verticalHeader().setVisible(False)
        sweep_layout.addWidget(self.sweep_table, 1)
        tabs.addTab(sweep_tab, "Sensitivity")

        layout.addWidget(tabs, 1)
        return container

    # -- parameters -------------------------------------------------------
    def _schedule_run(self) -> None:
        self._debounce.start()

    def read_parameters(self) -> None:
        self.params = {key: slider.scaled_value() for key, slider in self.sliders.items()}

    def reset_parameters(self) -> None:
        defaults = {
            "m0": AircraftParams().m0,
            "fuel_mass": AircraftParams().fuel_mass,
            "S": AircraftParams().S,
            "cd0": AircraftParams().cd0,
            "k": AircraftParams().k,
            "thrust_sl": EngineParams().thrust_sl,
            "tsfc": EngineParams().tsfc / 1e-5,
            "h_cruise": MissionProfile().h_cruise,
            "cruise_mach": MissionProfile().cruise_mach,
            "climb_cas": MissionProfile().climb_cas,
            "descent_throttle": MissionProfile().descent_throttle,
        }
        for key, value in defaults.items():
            self.sliders[key].set_value(value)
        self.run_simulation()
        self.run_sweep()

    # -- simulation -------------------------------------------------------
    def run_simulation(self) -> None:
        self.read_parameters()
        values = self.params
        try:
            aircraft = AircraftParams(
                m0=values["m0"], fuel_mass=values["fuel_mass"], S=values["S"],
                cd0=values["cd0"], k=values["k"],
            )
            engine = EngineParams(thrust_sl=values["thrust_sl"], tsfc=values["tsfc"])
            mission = MissionProfile(
                h_cruise=values["h_cruise"], cruise_mach=values["cruise_mach"],
                climb_cas=values["climb_cas"], descent_cas=values["climb_cas"],
                descent_throttle=values["descent_throttle"],
            )
            aircraft.validate()
            engine.validate()
            mission.validate()
            self.sim = AircraftTrajSim(aircraft, engine, mission)
            self.result = self.sim.run_simulation()
        except ValueError as exc:
            self.status_label.setText(str(exc))
            self.status_label.show()
            return

        self._render_summary()
        self._render_charts()

    def _render_summary(self) -> None:
        result = self.result
        if result is None:
            return
        summary = result.summary()
        self.metrics["Range"].set_value("%.0f km" % summary["range_km"])
        self.metrics["Cruise part"].set_value("%.0f km" % summary["cruise_range_km"])
        self.metrics["Endurance"].set_value("%.0f min" % summary["endurance_min"])
        self.metrics["Fuel burned"].set_value("%.0f kg" % summary["fuel_burned_kg"])
        throttle = summary["cruise_throttle_start"]
        self.metrics["Cruise throttle"].set_value(
            "%.2f" % throttle if math.isfinite(throttle) else "--"
        )
        self.metrics["Cruise TAS"].set_value("%.0f m/s" % summary["final_velocity_mps"])
        self.metrics["Peak CL"].set_value("%.2f" % summary["max_cl"])
        self.metrics["Top altitude"].set_value("%.1f km" % (summary["max_altitude_m"] / 1000))

        message = result.message
        if not message and result.stalled:
            message = ("The required lift coefficient exceeds CL_max: "
                       "the speed schedule stalls.")
        self.status_label.setText(message)
        self.status_label.setVisible(bool(message))

    def _render_charts(self) -> None:
        result = self.result
        if result is None or not result.t:
            for chart in self.charts.values():
                chart.clear()
            return

        minutes = [t / 60.0 for t in result.t]
        bands = []
        for name, i0, i1 in result.phases:
            if i1 > i0 and i1 <= len(minutes):
                bands.append((minutes[i0], minutes[i1 - 1], PHASE_COLORS.get(name, "#999999")))

        specs = [
            ("altitude", minutes, [h / 1000.0 for h in result.h], "Altitude",
             "Time [min]", "Altitude [km]"),
            ("airspeed", minutes, result.v, "True airspeed",
             "Time [min]", "TAS [m/s]"),
            ("mass", minutes, [m / 1000.0 for m in result.m], "Mass",
             "Time [min]", "Mass [t]"),
            ("range", minutes, [x / 1000.0 for x in result.x], "Ground distance",
             "Time [min]", "Range [km]"),
            ("profile", [x / 1000.0 for x in result.x], [h / 1000.0 for h in result.h],
             "Mission profile", "Range [km]", "Altitude [km]"),
        ]
        for key, xs, ys, title, xlabel, ylabel in specs:
            chart = self.charts[key]
            chart.clear()
            chart.title = title
            chart.xlabel = xlabel
            chart.ylabel = ylabel
            chart.show_markers = False
            if key != "profile":
                for x0, x1, color in bands:
                    chart.add_band(x0, x1, color)
            chart.add_series(xs, ys, SERIES_COLORS[key])

    # -- sweep ------------------------------------------------------------
    def run_sweep(self) -> None:
        key = self.sweep_param.currentData()
        if not key:
            return
        self.read_parameters()
        values = dict(self.params)

        bounds = {
            "m0": (7000.0, 20000.0),
            "cd0": (0.016, 0.055),
            "k": (0.024, 0.085),
            "thrust_sl": (14000.0, 70000.0),
            "tsfc": (1.0e-5, 2.8e-5),
            "h_cruise": (4000.0, 15000.0),
            "cruise_mach": (0.50, 0.86),
        }
        low, high = bounds.get(key, (None, None))
        base = values.get(key)
        if low is None or base is None:
            return
        count = max(3, int(self.sweep_points.value()))

        sweep_values = [low + (high - low) * i / (count - 1) for i in range(count)]
        rows = []
        for value in sweep_values:
            overrides = dict(values)
            overrides[key] = value
            # A coarse step keeps a full sweep interactive.
            mission = MissionProfile(
                h_cruise=overrides["h_cruise"], cruise_mach=overrides["cruise_mach"],
                climb_cas=overrides["climb_cas"], descent_cas=overrides["climb_cas"],
                descent_throttle=overrides["descent_throttle"],
                dt=10.0, n_points=60,
            )
            sim = AircraftTrajSim(
                AircraftParams(
                    m0=overrides["m0"], fuel_mass=overrides["fuel_mass"],
                    S=overrides["S"], cd0=overrides["cd0"], k=overrides["k"],
                ),
                EngineParams(thrust_sl=overrides["thrust_sl"], tsfc=overrides["tsfc"]),
                mission,
            )
            outcome = sim.run_simulation()
            rows.append({
                "x": value,
                "range_km": outcome.range_km,
                "endurance_min": outcome.endurance_min,
                "fuel_kg": outcome.fuel_burned_kg,
                "completed": outcome.completed,
            })
        self.sweep_rows = rows
        self._render_sweep(key)

    def _render_sweep(self, key: str) -> None:
        rows = self.sweep_rows
        axis_scale = 1e-3 if key in ("m0", "thrust_sl", "h_cruise") else 1.0
        if key == "tsfc":
            axis_scale = 1e5
        labels = {
            "m0": "Take-off mass [t]", "cd0": "CD0", "k": "k",
            "thrust_sl": "Sea-level thrust [kN]", "tsfc": "TSFC [1e-5]",
            "h_cruise": "Cruise altitude [km]", "cruise_mach": "Cruise Mach",
        }
        self.sweep_chart.clear()
        self.sweep_chart.title = "Range vs %s" % labels.get(key, key)
        self.sweep_chart.xlabel = labels.get(key, key)
        self.sweep_chart.ylabel = "Range [km]"
        self.sweep_chart.show_markers = True
        self.sweep_chart.add_series(
            [row["x"] * axis_scale for row in rows],
            [row["range_km"] for row in rows],
            SERIES_COLORS["sweep"],
        )

        headers = [labels.get(key, key), "Range [km]", "Endurance [min]", "Fuel [kg]", "OK"]
        self.sweep_table.setColumnCount(len(headers))
        self.sweep_table.setHorizontalHeaderLabels(headers)
        self.sweep_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            cells = [
                "%.4g" % (row["x"] * axis_scale),
                "%.0f" % row["range_km"],
                "%.0f" % row["endurance_min"],
                "%.0f" % row["fuel_kg"],
                "yes" if row["completed"] else "no",
            ]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.sweep_table.setItem(i, j, item)
        self.sweep_table.resizeColumnsToContents()

    # -- actions ----------------------------------------------------------
    def export_csv(self) -> None:
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export trajectory", "trajectory.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        export_result_csv(self.result, path)
        self.statusBar().showMessage("Wrote %s" % path, 5000)

    def export_sweep_csv(self) -> None:
        if not self.sweep_rows:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export sweep", "sweep.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        export_sweep_csv(self.sweep_rows, path)
        self.statusBar().showMessage("Wrote %s" % path, 5000)

    def save_charts(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Save charts to folder")
        if not directory:
            return
        for name, chart in self.charts.items():
            chart.grab().save(os.path.join(directory, "%s.png" % name))
        self.sweep_chart.grab().save(os.path.join(directory, "sweep.png"))
        self.statusBar().showMessage("Saved charts to %s" % directory, 5000)

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "About %s" % APP_NAME,
            "<b>%s %s</b><br><br>"
            "Point-mass aircraft trajectory simulator: an ISA atmosphere, a "
            "parabolic drag polar with compressibility drag and a turbofan "
            "thrust / fuel-flow model integrated through climb, cruise and "
            "descent.<br><br>"
            "The solver is dependency-free standard-library Python; this GUI "
            "only draws it.<br><br>"
            "Licensed under the MIT licence." % (APP_NAME, APP_VERSION),
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    app = QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setFont(QFont("Segoe UI", 9))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
