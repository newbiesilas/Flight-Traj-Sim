"""Minimal dependency-free chart widget for the desktop application.

The desktop GUI deliberately avoids matplotlib: a QPainter-based widget keeps
the packaged executable small, removes a heavy dependency from the installer
and makes "export image" a one-liner (``widget.grab().save(path)``).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

MARGIN_LEFT = 66
MARGIN_RIGHT = 16
MARGIN_TOP = 26
MARGIN_BOTTOM = 36

GRID_COLOR = QColor("#eceef1")
AXIS_COLOR = QColor("#c9cfd6")
TICK_COLOR = QColor("#8a939e")
TITLE_COLOR = QColor("#1c2430")
LABEL_COLOR = QColor("#66707d")


def _fmt_tick(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 10000:
        return "%dk" % round(value / 1000)
    if magnitude >= 100:
        return "%d" % round(value)
    if magnitude >= 1:
        return "%.1f" % value
    return "%.2f" % value


class ChartWidget(QWidget):
    """A small line chart with optional shaded bands and markers."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.title = ""
        self.xlabel = ""
        self.ylabel = ""
        self.show_markers = False
        self.empty_text = "no data"
        self._series: List[Tuple[Sequence[float], Sequence[float], QColor]] = []
        self._bands: List[Tuple[float, float, QColor]] = []
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setBackgroundRole(self.backgroundRole())

    # -- data ------------------------------------------------------------
    def clear(self) -> None:
        self._series = []
        self._bands = []
        self.update()

    def add_series(self, xs: Sequence[float], ys: Sequence[float], color: str,
                   markers: bool = False) -> None:
        self._series.append((list(xs), list(ys), QColor(color)))
        if markers:
            self.show_markers = True
        self.update()

    def add_band(self, x0: float, x1: float, color: str) -> None:
        self._bands.append((x0, x1, QColor(color)))
        self.update()

    # -- painting ---------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        width, height = self.width(), self.height()
        painter.fillRect(0, 0, width, height, QColor("#ffffff"))

        if not self._series:
            painter.setPen(TICK_COLOR)
            painter.drawText(self.rect(), Qt.AlignCenter, self.empty_text)
            return

        xmin, xmax, ymin, ymax = self._bounds()
        if xmax == xmin:
            xmax = xmin + 1.0
        if ymax == ymin:
            ymax = ymin + 1.0
        pad = (ymax - ymin) * 0.08
        ymin -= pad
        ymax += pad

        plot_w = width - MARGIN_LEFT - MARGIN_RIGHT
        plot_h = height - MARGIN_TOP - MARGIN_BOTTOM
        if plot_w <= 10 or plot_h <= 10:
            return

        def px(v: float) -> float:
            return MARGIN_LEFT + (v - xmin) / (xmax - xmin) * plot_w

        def py(v: float) -> float:
            return MARGIN_TOP + plot_h - (v - ymin) / (ymax - ymin) * plot_h

        tick_font = QFont("Segoe UI", 8)
        painter.setFont(tick_font)
        painter.setPen(QPen(GRID_COLOR, 1))
        for i in range(5):
            y = MARGIN_TOP + plot_h * i / 4.0
            painter.drawLine(int(MARGIN_LEFT), int(y), int(MARGIN_LEFT + plot_w), int(y))
            painter.setPen(TICK_COLOR)
            painter.drawText(
                QRectF(0, y - 8, MARGIN_LEFT - 8, 16),
                Qt.AlignRight | Qt.AlignVCenter,
                _fmt_tick(ymax - (ymax - ymin) * i / 4.0),
            )
            painter.setPen(QPen(GRID_COLOR, 1))
        for i in range(6):
            x = MARGIN_LEFT + plot_w * i / 5.0
            painter.drawLine(int(x), int(MARGIN_TOP), int(x), int(MARGIN_TOP + plot_h))
            painter.setPen(TICK_COLOR)
            painter.drawText(
                QRectF(x - 40, MARGIN_TOP + plot_h + 4, 80, 16),
                Qt.AlignCenter | Qt.AlignTop,
                _fmt_tick(xmin + (xmax - xmin) * i / 5.0),
            )
            painter.setPen(QPen(GRID_COLOR, 1))

        for x0, x1, color in self._bands:
            shaded = QColor(color)
            shaded.setAlpha(28)
            left = min(px(x0), px(x1))
            painter.fillRect(
                QRectF(left, MARGIN_TOP, max(abs(px(x1) - px(x0)), 0.6), plot_h), shaded
            )

        painter.setPen(QPen(AXIS_COLOR, 1))
        painter.drawRect(MARGIN_LEFT, MARGIN_TOP, plot_w, plot_h)

        for xs, ys, color in self._series:
            polygon = QPolygonF()
            for x, y in zip(xs, ys):
                polygon.append(QPointF(px(x), py(y)))
            painter.setPen(QPen(color, 1.8))
            painter.drawPolyline(polygon)
            if self.show_markers:
                painter.setBrush(color)
                painter.setPen(QPen(color, 1))
                for x, y in zip(xs, ys):
                    painter.drawEllipse(QPointF(px(x), py(y)), 2.6, 2.6)

        painter.setPen(TITLE_COLOR)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        painter.drawText(QRectF(MARGIN_LEFT, 4, plot_w, 18), Qt.AlignLeft | Qt.AlignVCenter,
                         self.title)
        painter.setPen(LABEL_COLOR)
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(
            QRectF(MARGIN_LEFT, height - 18, plot_w, 16), Qt.AlignRight | Qt.AlignVCenter,
            self.xlabel,
        )
        painter.save()
        painter.translate(14, MARGIN_TOP + plot_h / 2.0)
        painter.rotate(-90)
        painter.drawText(QRectF(-plot_h / 2.0, -10, plot_h, 20), Qt.AlignCenter, self.ylabel)
        painter.restore()

    # -- helpers ----------------------------------------------------------
    def _bounds(self) -> Tuple[float, float, float, float]:
        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")
        for xs, ys, _c in self._series:
            for x, y in zip(xs, ys):
                if x != x or y != y:  # NaN guard
                    continue
                xmin = min(xmin, x)
                xmax = max(xmax, x)
                ymin = min(ymin, y)
                ymax = max(ymax, y)
        if xmin == float("inf"):
            return 0.0, 1.0, 0.0, 1.0
        return xmin, xmax, ymin, ymax
