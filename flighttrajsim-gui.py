#!/usr/bin/env python
"""Launch the Flight-Traj-Sim desktop application.

This small script exists so that PyInstaller has a single, obvious entry
point to bundle::

    pyinstaller --onefile --windowed flighttrajsim-gui.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from desktop.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
