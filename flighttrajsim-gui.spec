# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the Flight-Traj-Sim desktop application.

Build with::

    pyinstaller flighttrajsim-gui.spec          # one-folder build
    pyinstaller --onefile flighttrajsim-gui.spec  # single executable

The solver is pure standard-library Python, so only PySide6 needs collecting.
matplotlib is *not* bundled: the GUI draws with QPainter.
"""

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# The GUI imports the solver package through `desktop.app`; make sure every
# submodule ends up in the bundle.
hiddenimports = collect_submodules("flighttrajsim") + collect_submodules("desktop")

excluded = [
    "matplotlib",
    "numpy",
    "scipy",
    "PIL.ImageQt",
    "tkinter",
    "pytest",
]

a = Analysis(
    ["flighttrajsim-gui.py"],
    pathex=["."],
    binaries=[],
    datas=[("README.md", "."), ("LICENSE", ".")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FlightTrajSim",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
    version=None,
)
