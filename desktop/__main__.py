"""Entry point: ``python -m desktop`` starts the graphical application."""

from __future__ import annotations

import sys

from .app import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
