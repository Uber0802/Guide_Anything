"""Isaac Sim must be running before any isaaclab module that touches USD is imported."""

from __future__ import annotations

import argparse
import sys


def launch(parser: argparse.ArgumentParser):
    """Add the AppLauncher flags (--headless, --device, ...), start Isaac Sim, return (args, app)."""
    from isaaclab.app import AppLauncher

    # app.close() exits the process without flushing Python's stdout buffer.
    sys.stdout.reconfigure(line_buffering=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    return args, AppLauncher(args).app
