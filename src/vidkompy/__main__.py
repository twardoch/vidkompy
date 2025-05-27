#!/usr/bin/env python
# this_file: src/vidkompy/__main__.py

"""Enable running vidkompy as a module with python -m vidkompy."""

import fire
from .align.cli import find_thumbnail

# Try to import comp module, but handle gracefully if it fails
try:
    from .comp.vidkompy import composite_videos

    COMP_AVAILABLE = True
except ImportError:
    COMP_AVAILABLE = False


def cli():
    """Main CLI entry point with subcommands."""
    commands = {
        "align": find_thumbnail,  # New name for thumbnail finding
        "find": find_thumbnail,  # Keep old name for backward compatibility
    }

    if COMP_AVAILABLE:
        commands["comp"] = composite_videos

    fire.Fire(commands)


if __name__ == "__main__":
    cli()
