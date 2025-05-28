#!/usr/bin/env python
# this_file: src/vidkompy/__main__.py

"""Enable running vidkompy as a module with python -m vidkompy."""

import fire


def _lazy_find_thumbnail(*args, **kwargs):
    """Lazy wrapper for thumbnail finding to avoid heavy imports at startup."""
    from .align.cli import find_thumbnail

    return find_thumbnail(*args, **kwargs)


def _lazy_composite_videos(*args, **kwargs):
    """Lazy wrapper for video composition to avoid heavy imports at startup."""
    from .comp.vidkompy import composite_videos

    return composite_videos(*args, **kwargs)


def cli():
    """Main CLI entry point with subcommands."""

    fire.Fire(
        {
            "align": _lazy_find_thumbnail,
            "compose": _lazy_composite_videos,  # Renamed from "comp" to avoid clash
            # Backward compatibility alias
            "comp": _lazy_composite_videos,
        }
    )


if __name__ == "__main__":
    cli()
