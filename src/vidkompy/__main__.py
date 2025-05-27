#!/usr/bin/env python
# this_file: src/vidkompy/__main__.py

"""Enable running vidkompy as a module with python -m vidkompy."""

import fire
from src.vidkompy.comp.vidkompy import composite_videos
from src.vidkompy.comp.vidkompy import find_thumbnail


def cli():
    fire.Fire({"comp": composite_videos, "find": find_thumbnail})


if __name__ == "__main__":
    cli()
