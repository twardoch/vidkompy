#!/usr/bin/env python
# this_file: src/vidkompy/__main__.py

"""Enable running vidkompy as a module with python -m vidkompy."""

import fire
from vidkompy.vidkompy import composite_videos
from vidkompy.thumbfind import find_thumbnail


def cli():
    fire.Fire({"comp": composite_videos, "find": find_thumbnail})


if __name__ == "__main__":
    cli()
