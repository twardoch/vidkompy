#!/usr/bin/env python
# this_file: src/vidkompy/__main__.py

"""Enable running vidkompy as a module with python -m vidkompy."""

import fire
from vidkompy.vidkompy import main

if __name__ == "__main__":
    fire.Fire(main)
