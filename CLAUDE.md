
## Coding style

<guidelines>
# When you write code

- Iterate gradually, avoiding major changes
- Minimize confirmations and checks
- Preserve existing code/structure unless necessary
- Use constants over magic numbers
- Check for existing solutions in the codebase before starting
- Check often the coherence of the code you’re writing with the rest of the code.
- Focus on minimal viable increments and ship early
- Write explanatory docstrings/comments that explain what and WHY this does, explain where and how the code is used/referred to elsewhere in the code
- Analyze code line-by-line
- Handle failures gracefully with retries, fallbacks, user guidance
- Address edge cases, validate assumptions, catch errors early
- Let the computer do the work, minimize user decisions
- Reduce cognitive load, beautify code
- Modularize repeated logic into concise, single-purpose functions
- Favor flat over nested structures
- Consistently keep, document, update and consult the holistic overview mental image of the codebase. 

Work in rounds: 

- Create `PROGRESS.md` as a detailed flat plan with `[ ]` items. 
- Identify the most important TODO items, and create `TODO.md` with `[ ]` items. 
- Implement the changes. 
- Update `PROGRESS.md` and `TODO.md` as you go. 
- After each round of changes, update `CHANGELOG.md` with the changes.
- Update `README.md` to reflect the changes.

## Keep track of paths

In each source file, maintain the up-to-date `this_file` record that shows the path of the current file relative to project root. Place the `this_file` record near the top of the file, as a comment after the shebangs, or in the YAML Markdown frontmatter.

## When you write Python

- Use `uv pip`, never `pip`
- Use `python -m` when running code
- PEP 8: Use consistent formatting and naming
- Write clear, descriptive names for functions and variables
- PEP 20: Keep code simple and explicit. Prioritize readability over cleverness
- Use type hints in their simplest form (list, dict, | for unions)
- PEP 257: Write clear, imperative docstrings
- Use f-strings. Use structural pattern matching where appropriate
- ALWAYS add "verbose" mode logugu-based logging, & debug-log
- For CLI Python scripts, use fire & rich, and start the script with

```
#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["PKG1", "PKG2"]
# ///
# this_file: PATH_TO_CURRENT_FILE
```

Ask before extending/refactoring existing code in a way that may add complexity or break things.

When you’re finished, print "Wait, but" to go back, think & reflect, revise & improvement what you’ve done (but don’t invent functionality freely). Repeat this. But stick to the goal of "minimal viable next version". Lead two experts: "Ideot" for creative, unorthodox ideas, and "Critin" to critique flawed thinking and moderate for balanced discussions. The three of you shall illuminate knowledge with concise, beautiful responses, process methodically for clear answers, collaborate step-by-step, sharing thoughts and adapting. If errors are found, step back and focus on accuracy and progress.

## After Python changes run:

```
fd -e py -x autoflake {}; fd -e py -x pyupgrade --py312-plus {}; fd -e py -x ruff check --output-format=github --fix --unsafe-fixes {}; fd -e py -x ruff format --respect-gitignore --target-version py312 {}; python -m pytest;
```

Be creative, diligent, critical, relentless & funny!
</guidelines>


## Project Overview

vidkompy is an intelligent video overlay tool that overlays foreground videos onto background videos with automatic alignment. The project prioritizes 
precision and intelligence over raw speed.

`vidkompy` is a Python CLI (Fire) tool that lets me overlay two videos: --bg (background) and --fg (foreground), and saves the result to --output (name is built automatically if not provided).

The tool needs to be robust and smart.

1. The bg video can be larger than the fg video. In that case, the tool needs to automatically find the best offset to overlay the fg video on the bg video.

2. The bg and fg videos can have different FPSes. In that case, the tool needs to use the higher FPS for the output video.

3. The bg and fg videos may have different durations (the difference would be typically rather slight). Ths tool needs to find the optimal time offset to overlay the fg video on the bg video. We need to find "the perfect pair of frames that when overlaid will form the START frame of the output video", and "the perfect pair of frames that when overlaid will form the END frame of the output video". The output video will be trimmed to go from that start to that end. Some starting and/or ending frames from either the fg or the bg video may be dropped. 

4. The bg and fg videos may have sound (it would be identical, but may be offset). If both have sound, the sound may be used to find the right time offset for the overlay (if I specify --match_time audio). The fg sound would be is for the output video. If only one has sound, that sound is used for the output video.

5. The content of the two videos would typically be very similar but not identical, because one video is typically a recompressed, edited variant of the other.

6. Overall, the fg video is thought to be "better quality", and temporally properly aligned. The bg video can be streched, modified, re-timed, frames added or dropped etc. 

7. All these conditions may be combined. That means, the tool needs to be able to find both the spatial and the temporal offsets. Spatially, the fg video is never larger than the bg video. Temporally, either video can be longer.



## Key Commands

### Development & Testing
```bash
# Run tests
hatch run test

# Run tests with coverage
hatch run test-cov

# Type checking
hatch run type-check

# Linting and formatting
hatch run lint      # Check code
hatch run fix       # Auto-fix issues
hatch run fmt       # Format code

# All lint checks
hatch run lint:all
```

### Running the Tool
```bash
python src/vidkompy/vidkompy_old.py --help|cat

INFO: Showing help with the command 'vidkompy_old.py -- --help'.

NAME
    vidkompy_old.py - Overlay foreground video onto background video with intelligent alignment.

SYNOPSIS
    vidkompy_old.py BG FG <flags>

DESCRIPTION
    Overlay foreground video onto background video with intelligent alignment.

POSITIONAL ARGUMENTS
    BG
        Type: str
        Background video path
    FG
        Type: str
        Foreground video path

FLAGS
    -o, --output=OUTPUT
        Type: Optional[str | None]
        Default: None
        Output video path (auto-generated if not provided)
    --match_space=MATCH_SPACE
        Type: str
        Default: 'precise'
        Spatial alignment method ('precise' or 'fast')
    --temporal_align=TEMPORAL_ALIGN
        Type: str
        Default: 'frames'
        Temporal alignment method ('audio', 'duration', or 'frames') [default: 'frames']
    --trim=TRIM
        Type: bool
        Default: True
        Trim output to overlapping segments only [default: True]
    -s, --skip_spatial_align=SKIP_SPATIAL_ALIGN
        Type: bool
        Default: False
        Skip spatial alignment (use center)
    -v, --verbose=VERBOSE
        Type: bool
        Default: False
        Enable verbose logging
    --max_keyframes=MAX_KEYFRAMES
        Type: int
        Default: 2000
        Maximum number of keyframes to use in frames mode [default: 2000]

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS


# Direct execution (main implementation in vidkompy_old.py)
python -m vidkompy --bg tests/bg.mp4 --fg tests/fg.mp4 -o tests/output_new.mp4 --match_time precise --verbose
```

## Architecture Notes

1. **Main Implementation**: The core functionality resides in `src/vidkompy/vidkompy_old.py` (60KB). 

The `vidkompy.py` file is a placeholder for future refactoring.

2. **Temporal Alignment Modes**:
   - `audio`: Cross-correlation of audio tracks
   - `duration`: Centers foreground within background duration
   - `frames`: Advanced frame-by-frame content matching with dynamic mapping

3. **Spatial Alignment Methods**:
   - `template` (precise, default): OpenCV template matching for exact sub-regions
   - `feature` (fast): ORB feature matching for more robust alignment

4. **Known Issue**: The `frames` temporal alignment method has drift-and-catchup issues where the background portion runs at varying speeds 

5. **Dependencies**: The tool requires FFmpeg binary installed separately. Python dependencies are listed in the script header and include opencv-python, ffmpeg-python, scipy, numpy, soundfile, scikit-image, fire, rich, and loguru.

## Testing Approach

The project uses pytest with minimal test coverage currently (only version check). When adding new functionality:
- Place tests in `tests/` directory
- Use pytest markers for categorization (unit, integration, benchmark)
- Run with `hatch run test` or `hatch run test-cov` for coverage
