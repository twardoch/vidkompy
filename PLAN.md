# Refactoring Plan

1. Analyze the entire codebase very carefully. 

2. Check if every .py file from the codebase structure: 

```
src
├── __init__.py
├── __pycache__
└── vidkompy
    ├── __init__.py
    ├── __main__.py
    ├── __pycache__
    ├── __version__.py
    ├── align
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── algorithms.py
    │   ├── cli.py
    │   ├── core.py
    │   ├── data_types.py
    │   ├── display.py
    │   ├── frame_extractor.py
    │   └── precision.py
    ├── comp
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── align.py
    │   ├── data_types.py
    │   ├── dtw_aligner.py
    │   ├── fingerprint.py
    │   ├── multires.py
    │   ├── precision.py
    │   ├── temporal.py
    │   ├── tunnel.py
    │   ├── video.py
    │   └── vidkompy.py
    └── utils
        ├── __init__.py
        ├── enums.py
        ├── image.py
        ├── logging.py
        └── numba_ops.py

9 directories, 28 files
```

is actually used. 

3. Check the content of every .py file. Check if the imports are correct, if all code is reachable. 

4. Especially check if all code that claims to be optimized with numba is actually using it. Check if numba implementations for all functions that are expecting them are present in `utils/numba_ops.py`.

5. For every function and method, check if it can be flattened into multiple smaller constructs.  

6. For every single file, check if it’s not too complex. If it is, break it down into smaller files. 

7. As you work, use tools: 

- use `sequentialthinking` from the `sequential-thinking` tool to think about the best way to refactor the code
- consult with the `openai/o3` model via `chat_completion` from the `ask-chatgpt` tool to get help with the refactoring 
- use `search` and `fetch_content` from the `search_web_ddg` tool to get more info on the web

