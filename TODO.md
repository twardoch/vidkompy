# TODO

This project has two components: 

- `vidkompy` - the older project, which we’re ignoring for now
- `thumbfind` - the newer project, which we’ll focus on. 

`varia/thumbfind1.md` describes how `src/vidkompy/thumbfind.py` works. `varia/thumbfind2.md` contains several extensive reports, which served as basis on which `src/vidkompy/thumbfind.py` was developed. 

But `src/vidkompy/thumbfind.py` is unreliable. Currently for an example video it suggested that the bg video has the thumbnail as 94% scaled down variant of the fg video — but in fact, the bg video was made without scaling, with just shifting of the fg. So `src/vidkompy/thumbfind.py` should have identified the thumbnail as 100% scaled down variant of the fg video. 

Read `varia/thumbfind1.md` and `varia/thumbfind2.md`, and perform deep intensive thinking on how to improve `src/vidkompy/thumbfind.py`. One thing is that the algorithm should be more robust. The other thing is that the algorithm should prefer scale 100% and only if the confidence is quite high that we actually have a scaled-down version, only then it should suggest the scaled down version. Or perhaps it should report two results: confidence and shift for 100% scale approach, as well as confidence, shift and scale for the scaled down approach. 

