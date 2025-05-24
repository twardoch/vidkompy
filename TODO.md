# TODO: 

Read vidoverlay-README.md and vidoverlay.md to understand the codebase. 

So... the `frames` method is not working well. `./vidoverlay.py --bg vlveo-buenos-diego3_cloud_apo8_prob4.mp4 --fg vlveo-buenos-diego2.mp4 --spatial_method template --temporal_align frames -o test2-template-duration.mp4` where the bg video is 1920x1080 and fg is 1920 x870. With the tool, we overlay the videos at 0,0, so in the result the bottom 1920x210 is from the bg video. 

The `frames` mode gets huge temporal drift-and-catchup. The bottom part runs extremely fast until some particular frame, and then that frame portion gets frozen, and then the bottom part gets fast again. 

Now: I want that the frames mode to smartly align as many keyframes as possible, and the rest needs to be interpolated. Maybe there needs to be some kind of --max_keyframes parameter, defaulting to, say, 2000. I’m mostly working with videos that have < 1000 total frames anyway!  