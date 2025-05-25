# TODO

The recent changes (check the git) worsened the drift. Around 5 seconds in the test video, suddenly the bg video speeds up and the fg video goes at normal speed, the hand movement is like 1 second delayed in the fg vs. bg. 

One important question: 

When we're extracting frames, the general assumption is that fg is smaller than bg. So shouldn't we spatially align the fg to the bg, and then, for the purpose of temoral alignment analysis, we should crop the bg frames to the size of the fg? 

TASK: Fix the error. Look at the git changes and what has changed. And think about what I just said about the cropping. 

Carefully, very carefully analyze the precise alignment algorithm. Then make a plan in SPEC.md in which you describe the precise your steps to fix the problem. 

EXECUTE THE TASK (= write SPEC.md) now. 