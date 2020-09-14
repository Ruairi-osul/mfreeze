# mFreeze

mFreeze is a computer vision package for analysing mouse behavioural experiments. It is a rewrite of eztrack to add some features that I wanted (https://github.com/DeniseCaiLab/ezTrack).

## Install

```
$ pip install git+https://github.com/Ruairi-osul/mfreeze
```

## Detect Freezes

```
from mfreeze import FreezeDetector 

my_video = "video.mp4"

# change this to alter the sensitivity of the detector.
# lower values result in a detector which is more conservative
freeze_threshold = -0.5   

detector = FreezeDetector(my_video)
detector.detect_motion()
detector.detect_freezes()
detector.save_video()
report = detector.generate_report()
```

## Tuning Performance

I recommend inspecting the output video and altering the freeze threshold accordingly.





    
