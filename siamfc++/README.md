## Get ready for Siamfc++ cell detection

### Setup

#### 1. Get programs from [Siamfc++ Video Analyst](https://github.com/MegviiDetection/video_analyst)

Download codes from [here](https://github.com/MegviiDetection/video_analyst/archive/refs/heads/master.zip).

Or run this code
```bash
git clone https://github.com/MegviiDetection/video_analyst.git
```

#### 2. Download trained models
There are two models, alexnet and googlenet, but recommended to use googlenet which is better than alexnet.
Download the three files written in this yaml file (video_analyst/experiments/siamfcpp/test/vot/siamfcpp_googlenet.yaml).

1. [https://drive.google.com/file/d/1zLj0nsI0LXrQ2yANbLQ1VpsGtTMrEkjB/view?usp=sharing](https://drive.google.com/file/d/1zLj0nsI0LXrQ2yANbLQ1VpsGtTMrEkjB/view?usp=sharing)
2. [https://drive.google.com/file/d/1ajKfOTbUG_l8NQWMaFZSPoqIIseoreI7/view?usp=sharing](https://drive.google.com/file/d/1ajKfOTbUG_l8NQWMaFZSPoqIIseoreI7/view?usp=sharing)
3. [https://drive.google.com/file/d/1w-zH-G_Z0X9JLpCTGNqGyttm4FAFWhek/view?usp=sharing](https://drive.google.com/file/d/1w-zH-G_Z0X9JLpCTGNqGyttm4FAFWhek/view?usp=sharing)

Put these files on **video_analyst/models/siamfcpp//**

#### 3. Modify some codes.

Changed the code to make it easier to use. The edited codes are publish here, so you can just replace it.

1. video_analyst/videoanalyst/pipeline/tracker_impl/siamfcpp_track.py
    Modified to allow control of ROI size.

2. video_analyst/demo/main/video/sot_video.py
    - Extended the time for the first ROI selection from 5s to 30s.
        Press the "s" key during 30 seconds to select the ROI.

    - Changed to create saving folder automatically when the folder does not exist.

    - Changed to synchronously display the tracking in the ROI in the upper left corner of the tracker

3. video_analyst/videoanalyst/utils/visualization.py
    - Solve these problems
        - Video that cannot be played by QuickTime Player on mac is saved.
        - Tracking results (box coordinates) are not output to the file as numerical values.

#### 4. Setup the environment for siamfc++

```bash
cd video_analyst
python setup.py develop
```