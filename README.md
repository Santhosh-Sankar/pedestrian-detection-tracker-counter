# Pedestrian tracking and counting using Yolov3 and DeepSORT.

<p align='center'>
    <img src="/data/videos/demo3.gif" alt="animation" width="1000"/>
</p>

This repository contains code for detecting, tracking and counting pedestrians using Yolov3 as object or pedestrian detector and utilizes DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric) for real-time pedestrian tracking and counting. The repository also contains code for generating detections for tracker evaluation using MOT Benchmarks.  


## Usage

### Installation

#### Using conda
```
conda env create -f environment.yml
conda activate ped-track
```

#### Pip
```
pip install -r requirements.txt
```

### Download and convert the pretrained Darknet weights to Tensorflow weights
Download the yolov3 weights pretrained on COCO dataset. 
```
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf
```

### Run the tracker 
```
python pedestrian_detection.py --video='./data/videos/<VIDEO FILE> --output='./data/videos/results.avi'
```

### Get detections for evaluation
Download MOT16 dataset here: https://motchallenge.net/data/MOT16/
```
#Generate yolov3 detections
python generate_detections.py

#Convert and save detections in numpy format
python tools/generate_detections_np.py --model='./resources/networks/mars-small128.pb' --mot_dir=<MOT LOCATION> --detection_dir=<DETECTION FOLDER LOCATION> --output_dir=<OUTPUT FOLDER LOCATION>

#Update tracking IDs of the detections
python evaluate_motchallenge.py --mot_dir=<MOT LOCATION> --detection_dir=<DETECTION FOLDER LOCATION> --output_dir=<OUTPUT FOLDER LOCATION>
```
### Outputs
The code outputs the detected pedestrians with a bounding box and the path of motion of the centre point of the bounding boxes for the last 50 frames are plotted. The total number of pedestrians detected and the pedestrians currently visible in the frame are displayed.

#### Less crowded dataset
<p align='center'>
    <img src="/data/videos/demo1.gif" alt="animation" width="1000"/>
</p>

#### More crowded dataset
<p align='center'>
    <img src="/data/videos/demo2.gif" alt="animation" width="1000"/>
</p>

### Tracker Evaluation results
The tracker was evaluated using MOT Benchmark (https://motchallenge.net/) with MOT16 Dataset (https://motchallenge.net/data/MOT16/) which contains 14 challenging video sequences (7 training, 7 test) in unconstrained environments filmed with both static and moving cameras. The evaluation was done on the training set and HOTA and CLEAR-MOT metrics were computed. 

| Metric         | Yolov3 & DeepSORT |
| :-----------:  | :---------------: |
| **HOTA**       | 33.226            |
| **MOTA**       | 33.476            |
| **MOTP**       | 75.065            |
| **Rcll**       | 40.597            |
| **Prcn**       | 86.118            |
| **AssA**       | 36.797            | 
| **DetA**       | 30.459            |
| **AssRe**      | 41.491            |
| **AssPr**      | 68.809            |
| **DetRe**      | 32.602            |
| **DetPr**      | 69.159            |


The tracking results were compared with another tracker utilizing Faster RCNN as object/pedestrian detector trained on COCO dataset and DeepSORT as tracker to analyze differences in performance of DeepSORT with different object detectors. The detailed results can be found here: https://github.com/Santhosh-Sankar/pedestrian-detection-tracking-counting/wiki


#### References
- [Yolov3] (https://github.com/zzh8829/yolov3-tf2)
- [DeepSORT] (https://github.com/nwojke/deep_sort) 
- [MOT_Benchmark] (https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md)















