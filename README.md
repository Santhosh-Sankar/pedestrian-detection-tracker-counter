# Pedestrian detection, tracking and counting using Yolov3 and DeepSORT.
This repository contains code for detecting, tracking and counting pedestrians using Yolov3 as object or pedestrian detector and utilizes DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric) for real-time pedestrian tracking and counting. The repository also contains code for generating detections for tracker evaluation using MOT Benchmarks.  


<p align='center'>
    <img src="/data/videos/demo3.gif" alt="animation" width="600"/>
</p>

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
The code outputs the detected pedestrians with a bounding box and the path of motion of the centre point of the bounding boxes for the last 50 frames are plotted. The total number of pedestriand detected and the pedestrians currently visible in the frame are displayed.

#### Less crowded dataset
<p align='center'>
    <img src="/data/videos/demo1.gif" alt="animation" width="600"/>
</p>

#### More crowded dataset
<p align='center'>
    <img src="/data/videos/demo2.gif" alt="animation" width="600"/>
</p>


#### References
Yolov3: https://github.com/zzh8829/yolov3-tf2
DeepSORT: https://github.com/nwojke/deep_sort 















