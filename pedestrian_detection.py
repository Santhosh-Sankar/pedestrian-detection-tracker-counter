import time
import numpy as np
import tensorflow as tf
import cv2
from collections import deque

from absl import app, flags, logging
from absl.flags import FLAGS

from yolov3.model import YoloV3
from yolov3.dataset import transform_images

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections_np as gdet


flags.DEFINE_string('classes', './data/coco.names', 'classes file path')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/videos/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

centre_points = [deque(maxlen=50) for _ in range(9999)]


def main(_argv):
    #Use GPU if exists
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Define the parameters and variables
    max_cosine_distance = 0.5
    nn_budget = None

    #Initialize deep sort
    model_filename = 'resources/networks/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    yolo = YoloV3(classes=FLAGS.num_classes)

    #Load weights
    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    #Load classes
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    try:
        video = cv2.VideoCapture(int(FLAGS.video))
    except:
        video = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        #Get frame width and height
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))


    fps = 0.0
    count = 0 
    counter = []
    color = {}
    while True:
        curr_obj_counter = 0
    
        _, image = video.read()

        if image is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break
        
        #Reorder channels from BGR to RGB
        image_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image_in = tf.expand_dims(image_in, 0)
        image_in = transform_images(image_in, FLAGS.size)
        
        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(image_in)
        classes = classes[0]
        scores = scores[0]
        names = []
        

        for i in range(nums[0]):
            names.append(class_names[int(classes[i])])
        names = np.array(names)

        #Convert bboxes to tlwh format
        tlwh_boxes = boxes[0, :nums[0], :]
        wh= np.flip(image.shape[0:2])
        w, h = wh[0], wh[1]
        tlwh_boxes[:,0], tlwh_boxes[:,1], tlwh_boxes[:,2], tlwh_boxes[:,3] = \
            tlwh_boxes[:,0]*w, tlwh_boxes[:,1]*h, (tlwh_boxes[:,2] - tlwh_boxes[:,0])*w, (tlwh_boxes[:,3] - tlwh_boxes[:,1])*h

        features = encoder(image, tlwh_boxes.tolist())    
        detections = [Detection(bbox, score, feature, class_name=class_name) for bbox, score, class_name, feature in zip(tlwh_boxes.tolist(), scores, names, features)]     

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            class_name = track.get_class()
            if str(class_name) == 'person':
                if int(track.track_id) not in counter:
                    counter.append(int(track.track_id))
                curr_obj_counter += 1
                bbox = track.to_tlbr()

                #Generate unique color of bounding boxes
                if int(track.track_id) not in color:
                    box_color = np.random.randint(256, size=3)
                    while (int(box_color[0]), int(box_color[1]), int(box_color[2]))  in color.values():
                        box_color = np.random.randint(256, size=3)
                    color[int(track.track_id)] = (int(box_color[0]), int(box_color[1]), int(box_color[2])) 
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color[int(track.track_id)], 2)
                cv2.putText(image, class_name + " ID: " + str(track.track_id),(int(bbox[0]), int(bbox[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[int(track.track_id)],2)

                #Find center point of bounding box
                box_center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
                centre_points[track.track_id].append(box_center)
                cv2.circle(image,  (box_center), 1, color[int(track.track_id)] , 5)

                #Plot the centre of the oldest frame
                cv2.circle(image, centre_points[track.track_id][0], 1, color[int(track.track_id)] , 10)

                #Plot the motion path
                for j in range(1, len(centre_points[track.track_id])):
                    if centre_points[track.track_id][j - 1] is None or centre_points[track.track_id][j] is None:
                        continue
                    thickness = 2
                    #Plot line betwen centres of 2 consecutive frames
                    cv2.line(image,(centre_points[track.track_id][j-1]), (centre_points[track.track_id][j]), color[int(track.track_id)], thickness)

        
    
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        count = len(set(counter))

        #FPS info for performance
        #cv2.putText(image, "FPS: {:.2f}".format(fps), (20, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(image, "Pedestrians Detected: "+str(count),(20, 450),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
        cv2.putText(image, "Pedestrians Visible: "+str(curr_obj_counter),(20, 500),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
        cv2.imshow('output', image)
        if FLAGS.output:
            out.write(image)

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()

    if FLAGS.output:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
