import time
import os
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3.model import YoloV3
from yolov3.dataset import transform_images

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_string('mot_dir', './MOT16/train', 'path to MOT16')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)


    yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    # class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    # logging.info('classes loaded')

    for sequence in os.listdir(FLAGS.mot_dir):
        logging.info('Opening {}'.format(sequence))

        sequence_dir = os.path.join(FLAGS.mot_dir, sequence)
        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {int(os.path.splitext(f)[0]): os.path.join(image_dir, f) for f in os.listdir(image_dir)}
        
        for i in range(len(image_filenames)):
            #logging.info('image: {}'.format(i+1))
            img_raw = tf.image.decode_image(open(image_filenames[i+1], 'rb').read(), channels=3)

            img = tf.expand_dims(img_raw, 0)
            img = transform_images(img, FLAGS.size)

            boxes, scores, classes, nums = yolo(img)

            op_img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
            norm_boxes = np.array(boxes[0])
            wh= np.flip(op_img.shape[0:2])
            w, h = wh[0], wh[1]
            x1_norm, y1_norm, x2_norm, y2_norm = norm_boxes[:,0], norm_boxes[:,1], norm_boxes[:,2], norm_boxes[:,3]
            x1, y1, x2, y2 = x1_norm*w, y1_norm*h, x2_norm*w, y2_norm*h
            bbox_w, bbox_h = x2 - x1 , y2 - y1 
            score_arr = np.array(scores[0])
            classes = classes[0]
            
            if i == 0:
                op = np.ones((nums[0], 11))
                op[:,0] *= i+1
                op[:,1] *= -1
                op[:,2] = x1[:nums[0]]
                op[:,3] = y1[:nums[0]]
                op[:,4] = bbox_w[:nums[0]]
                op[:,5] = bbox_h[:nums[0]]
                op[:,6] = score_arr[:nums[0]]
                op[:,7:10] *= -1
                op[:,10] = classes[:nums[0]]

            else:
                op_temp = np.ones((nums[0], 11))
                op_temp[:,0] *= i+1
                op_temp[:,1] *= -1
                op_temp[:,2] = x1[:nums[0]]
                op_temp[:,3] = y1[:nums[0]]
                op_temp[:,4] = bbox_w[:nums[0]]
                op_temp[:,5] = bbox_h[:nums[0]]
                op_temp[:,6] = score_arr[:nums[0]]
                op_temp[:,7:10] *= -1
                op_temp[:,10] = classes[:nums[0]]

                op = np.append(op, op_temp, axis=0)
            
            det_filename = str(sequence) + '_det.txt' 
            with open(det_filename, 'w') as f:
                for j in range(op.shape[0]):
                    if int(op[j,10]) == 0:
                        f.write(str(int(op[j,0])) + ',' + str(int(op[j,1])) + ',' + str(round(float(op[j,2]), 5)) + ',' 
                        + str(round(float(op[j,3]), 5)) + ',' + str(round(float(op[j,4]), 5))+ ',' + str(round(float(op[j,5]), 5))
                        + ',' + str(round(float(op[j,6]), 5)) + ',' + str(int(op[j,7])) + ',' + str(int(op[j,8])) + ',' + str(int(op[j,9])) + "\n")

        logging.info('Detections generated for {}'.format(sequence))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
