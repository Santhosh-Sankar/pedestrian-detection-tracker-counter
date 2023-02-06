import numpy as np
import tensorflow as tf
from .utils import yolo_boxes, yolo_nms
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2


yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


def Darknet_CNN(cnn_in, filters, size, padding='same', strides=1, use_batchnorm=True):
    #To avoid loss of information
    if strides != 1:
        padding = 'valid'
        cnn_in = ZeroPadding2D(((1, 0), (1, 0)))(cnn_in)

    temp_out =  Conv2D(filters=filters, kernel_size=size, padding=padding, strides=strides, \
                use_bias=not use_batchnorm, kernel_regularizer=l2(0.0005))(cnn_in)
    if use_batchnorm:
        temp_out = BatchNormalization()(temp_out)
        cnn_out = LeakyReLU(alpha=0.1)(temp_out)    
    else:
        cnn_out = temp_out
    
    return cnn_out


def Darknet_RNN_layer(rnn_in, filters):
    inputs = rnn_in
    temp_out = Darknet_CNN(rnn_in, filters//2, size=1)
    temp_out = Darknet_CNN(temp_out, filters, size=3)
    rnn_out = Add()([inputs, temp_out])

    return rnn_out


def Darknet_RNN(rnn_in, filters, num_blocks):
    temp_out = Darknet_CNN(rnn_in, filters, size=3, strides=2)
    for _ in range(num_blocks):
        temp_out = Darknet_RNN_layer(temp_out, filters)
    rnn_out = temp_out 

    return rnn_out


def Darknet(**kwargs):
    inputs = darknet_in = Input([None, None, 3])
    temp_out = Darknet_CNN(darknet_in, 32, size=3)
    temp_out= Darknet_RNN(temp_out, 64, 1)
    temp_out = Darknet_RNN(temp_out, 128, 2)
    temp_out = skip_out_36 = Darknet_RNN(temp_out, 256, 8)
    temp_out = skip_out_61 = Darknet_RNN(temp_out, 512, 8)
    darknet_out = Darknet_RNN(temp_out, 1024, 4)

    return tf.keras.Model(inputs, (darknet_out, skip_out_36, skip_out_61), **kwargs)


def Yolo_CNN(filters, **kwargs):
    def yolo_cnn(cnn_in):
        if isinstance(cnn_in, tuple):
            #size of input ignores batch size
            darknet_out, skip_in = yolo_cnn_in = Input(cnn_in[0].shape[1:]), Input(cnn_in[1].shape[1:])
            temp_out = Darknet_CNN(darknet_out, filters, size=1)
            temp_out = UpSampling2D(2)(temp_out)
            #Concatenate output with skip connection
            temp_out = Concatenate()([temp_out, skip_in])
            
        else:
            temp_out = yolo_cnn_in = Input(cnn_in.shape[1:])

        temp_out = Darknet_CNN(temp_out, filters, size=1)
        temp_out = Darknet_CNN(temp_out, filters*2, size=3)
        temp_out = Darknet_CNN(temp_out, filters, size=1)
        temp_out = Darknet_CNN(temp_out, filters*2, size=3)
        cnn_out = Darknet_CNN(temp_out, filters, size=1)

        return Model(yolo_cnn_in, cnn_out, **kwargs)(cnn_in)

    return yolo_cnn


def Yolo_Predictor(filters, anchors, classes, **kwargs):
    def yolo_predictor(cnn_in):
        inputs = predictor_in = Input(cnn_in.shape[1:])
        temp_out = Darknet_CNN(predictor_in, filters*2, size=3)
        #get output dim: batch x grid x grid x 255
        temp_out = Darknet_CNN(temp_out, anchors*(classes+5),
                                size=1,  use_batchnorm=False)
        #get output dim: batch x grid x grid x 3 x 85
        prediction = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(temp_out)
        return tf.keras.Model(inputs, prediction, **kwargs)(cnn_in)
    return yolo_predictor


def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80, training=False):

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    inputs = yolo_in = Input([size, size, channels], name='input')

    darknet_out, skip_out_36, skip_out_61 = Darknet(name='yolo_darknet')(yolo_in)

    temp_out = Yolo_CNN(512, name='yolo_conv_13')(darknet_out)
    output_13 = Yolo_Predictor(512, len(masks[0]), classes, name='yolo_output_13')(temp_out)

    temp_out = Yolo_CNN(256, name='yolo_conv_26')((temp_out, skip_out_61))
    output_26 = Yolo_Predictor(256, len(masks[1]), classes, name='yolo_output_26')(temp_out)

    temp_out = Yolo_CNN(128, name='yolo_conv_52')((temp_out, skip_out_36))
    output_52 = Yolo_Predictor(128, len(masks[2]), classes, name='yolo_output_52')(temp_out)

    if training:
        return Model(inputs, (output_13, output_26, output_52), name='yolov3')

    boxes_13 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_13')(output_13)
    boxes_26 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_26')(output_26)
    boxes_52 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_52')(output_52)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_13[:3], boxes_26[:3], boxes_52[:3]))

    return Model(inputs, outputs, name='yolov3')



















