from keras.models import load_model
import os
import numpy as np
import cv2

from yolo import YOLO, dummy_loss
from preprocessing import load_image_predict, load_image_predict_from_numpy_array
from postprocessing import decode_netout, draw_boxes, get_state


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_model(config):
    model = YOLO(
         config =config
    )
    model.model.load_weights(os.path.join(BASE_DIR, 'checkpoints',config['model']['saved_model_name']))
    return model


def get_model_from_file(config):
    path = os.path.join(BASE_DIR, 'checkpoints', config['model']['saved_model_name'])
    model = load_model(path, custom_objects={'custom_loss': dummy_loss})
    return model


def predict_with_model_from_file(config, model, image_path):
    image = load_image_predict(image_path, config['model']['image_h'], config['model']['image_w'])

    dummy_array = np.zeros((1, 1, 1, 1, config['model']['max_obj'], 4))
    netout = model.model.predict([image, dummy_array])[0]

    boxes = decode_netout(netout=netout, anchors=config['model']['anchors'],
                          nb_class=config['model']['num_classes'],
                          obj_threshold=config['model']['obj_thresh'],
                          nms_threshold=config['model']['nms_thresh'])
    return boxes

# Identification and prediction of trafficlight state
def predict_traffic_light_state(model, image, config):
    """ Identify and predict of trafficlight state

        args:
            model: Keras model for prediction

            image: np array on which apply the detector

            config: json file with config parameters
        returns:
            trafficlight_state: current detections and accuracy from image
                labels can be "go" or "stop", accuracy is float beetween 0 and 1
                format: [[label,accuracy]]
                example: [["go",0.9042]]

            plt_image: image with Bounding Boxes and accuracy

            boxes: parameters on the Bounding boxes from prediction
    """
    image_to_detect = load_image_predict_from_numpy_array(image, config['model']['image_h'],config['model']['image_h'])
    dummy_array = np.zeros((1, 1, 1, 1, config['model']['max_obj'], 4))
    netout = model.model.predict([image_to_detect, dummy_array])[0]
    boxes = decode_netout(netout=netout, anchors=config['model']['anchors'],
            nb_class=config['model']['num_classes'],
            obj_threshold=config['model']['obj_thresh'],
            nms_threshold=config['model']['nms_thresh'])
    plt_image = draw_boxes(image, boxes, config['model']['classes'])
    trafficlight_state = get_state(boxes, config['model']['classes'])

    return trafficlight_state, plt_image, boxes