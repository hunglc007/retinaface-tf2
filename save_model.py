from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import tfcoreml
import time
from tensorflow.python.saved_model import signature_constants

from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output)


flags.DEFINE_string('cfg_path', './configs/retinaface_res50.yaml',
                    'config file path')
flags.DEFINE_string('weights', './checkpoints/retinaface_res50', 'path to input image')
flags.DEFINE_string('output', './checkpoints/retinaface_res50.tf', 'path to input image')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')

def main(_argv):
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)

    # load checkpoint
    checkpoint_dir = FLAGS.weights
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    model.summary()
    for i in model.layers:
        print(i.output)
    model.save(FLAGS.output)
    # model.save("model.h5")

    # model = tfcoreml.convert(FLAGS.output,
    #                          mlmodel_path='./model.mlmodel',
    #                          input_name_shape_dict={'input_image': (1, 320, 320, 3)},
    #                          output_feature_names=['Identity'],
    #                          minimum_ios_deployment_target='13')
    # model.save('./checkpoints/keras_model.mlmodel')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
