import numpy as np
import tensorflow as tf
import time
import cv2
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.compiler import tensorrt
from absl import app, flags, logging
from absl.flags import FLAGS
from tensorflow.python.saved_model import tag_constants
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output)

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/retinaface_mbv2.tf', 'path to weights file')
flags.DEFINE_string('image', './photo/0_Parade_marchingband_1_149.jpg', 'path to input image')
flags.DEFINE_integer('size', 640, 'resize images to')
flags.DEFINE_integer('batchsize', 1, 'resize images to')
flags.DEFINE_float('down_scale_factor', 1.0, 'down-scale factor for inputs')
flags.DEFINE_string('cfg_path', './configs/retinaface_mbv2.yaml',
                    'config file path')


def main(_argv):
    cfg = load_yaml(FLAGS.cfg_path)
    input_size = FLAGS.size
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    if FLAGS.framework == 'tf':
        infer = tf.keras.models.load_model(FLAGS.weights)

    elif FLAGS.framework == 'trt':
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        signature_keys = list(saved_model_loaded.signatures.keys())
        print(signature_keys)
        infer = saved_model_loaded.signatures['serving_default']
    logging.info('weights loaded')

    sum = 0

    img_raw = cv2.imread(FLAGS.image)
    img_height_raw, img_width_raw, _ = img_raw.shape
    img = np.float32(img_raw.copy())

    if FLAGS.down_scale_factor < 1.0:
        img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                         fy=FLAGS.down_scale_factor,
                         interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # pad input image to avoid unmatched shape problem
    img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))
    batched_input = img[np.newaxis, ...]
    if FLAGS.framework == 'tf':
        # pred_bbox = run_model(images_data)
        outputs = infer(batched_input).numpy()
    elif FLAGS.framework == 'trt':
        pred_bbox = infer(batched_input)
        # recover padding effect
    outputs = recover_pad_output(outputs, pad_params)

    # draw and save results
    save_img_path = 'out.jpg'
    for prior_index in range(len(outputs)):
        draw_bbox_landm(img_raw, outputs[prior_index], img_height_raw,
                        img_width_raw)
        cv2.imwrite(save_img_path, img_raw)
    print(f"[*] save result at {save_img_path}")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
