import numpy as np
import tensorflow as tf
import time
import cv2
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.compiler import tensorrt
from absl import app, flags, logging
from absl.flags import FLAGS
from tensorflow.python.saved_model import tag_constants

flags.DEFINE_string('framework', 'trt', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/retinaface_res50_trt_fp16', 'path to weights file')
flags.DEFINE_string('image', './photo/0_Parade_marchingband_1_149.jpg', 'path to input image')
flags.DEFINE_integer('size', 640, 'resize images to')
flags.DEFINE_integer('batchsize', 1, 'resize images to')


def main(_argv):
    input_size = FLAGS.size
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    if FLAGS.framework == 'tf':
        input_layer = tf.keras.layers.Input([input_size, input_size, 3])

    elif FLAGS.framework == 'trt':
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        signature_keys = list(saved_model_loaded.signatures.keys())
        print(signature_keys)
        infer = saved_model_loaded.signatures['serving_default']
    logging.info('weights loaded')

    sum = 0
    original_image = cv2.imread(FLAGS.image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = cv2.resize(original_image, (FLAGS.size, FLAGS.size))
    images_data = []
    for i in range(FLAGS.batchsize):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)
    # image_data = image_data[np.newaxis, ...].astype(np.float32)
    # img_raw = tf.image.decode_image(
    #     open(FLAGS.image, 'rb').read(), channels=3)
    # img_raw = tf.expand_dims(img_raw, 0)
    # img_raw = tf.image.resize(img_raw, (FLAGS.size, FLAGS.size))
    batched_input = tf.constant(images_data)
    for i in range(1000):
        prev_time = time.time()
        # pred_bbox = model.predict(image_data)
        if FLAGS.framework == 'tf':
            # pred_bbox = run_model(images_data)
            infer(batched_input)
        elif FLAGS.framework == 'trt':
            pred_bbox = infer(batched_input)
            # for key, value in pred_bbox.items():
            #     print(key)
        # pred_bbox = pred_bbox.numpy()
        curr_time = time.time()
        exec_time = curr_time - prev_time
        if i == 0: continue
        sum += (1 / exec_time)
        info = str(i) + " time:" + str(round(exec_time, 3)) + " average FPS:" + str(
            round(sum / i, 2)) + ", FPS: " + str(
            round((1 / exec_time), 1))
        print(info)
        # time.sleep(0.05)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
