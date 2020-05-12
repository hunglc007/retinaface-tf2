import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import cv2
import os

flags.DEFINE_string('weights', './checkpoints/retinaface_res50.tf', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/retinaface_res50-fp16.tflite', 'path to output')
flags.DEFINE_integer('input_size', 640, 'path to output')
flags.DEFINE_string('quantize_mode', "float16", 'quantize mode (int8, float16, full_int8)')
flags.DEFINE_string('dataset', "/media/user/Source/Data/coco_dataset/coco/5k.txt", 'path to dataset')

def representative_data_gen():
  fimage = open(FLAGS.dataset).read().split()
  for input_value in range(10):
    if os.path.exists(fimage[input_value]):
      original_image=cv2.imread(fimage[input_value])
      original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
      img_in = original_image[np.newaxis, ...].astype(np.float32)
      print(input_value)
      yield [img_in]
    else:
      continue

def save_tflite():
  model = tf.saved_model.load(FLAGS.weights)
  concrete_func = model.signatures[
    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  concrete_func.inputs[0].set_shape([1, FLAGS.input_size, FLAGS.input_size, 3])
  converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
  if FLAGS.quantize_mode == 'int8':
    converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0]: (0., 1.)}  # mean_value, std_dev
  elif FLAGS.quantize_mode == 'float16':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
  elif FLAGS.quantize_mode == 'full_int8':
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    converter.representative_dataset = representative_data_gen

  tflite_model = converter.convert()
  open(FLAGS.output, 'wb').write(tflite_model)

  logging.info("model saved to: {}".format(FLAGS.output))

def demo():
  interpreter = tf.lite.Interpreter(model_path=FLAGS.output)
  interpreter.allocate_tensors()
  logging.info('tflite model loaded')

  input_details = interpreter.get_input_details()
  print(input_details)
  output_details = interpreter.get_output_details()
  print(output_details)

  input_shape = input_details[0]['shape']

  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

  print(output_data)

def main(_argv):
  save_tflite()
  demo()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


