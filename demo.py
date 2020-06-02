import tensorflow as tf
from tensorflow.python.saved_model import signature_constants

saved_model_loaded = tf.saved_model.load("./checkpoints/retinaface_res50.tf")
graph_func = saved_model_loaded.signatures[
signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
trt_graph = graph_func.graph.as_graph_def()
for n in trt_graph.node:
    print(n.op)
