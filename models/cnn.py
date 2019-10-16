import tensorflow as tf 
from tensorflow.python.client import device_lib
print("Listing local devices: {}".format(device_lib.list_local_devices()))
print("Is built with Cuda: {}".format(tf.test.is_built_with_cuda))
print("Gpu Available: {}".format(tf.compat.v1.test.is_gpu_available))
print("GPU devices: {}".format(tf.config.experimental.list_physical_devices('GPU')))