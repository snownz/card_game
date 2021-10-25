"""
    Configure Tensorflow
"""
import tensorflow as tf

# tf.debugging.set_log_device_placement( True )
tf.config.set_soft_device_placement( True )

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")