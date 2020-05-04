import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

'''
Console output if all is set up in a correct way:
----------------------------------------------------------------------------------------
This test will end up with an error if GPU training is not possible.
Num GPUs Available:  1
----------------------------------------------------------------------------------------

Debugging (CPU):
tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)
----------------------------------------------------------------------------------------

Debugging (GPU):
tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)
'''

tf.debugging.set_log_device_placement(True)  # Logs placement of GPU device
print("----------------------------------------------------------------------------------------")
print("This test will end up with an error if GPU training is not possible.")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

print("----------------------------------------------------------------------------------------")
print("\nDebugging (CPU):")

with tf.device('/CPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

c = tf.matmul(a, b)
print(c)

print("----------------------------------------------------------------------------------------")
print("\nDebugging (GPU):")
# Create some tensors
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

print(c)
