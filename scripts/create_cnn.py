# this program trains, and then outputs the CNN used in the javascript script.

from tensorflow.examples.tutorials.mnist import input_data
from math import floor
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import numpy as np
import struct
import math

import tensorflow as tf
sess = tf.InteractiveSession()

import numpy as np
np.set_printoptions(threshold=np.inf)

def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W, name):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], ## CHAN
                        strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5, 5, 1, 16], name='w_conv1')
b_conv1 = bias_variable([16], name='b_conv1')

x_image = tf.reshape(x, [-1,28,28,1])

# conv, bias, relu
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, name='conv2d_1') + b_conv1)

# max pool
h_pool1 = max_pool_2x2(h_conv1)

W_fc1 = weight_variable([14 * 14 * 16, 64], name='w_fc1')
b_fc1 = bias_variable([64], name='b_fc1')

# densely connected layer, then relu.
h_pool2_flat = tf.reshape(h_pool1, [-1, 14*14*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# drop-out
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer, then softmax.
W_fc2 = weight_variable([64, 10], name='w_fc2')
b_fc2 = bias_variable([10], name='b_fc2')
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# next we train the above network.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(10000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

def output_arr(arr, out_file):
    a = arr.astype(dtype=np.float16)
    with open(out_file, 'wb') as f:
        l = len(a.tobytes()) / 2
        f.write(struct.pack('I', l))
        f.write(a.tobytes())

#
# next, we output all the weights of the trained network to binary files.
# but we reformat some of the tensors, so that they are easier to handle
# when implementing the network on the GPU.
#

res = (sess.run(W_conv1, feed_dict={x: mnist.test.images[:1], y_: mnist.test.labels[:1], keep_prob: 1.0}))
arr = np.zeros((res.shape[3],  res.shape[0], res.shape[1] ))

# reformat from [5, 5, 1, 16] to [16, 5, 5]
for i in range(0, res.shape[3]):
    arr[i] = res[:,:,0,i]
output_arr(arr, 'w_conv1.bin')

res = (sess.run(b_conv1, feed_dict={x: mnist.test.images[:1], y_: mnist.test.labels[:1], keep_prob: 1.0}))
output_arr(res, 'b_conv1.bin')

res = (sess.run(W_fc1, feed_dict={x: mnist.test.images[:1], y_: mnist.test.labels[:1], keep_prob: 1.0}))
arr = np.zeros((res.shape[0],  res.shape[1] ))

# h_conv1 is stored in the weird shape [14,14,16], so we have to rearrange some weights:
for col in range(0, res.shape[1]): # iter cols.
    for row in range(0, res.shape[0]): # iter rows.
        my_layer = int(floor(row / (14 * 14)))
        my_pixel = row % (14 * 14)
        my_id = my_layer + (my_pixel) * 16
        arr[row, col] = res[my_id, col]
output_arr(arr, 'w_fc1.bin')

res = (sess.run(b_fc1, feed_dict={x: mnist.test.images[:1], y_: mnist.test.labels[:1], keep_prob: 1.0}))
output_arr(res, 'b_fc1.bin')

res = (sess.run(W_fc2, feed_dict={x: mnist.test.images[:1], y_: mnist.test.labels[:1], keep_prob: 1.0}))
output_arr(res, 'w_fc2.bin')

res = (sess.run(b_fc2, feed_dict={x: mnist.test.images[:1], y_: mnist.test.labels[:1], keep_prob: 1.0}))
output_arr(res, 'b_fc2.bin')
