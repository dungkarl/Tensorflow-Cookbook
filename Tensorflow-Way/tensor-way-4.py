"""
    + Implementing Loss Functions
"""

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 

sess = tf.Session()

# create values from -1 to 1 /500
x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)
# L2 norm loss is also known as the Euclidean loss function 
l2_y_vals = tf.square(target - x_vals)
l2_y_out = sess.run(l2_y_vals)

# hàm loss L1 là hàm mất mát abs
l1_y_vals = tf.abs(target - x_vals)
l1_y_out = sess.run(l1_y_vals)

# Pseudo-Huber loss

delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.)
phuber1_y_out = sess.run(phuber1_y_vals)
#print('Pseudo-Huber 1', phuber1_y_out)
delta2 = tf.constant(5.)
phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_vals)/delta2)) - 1.)
phuber2_y_out = sess.run(phuber2_y_vals)
#print('Pseudo-Huber 2', phuber2_y_out)

# 6: Hinge loss is mostly used for support vector machines
x_vals = tf.linspace(-3., 5., 500)
target = tf.constant(1.)
targets = tf.fill([500,], 1.)

hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals))
hinge_y_out = sess.run(hinge_y_vals)

# 7: Cross-entropy loss for a binary

xentropy_y_vals = - tf.multiply(target, tf.log(x_vals)) - tf.multiply((1. - target), tf.log(1. - x_vals))
xentropy_y_out = sess.run(xentropy_y_vals)

print('Cross_entropy:', xentropy_y_out)

# 8: Sigmoid cross entropy loss

xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(x_vals, targets)
xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)

# 9: Weighted cross entropy loss is a weighted version of the sigmoid cross entropy loss
weight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals, targets, weight)
xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)

# 10: Softmax cross-entropy loss 
unscaled_logits = tf.constant([[1., -3., 10.]])
target_dist = tf.constant([[0.1, 0.02, 0.88]])
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(unscaled_logits, target_dist)
print(sess.run(softmax_xentropy))