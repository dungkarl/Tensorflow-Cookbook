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