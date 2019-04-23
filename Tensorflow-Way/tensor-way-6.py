"""
    + Working with Batch and Stochastic Training
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

sess = tf.Session()
# 1: bắt đầu bằng cách khai báo batch size
batch_size = 20

# 2: declare the data, placeholder, variable
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1,1]))

#3:
my_output = tf.matmul(x_data, A)
# 4:
loss = tf.reduce_mean(tf.square(my_output - y_target))
# 5: declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)
# 6: Training model
loss_batch = []
for i in range(100):
    rand_index = np.random.choice(100, size = batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})
    if (i+1)%5==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_batch.append(temp_loss)
