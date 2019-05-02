"""
Combining Gates and Activation Functions
This function shows how to implement
various gates with activation functions
in TensorFlow
This function is an extension of the
prior gates, but with various activation
functions.

    + Working with Gates and Activation Functions
"""

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 


tf.reset_default_graph()

sess = tf.Session()
tf.set_random_seed(5)
np.random.seed(42)

batch_size = 50

a1 = tf.Variable(tf.random_normal(shape=[1, 1]))
b1 = tf.Variable(tf.random_uniform(shape=[1, 1]))
a2 = tf.Variable(tf.random_normal(shape=[1, 1]))
b2 = tf.Variable(tf.random_uniform(shape=[1, 1]))
x = np.random.normal(2, 0.1, 500)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)


# create activation function

sigmoid_activation = tf.sigmoid(tf.add(tf.multiply(a1, x_data), b1))
relu_activation = tf.nn.relu(tf.add(tf.multiply(a2, x_data), b2))


loss1 = tf.reduce_mean(tf.square(tf.subtract(sigmoid_activation, 0.75)))
loss2 = tf.reduce_mean(tf.square(tf.subtract(relu_activation, 0.75)))

# initialize variables global
init = tf.global_variables_initializer()
sess.run(init)

# declare optimizer

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step_sigmoid = my_opt.minimize(loss1)
train_step_relu = my_opt.minimize(loss2)
print('\nOptimizing Sigmoid AND Relu Output to 0.75')

loss_vec_sigmoid = []
loss_vec_relu = []

for i in range(1000):
    rand_indices = np.random.choice(len(x), size=batch_size)
    x_vals = np.transpose([x[rand_indices]])
    sess.run(train_step_sigmoid, feed_dict={x_data: x_vals})
    sess.run(train_step_relu, feed_dict={x_data: x_vals})
    loss_vec_sigmoid.append(sess.run(loss1, feed_dict={x_data:x_vals}))
    loss_vec_relu.append(sess.run(loss2, feed_dict={x_data:x_vals}))

    sigmoid_output = np.mean(sess.run(sigmoid_activation, feed_dict={x_data: x_vals}))
    relu_output = np.mean(sess.run(relu_activation, feed_dict={x_data: x_vals}))
    
    if i % 50 == 0:
        print('sigmoid = ' + str(np.mean(sigmoid_output)) + ' relu = ' + str(np.mean(relu_output)))

# Plot the loss
plt.plot(loss_vec_sigmoid, 'k-', label='Sigmoid Activation')
plt.plot(loss_vec_relu, 'r--', label='Relu Activation')
plt.ylim([0, 1.0])
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()