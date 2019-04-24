"""
    + Implementing Deming Regression
"""
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn import datasets
tf.reset_default_graph() # reset graph after trainin model
sess = tf.Session()

# 1: prepare data, variables, placeholders, graph ...
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data]) 
y_vals = np.array([y[0] for y in iris.data])

batch_size = 50
x_data = tf.placeholder(tf.float32, shape=[None, 1])
y_target = tf.placeholder(tf.float32, shape=[None, 1])
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
model_output = tf.add(tf.matmul(x_data, A), b)
# 2: Loss function initialize 
demming_numerator = tf.abs(tf.subtract(y_target, tf.add(tf.matmul(x_data, A),b)))
demming_denominator = tf.sqrt(tf.add(tf.square(A), 1))
loss = tf.reduce_mean(tf.truediv(demming_numerator, demming_denominator))

# 3: initialize variables and placeholders
init = tf.global_variables_initializer()
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_step = my_opt.minimize(loss)

loss_vec = []
for i in range(250):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data:rand_x, y_target:rand_y})
    loss_vec.append(temp_loss)
    if (i+1)%50 == 0:
        print("Step # {} A = {} b = {}".format(i+1, sess.run(A), sess.run(b)))
        print("Loss: {}".format(sess.run(loss)))
