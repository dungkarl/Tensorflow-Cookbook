"""
    + Understanding Loss Functions in Linear Regression
"""
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets

tf.reset_default_graph()
sess = tf.Session()


# 1: 
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])
batch_size = 25
learning_rate = 0.05 # Will not converge with learning rate at 0.4
iterations = 50
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
model_output = tf.add(tf.matmul(x_data, A), b)

# 2: Our loss function will change to the L1 loss
loss_l1 = tf.reduce_mean(tf.abs(y_target - model_output))
loss_l2 = tf.reduce_mean(tf.square(y_target - model_output))
# 3: resume initialize variables and placeholders
init = tf.global_variables_initializer()
sess.run(init)
my_opt_l1 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_step_l1 = my_opt_l1.minimize(loss_l1)
loss_vec_l1 = []
loss_vec_l2 = []
for i in range(iterations):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step_l1, feed_dict={x_data:rand_x, y_target:rand_y})
    temp_loss_l1 = sess.run(loss_l1, feed_dict={x_data:rand_x, y_target:rand_y})
    temp_loss_l2 = sess.run(loss_l2, feed_dict={x_data:rand_x, y_target:rand_y})
    loss_vec_l1.append(temp_loss_l1)
    loss_vec_l2.append(temp_loss_l2)
    if (i+1) %25 ==0:
        print("Step # {}  A = {} b = {}".format(i+1, sess.run(A), sess.run(b)))

plt.plot(loss_vec_l1, 'k-', label='L1 Loss')
plt.plot(loss_vec_l2, 'r--', label='L2 Loss')
plt.title("Loss Function L1 and L2")
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
