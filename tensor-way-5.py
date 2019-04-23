"""
    + Implementing Back Propagation
"""

import tensorflow as tf 
import numpy as np

sess = tf.Session()

#3: Next we create the data, placeholders, and the A variable
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1]))
print(x_vals)
# 4: We add the multiplication operation to our graph
my_output = tf.multiply(x_data, A)

# 5:Next we add our L2 loss function between the multiplication output and the target data:
loss = tf.square(my_output - y_target)

# 6: Before we can run anything, we have to initialize the variables:
init = tf.global_variables_initializer()
sess.run(init)

# 7: 
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)


for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target:rand_y})
    if (i+1)%25==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data:rand_x, y_target: rand_y})))