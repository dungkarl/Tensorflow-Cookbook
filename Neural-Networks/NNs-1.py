"""
    + Implementing Operational Gates
    + Working with Gates and Activation Functions
    + Implementing a One-Layer Neural Network
    + Implementing Different Layers
    + Using Multilayer Networks
    + Improving Predictions of Linear Models
    + Learning to Play Tic Tac Toe
"""

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 


tf.reset_default_graph()
sess = tf.Session()


#----------------------------------
# Create a multiplication gate:
#   f(x) = a * x
#
#  a --
#      |
#      |---- (multiply) --> output
#  x --|
#
a = tf.Variable(tf.constant(4.))
x_vals = 5.
x_data = tf.placeholder(tf.float32)

mutil = tf.multiply(a, x_data)

loss = tf.square(tf.subtract(mutil,50))

init = tf.global_variables_initializer()
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

print('Optimizing a Multiplication Gate Output to 50.')
for _ in range(10):
    sess.run(train_step, feed_dict={x_data:x_vals})
    a_val = sess.run(a)
    mult_out = sess.run(mutil, feed_dict={x_data:x_vals})
    print(str(a_val) + ' * ' + str(x_vals) + ' = ' + str(mult_out))


tf.reset_default_graph()
sess = tf.Session()


'''
Create a nested gate:
   f(x) = a * x + b
  a --
      |
      |-- (multiply)--
  x --|              |
                     |-- (add) --> output
                 b --|
'''

print("--------------New output----------------")
a = tf.Variable(tf.constant(1.))
b = tf.Variable(tf.constant(1.))
x_val = 5.
x_data = tf.placeholder(tf.float32)

two_gate = tf.add(tf.multiply(a,x_data), b)


loss = tf.square(tf.subtract(two_gate,50))

init = tf.global_variables_initializer()
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = my_opt.minimize(loss)

for _ in range(10):
    sess.run(train_step, feed_dict={x_data:x_val})
    a_val, b_val = sess.run(a), sess.run(b)
    two_gate_out = sess.run(two_gate, feed_dict={x_data:x_val})
    print(str(a_val) + ' * ' + str(x_val) + ' + ' + str(b_val) + ' = ' + str(two_gate_out))