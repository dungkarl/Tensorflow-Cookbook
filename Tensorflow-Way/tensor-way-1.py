"""
    All step in file 
    + Operations in a Computational Graph
    + Layering Nested Operations
    + Working with Multiple Layers
    + Implementing Loss Functions
    + Implementing Back Propagation
    + Working with Batch and Stochastic Training
    + Combining Everything Together
    + Evaluating Models
"""

import tensorflow as tf 
import numpy as np

sess = tf.Session()

# A: Operations in a Computational Graph
# x_vals = np.array([1., 3., 5., 7., 9.])
# x_data = tf.placeholder(tf.float32)
# m_const = tf.constant(3.)
# my_product = tf.multiply(x_data, m_const)
# for x_val in x_vals:
#     print(sess.run(my_product, feed_dict={x_data: x_val}))

"""--------------------------------------------------------------------
"""

# B: Layering Nested Operations

# 1: Tạo dữ liệu để cung cấp cho placeholder tương ứng

my_array = np.array([[1., 3., 5., 7., 9.],
                    [-2., 0., 2., 4., 6.],
                    [-6., -3., 0., 3., 6.]])
x_vals = np.array([my_array, my_array + 1])
x_data = tf.placeholder(tf.float32, shape=(3, 5))

# Khi chưa biết chính xác số lượng features của data ta để None
# x_data = tf.placeholder(tf.float32, shape=(3,None)) 
# 2:Tạo các hằng số dùng để nhân và cộng ma trận
m1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

# 3: Khai báo các hoạt động trên graph
prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2, a1)
# 4: Feed data thông qua graph đã tạo
for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data: x_val}))

"""-------------------------------------------------------------------
"""
# C: Working with Multiple Layers

# 1: Tạo samples 2D image bằng numpy
x_shape = [1,4,4,1]
x_val = np.random.uniform(size=x_shape)

# 2: Tạo placeholder cho graph 
x_data = tf.placeholder(tf.float32, shape=x_shape)
