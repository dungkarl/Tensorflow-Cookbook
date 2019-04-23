"""
    + Working with Multiple Layers
"""



import tensorflow as tf 
import numpy as np 


sess = tf.Session()

# C: Working with Multiple Layers

# 1: Tạo samples 2D image bằng numpy
x_shape = [1,4,4,1]
x_val = np.random.uniform(size=x_shape)

# 2: Tạo placeholder cho graph 
x_data = tf.placeholder(tf.float32, shape=x_shape)

# 3: Tạo filter 
my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
my_strides = [1, 2, 2, 1]
mov_avg_layer= tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='Moving_Avg_Window')
"""
Output = (W-F+2P)/S+1, where W is the input size,
F is the filter size, P is the padding of zeros, and S is the stride.
"""
# 4:
# 5: Define custom layer 
def custom_layer(input_matrix):
    input_matrix_sqeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    temp1 = tf.matmul(A, input_matrix_sqeezed)
    temp = tf.add(temp1, b) # Ax + b
    return(tf.sigmoid(temp))

# 6: 
with tf.name_scope('Custom_Layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)
    print(sess.run(custom_layer1, feed_dict={x_data: x_val}))

    