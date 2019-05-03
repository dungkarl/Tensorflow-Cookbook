"""
    Implementing a one-layer Neural Network
    We will illustrate how to create a one hidden layer NN
    We will use the iris data for this exercise
    We will build a one-hidden layer neural network
    to predict the fourth attribute, Petal Width from
    the other three (Sepal length, Sepal width, Petal length).

    + Implementing a One-Layer Neural Network

"""
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets


tf.reset_default_graph()
sess = tf.Session()

# load iris dataset 
iris = datasets.load_iris()
# 2:

x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])
# 3:
seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

#4: prepare the data
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))


# 5: Now we will declare the batch size and placeholders for the data and target
batch_size = 50
x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
# 6: create variables for both NN layers

hidden_layer_nodes = 10
A1 = tf.Variable(tf.random_normal(shape=[3, hidden_layer_nodes]))  # inputs -> hidden nodes
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))  # one biases for each hidden node
A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))  # hidden inputs -> 1 output
b2 = tf.Variable(tf.random_normal(shape=[1]))   # 1 bias for the output
# 7:  Declare model operations
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2 ), b2))

# declare loss MSE

loss = tf.reduce_mean(tf.square(tf.subtract(final_output, y_target)))
# declare optimizer

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)
# Training loop


loss_vec = []
test_loss = []
for i in range(500):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data:rand_x, y_target:rand_y})

    loss_vec.append(np.sqrt(temp_loss))
    test_temp_loss = sess.run(loss, feed_dict={x_data:x_vals_test, y_target:np.transpose([y_vals_test])})
    test_loss.append(np.sqrt(test_temp_loss))
    if (i+1) % 50 ==0:
        print("Step # {} Loss: {}".format(i+1, temp_loss))
        print("Step # {} Test Loss: {}".format(i+1, test_temp_loss))

# Plot loss (MSE) over time
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()