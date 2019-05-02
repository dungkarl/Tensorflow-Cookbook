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

