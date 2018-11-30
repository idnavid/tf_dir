"""Simple toy experiment for TensorFlow to compute dnn regression for Sinc function."""

import numpy as np
import math
import tensorflow as tf
from sklearn.cross_decomposition import CCA
import scipy.stats 

import sys
sys.path.append('/home/nshokouhi/ccaorderselection/tools/svcca/')
import cca_core

# Sinc function
n_observations = 2000
xs = np.linspace(-10, 10, n_observations)
ys = np.zeros((n_observations,4))
for i in range(4):
    ys[:,i] = np.random.randn(1)*np.sin(2*math.pi*(i+1)*xs/4 + 2*math.pi*np.random.randn(1))
ys = ys/np.max(ys)
ys = ys + 0.1*np.random.randn(n_observations,ys.shape[1])
xs = xs.reshape((n_observations,1))

dimensions=[1, 200, 200, 200, 4]
pwd = '/home/nshokouhi/ccaorderselection/tools/tf_dir/regression/'

# variables which we need to fill in when we are ready to compute the graph.
X = tf.placeholder(tf.float32, [None, dimensions[0]], name='X')
Y = tf.placeholder(tf.float32, [None, dimensions[-1]], name='Y')

current_input = X
activation_list = []
for layer_i, n_output in enumerate(dimensions[1:]):
    print(layer_i)
    n_input = int(current_input.get_shape()[1])
    W = tf.Variable(
        tf.random_uniform([n_input, n_output],
                          -1.0 / math.sqrt(n_input),
                          1.0 / math.sqrt(n_input)))
    b = tf.Variable(tf.zeros([n_output]))
    if (layer_i+2) == len(dimensions):
        # Last layer is a linear regression layer
        output = tf.nn.tanh(tf.matmul(current_input,W) + b)
    else:
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
    current_input = output
    activation_list.append(current_input)



cost = tf.reduce_sum(tf.pow(current_input - Y, 2)) / (n_observations - 1)

# %% Use gradient descent to optimize network
# Performs a single step in the negative gradient
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# %% We create a session to use the graph
n_epochs = 50000
with tf.Session() as sess:
    # Here we tell tensorflow that we want to initialize all
    # the variables in the graph so we can use them
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    prev_training_cost = 0.0
    for epoch_i in range(n_epochs):
        for (x, y) in zip(xs, ys):
            x = x.reshape((1,dimensions[0]))
            y = y.reshape((1,dimensions[-1]))
            sess.run(optimizer, feed_dict={X: x, Y: y})

        training_cost = sess.run(
            cost, feed_dict={X: xs, Y: ys})
        print(training_cost)

        if epoch_i%100 == 0:
            second_layer = activation_list[0].eval(feed_dict={X: xs}, session=sess)
            last_layer = activation_list[-1].eval(feed_dict={X: xs}, session=sess)
         
        # Allow the training to quit if we've reached a minimum
        MIN_LOSS = 0.000001
        if np.abs(prev_training_cost - training_cost) < MIN_LOSS:
            second_layer = activation_list[0].eval(feed_dict={X: xs}, session=sess)
            last_layer = activation_list[-1].eval(feed_dict={X: xs}, session=sess)
            print("Training loss < "+str(MIN_LOSS))
            break
        prev_training_cost = training_cost
