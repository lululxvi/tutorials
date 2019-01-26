from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def main():
    layer_size = [1, 20, 20, 1]

    # input
    x = tf.placeholder(tf.float32, [None, layer_size[0]])

    # layer 1
    W1 = tf.Variable(tf.random_normal(
        [layer_size[0], layer_size[1]],
        stddev=1))
    b1 = tf.Variable(tf.zeros(layer_size[1]))
    y1 = tf.nn.tanh(tf.matmul(x, W1) + b1)

    # layer 2
    W2 = tf.Variable(tf.random_normal(
        [layer_size[1], layer_size[2]],
        stddev=1))
    b2 = tf.Variable(tf.zeros(layer_size[2]))
    y2 = tf.nn.tanh(tf.matmul(y1, W2) + b2)

    # layer 3
    W3 = tf.Variable(tf.random_normal(
        [layer_size[2], layer_size[3]],
        stddev=1))
    b3 = tf.Variable(tf.zeros(layer_size[3]))
    y = tf.matmul(y2, W3) + b3

    # init
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # load data
    train_x = np.linspace(-1, 1, num=128)[:, None]

    # train
    pred_y = sess.run(y, feed_dict={x: train_x})

    # plot
    plt.plot(train_x, pred_y, 'v')
    plt.show()


if __name__ == '__main__':
    main()
