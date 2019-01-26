from __future__ import division
from __future__ import print_function

import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def main():
    layer_size = [1, 20, 20, 1]
    lr = 0.001
    nsteps = 10000

    # input
    x = tf.placeholder(tf.float32, [None, layer_size[0]])

    # layer 1
    W1 = tf.Variable(tf.random_normal(
        [layer_size[0], layer_size[1]],
        stddev=math.sqrt(2 / (layer_size[0]+layer_size[1]))))
    b1 = tf.Variable(tf.zeros(layer_size[1]))
    y1 = tf.nn.tanh(tf.matmul(x, W1) + b1)

    # layer 2
    W2 = tf.Variable(tf.random_normal(
        [layer_size[1], layer_size[2]],
        stddev=math.sqrt(2 / (layer_size[1]+layer_size[2]))))
    b2 = tf.Variable(tf.zeros(layer_size[2]))
    y2 = tf.nn.tanh(tf.matmul(y1, W2) + b2)

    # layer 3
    W3 = tf.Variable(tf.random_normal(
        [layer_size[2], layer_size[3]],
        stddev=math.sqrt(2 / (layer_size[2]+layer_size[3]))))
    b3 = tf.Variable(tf.zeros(layer_size[3]))
    y = tf.matmul(y2, W3) + b3

    # true output
    y_ = tf.placeholder(tf.float32, [None, layer_size[-1]])

    # loss
    loss = tf.reduce_mean((y - y_)**2)

    # optimizer
    opt = tf.train.AdamOptimizer(lr)
    train = opt.minimize(loss)

    # init
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # load data
    train_x = np.linspace(-1, 1, num=128)[:, None]
    train_y = train_x ** 2

    # train
    for i in range(nsteps):
        err_train, pred_y, _ = sess.run(
            [loss, y, train], feed_dict={x: train_x, y_: train_y})
        if i % 1000 == 0 or i == nsteps - 1:
            print(i, err_train)

    # plot
    plt.plot(train_x, train_y, 'o')
    plt.plot(train_x, pred_y, 'v')
    plt.show()


if __name__ == '__main__':
    main()
