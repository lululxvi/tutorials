from __future__ import division
from __future__ import print_function

import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def dense(inputs, units, use_activation=True):
    shape = inputs.get_shape().as_list()
    fan_in = shape[1]
    W = tf.Variable(tf.random_normal(
        [fan_in, units],
        # stddev=math.sqrt(2 / (fan_in+units))))
        stddev=math.sqrt(2 / fan_in)))
    b = tf.Variable(tf.zeros(units))
    y = tf.matmul(inputs, W) + b
    if use_activation:
        return tf.nn.relu(y)
    return y


def main():
    layer_size = [1] + [20] * 10 + [1]
    lr = 0.001
    nsteps = 50000

    # input
    x = tf.placeholder(tf.float32, [None, layer_size[0]])

    # build NN
    y = x
    for i in range(len(layer_size) - 2):
        y = dense(y, layer_size[i+1])
    y = dense(y, layer_size[-1], use_activation=False)

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
    train_data = np.loadtxt('train2.txt')
    train_x, train_y = train_data[:, :1], train_data[:, 1:]

    test_data = np.loadtxt('test2.txt')
    test_x, test_y = test_data[:, :1], test_data[:, 1:]

    # normalize data
    x_mean, x_std = np.mean(train_x), np.std(train_x)
    train_x = (train_x - x_mean) / x_std
    test_x = (test_x - x_mean) / x_std

    # train
    for i in range(nsteps):
        err_train, _ = sess.run(
            [loss, train], feed_dict={x: train_x, y_: train_y})
        if i % 1000 == 0 or i == nsteps - 1:
            err_test, pred_y = sess.run(
                [loss, y], feed_dict={x: test_x, y_: test_y})
            print(i, err_train, err_test)

    # plot
    plt.plot(test_x, test_y, 'o')
    plt.plot(test_x, pred_y, 'v')
    plt.show()


if __name__ == '__main__':
    main()
