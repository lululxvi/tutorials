import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# Load data
train_data = np.loadtxt("train.txt").astype(np.float32)
train_x, train_y = train_data[:, :1], train_data[:, 1:]
test_data = np.loadtxt("test.txt").astype(np.float32)
test_x, test_y = test_data[:, :1], test_data[:, 1:]

# Hyperparameters
layer_sizes = [1] + [128] * 4 + [1]
lr = 0.001
nsteps = 10000

# Input
x = tf.placeholder(tf.float32, [None, layer_sizes[0]])
# Build NN
y = x
for i in range(1, len(layer_sizes) - 1):
    y = tf.keras.layers.Dense(layer_sizes[i], activation="relu")(y)
y = tf.keras.layers.Dense(layer_sizes[-1])(y)
# True output
y_ = tf.placeholder(tf.float32, [None, layer_sizes[-1]])

# Loss
loss = tf.reduce_mean((y - y_) ** 2)

# Optimizer
opt = tf.train.AdamOptimizer(lr)
train = opt.minimize(loss)

# Init
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train
for i in range(nsteps):
    err_train, _ = sess.run([loss, train], feed_dict={x: train_x, y_: train_y})
    if i % 1000 == 0 or i == nsteps - 1:
        pred_y, err_test = sess.run([y, loss], feed_dict={x: test_x, y_: test_y})
        print(i, err_train, err_test)

# Plot
plt.plot(test_x, test_y, "o")
plt.plot(test_x, pred_y, "v")
plt.show()
