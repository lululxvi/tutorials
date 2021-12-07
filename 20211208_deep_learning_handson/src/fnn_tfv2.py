import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class FNN(tf.keras.Model):
    def __init__(self, layer_sizes):
        super(FNN, self).__init__()
        self.denses = []
        for units in layer_sizes[1:-1]:
            self.denses.append(tf.keras.layers.Dense(units, activation="relu"))
        self.denses.append(tf.keras.layers.Dense(layer_sizes[-1]))

    def call(self, x):
        for f in self.denses:
            x = f(x)
        return x


# Load data
train_data = np.loadtxt("train.txt").astype(np.float32)
train_x, train_y = train_data[:, :1], train_data[:, 1:]
test_data = np.loadtxt("test.txt").astype(np.float32)
test_x, test_y = test_data[:, :1], test_data[:, 1:]

# Hyperparameters
layer_sizes = [1] + [128] * 4 + [1]
lr = 0.001
nsteps = 10000

# Build NN
nn = FNN(layer_sizes)

# Optimizer
opt = tf.keras.optimizers.Adam(learning_rate=lr)


@tf.function
def train(x, y_):
    with tf.GradientTape() as tape:
        y = nn(x)
        # Loss
        loss = tf.reduce_mean((y - y_) ** 2)
    gradients = tape.gradient(loss, nn.trainable_variables)
    opt.apply_gradients(zip(gradients, nn.trainable_variables))
    return loss


# Train
for i in range(nsteps):
    err_train = train(train_x, train_y).numpy()
    if i % 1000 == 0 or i == nsteps - 1:
        pred_y = nn(test_x).numpy()
        err_test = np.mean((pred_y - test_y) ** 2)
        print(i, err_train, err_test)

# Plot
plt.plot(test_x, test_y, "o")
plt.plot(test_x, pred_y, "v")
plt.show()
