"""
Solve the Poisson equation:

    - d^2u/dx^2 = pi^2 sin(pi * x)

for x in [-1, 1]

with Dirichlet boundary conditions:

    u(-1) = u(1) = 0

Exact solution:

    u(x) = sin(pi * x)
"""

import deepxde as dde
import numpy as np
from deepxde.backend import tf


# Define the geometry
geom = dde.geometry.Interval(-1, 1)

# Define the PDE
def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return -dy_xx - np.pi ** 2 * tf.sin(np.pi * x)


# Define BC
# BC location
def boundary(x, on_boundary):
    return on_boundary


# BC value
def func(x):
    return 0


bc = dde.icbc.DirichletBC(geom, func, boundary)


# Exact solution (Optional)
def sol(x):
    return np.sin(np.pi * x)


# Define the entire PDE
data = dde.data.PDE(geom, pde, bc, 16, 2, solution=sol, num_test=100)

# Define network
layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# Define a Model
model = dde.Model(data, net)

# Compile and Train
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=10000)

# Plot the loss trajectory and PINN solution
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
