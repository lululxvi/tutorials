"""
Solve an inverse problem for the diffusion equation with an unknown parameter C:

    du/dt = C d^2u/dx^2 - exp(-t) (sin(pi x) - pi^2 sin(pi x))

for x in [-1, 1], t in [0, 1]

with the initial condition

    u (x, 0) = sin(pi x)

and the Dirichlet boundary condition

    u(-1, t) = u(1, t) = 0

Data:
  1 | * * * * * * * * |
    |                 |
t   |                 |
  0 |_________________|
    -1                1
             x

Exact solution:

    u = exp(-t) sin(pi x)

Truth: C = 1
"""

import deepxde as dde
import numpy as np
from deepxde.backend import tf

# Define the geometry
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Define the unknowns and PDE
C = dde.Variable(2.0)  # Expect: 1


def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return (
        dy_t
        - C * dy_xx
        + tf.exp(-x[:, 1:])
        * (tf.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0:1]))
    )


# Define BC
# BC location
def boundary(x, on_boundary):
    return on_boundary


# BC value
def func_bc(x):
    return 0


bc = dde.icbc.DirichletBC(geomtime, func_bc, boundary)

# Define IC
# IC location
def initial(x, on_initial):
    return on_initial


# IC value
def func_ic(x):
    return np.sin(np.pi * x[:, 0:1])


ic = dde.icbc.IC(geomtime, func_ic, initial)

# Data
# Exact solution
def sol(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])


# 8 points at t = 1
observe_x = np.vstack((np.linspace(-1, 1, num=10), np.full((10), 1))).T
observe_y = dde.icbc.PointSetBC(observe_x, sol(observe_x))

# Define the entire PDE
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic, observe_y],
    num_domain=40,
    num_boundary=20,
    num_initial=10,
    anchors=observe_x,
    num_test=10000,
)

# Define network
layer_size = [2] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# Define a Model
model = dde.Model(data, net)

# Compile and Train
model.compile("adam", lr=0.001, external_trainable_variables=C)
variable = dde.callbacks.VariableValue(C, period=1000)
model.train(epochs=10000, callbacks=[variable])
