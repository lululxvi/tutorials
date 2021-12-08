import matplotlib.pyplot as plt
import numpy as np
import torch


class FNN(torch.nn.Module):
    def __init__(self, layer_sizes):
        super(FNN, self).__init__()
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

    def forward(self, x):
        for linear in self.linears[:-1]:
            x = torch.nn.functional.relu(linear(x))
        x = self.linears[-1](x)
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
optimizer = torch.optim.Adam(nn.parameters(), lr=lr)

# Train
for i in range(nsteps):
    y = nn(torch.from_numpy(train_x))
    # Loss
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(y, torch.from_numpy(train_y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 1000 == 0 or i == nsteps - 1:
        with torch.no_grad():
            pred_y = nn(torch.from_numpy(test_x)).detach().cpu().numpy()
        err_test = np.mean((pred_y - test_y) ** 2)
        print(i, loss.item(), err_test)

# Plot
plt.plot(test_x, test_y, "o")
plt.plot(test_x, pred_y, "v")
plt.show()
