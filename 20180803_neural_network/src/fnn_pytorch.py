import matplotlib.pyplot as plt
import numpy as np
import torch


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, 20)
        self.fc2 = torch.nn.Linear(20, 20)
        self.fc3 = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = torch.nn.functional.tanh(self.fc1(x))
        x = torch.nn.functional.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    lr = 0.001
    nsteps = 10000

    net = Net()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    input = np.linspace(-1, 1, num=128, dtype=np.float32)[:, None]
    target = input ** 2

    for i in range(nsteps):
        optimizer.zero_grad()
        output = net(torch.from_numpy(input).float())
        loss = loss_fn(output, torch.from_numpy(target).float())
        loss.backward()
        optimizer.step()

        if i % 1000 == 0 or i == nsteps - 1:
            print(i, loss.item())

    with torch.no_grad():
        output = net(torch.from_numpy(input).float())

    plt.plot(input, target, "o")
    plt.plot(input, output, "v")
    plt.show()


if __name__ == "__main__":
    main()
