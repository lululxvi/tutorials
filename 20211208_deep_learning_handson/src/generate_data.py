import numpy as np


def f(x):
    return np.heaviside(x, 0.5) + x * np.sin(20 * x) / 10


def main():
    ntrain = 256
    ntest = 512

    x = np.linspace(-(3 ** 0.5), 3 ** 0.5, num=ntrain)
    y = f(x)
    np.savetxt("train.txt", np.vstack((x, y)).T)

    x = np.linspace(-(3 ** 0.5), 3 ** 0.5, num=ntest)
    y = f(x)
    np.savetxt("test.txt", np.vstack((x, y)).T)


if __name__ == "__main__":
    main()
