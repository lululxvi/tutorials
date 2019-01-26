from __future__ import division

import numpy as np


def main():
    def f(x):
        # return x * np.sin(10 * x)
        return np.heaviside(x, 0.5) + x*np.sin(20*x) / 10

    ntrain = 256
    ntest = 512

    x = 2*3**0.5*np.random.rand(ntrain) - 3**0.5
    y = f(x)
    np.savetxt('train.txt', np.vstack((100*x+100, y)).T)

    x = np.linspace(-3**0.5, 3**0.5, num=ntest)
    y = f(x)
    np.savetxt('test.txt', np.vstack((100*x+100, y)).T)


if __name__ == '__main__':
    main()
