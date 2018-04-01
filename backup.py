import matplotlib.pyplot as plt
import numpy as np


def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)


t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 10.0, 0.1)
ep = np.arange(0, 100, 1)
plt.figure("BPNN")
plt.subplot(211)
plt.plot(t2, f(t2), 'k')
plt.subplot(212)
plt.plot(ep, 'r')
plt.show()
