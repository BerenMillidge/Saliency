from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

E0 = 10

def tradeoff(E, N):
	return E - (E / np.sqrt(N)) - N

def tradeoff2(E,N):
	return -(E / np.sqrt(N)) - N


xs = [i for i in range(50)]
vals = [tradeoff2(E0, N) for N in xs]
plt.plot(xs, vals)
plt.show()