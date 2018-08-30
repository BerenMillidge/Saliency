from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

E0 = 5
alpha = 1.

def tradeoff(E, N):
	return E - (E / np.sqrt(N )) - N

def tradeoff2(E,N, alpha):
	return -(E / np.sqrt(N )) - (alpha * N)


xs = [int(i) for i in range(10)]
vals = [tradeoff2(E0, N, alpha) for N in xs]
plt.plot(xs, vals)
plt.title("Hemispheric tradeoffs - accuracy vs metabolic cost")
plt.xlabel("Number of hemispheres")
plt.ylabel("Usefulness score")
plt.show()