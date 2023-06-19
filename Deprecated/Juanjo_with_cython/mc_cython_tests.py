import sys

sys.path.insert(1, '/home/gabriel/OneDrive/2021/Avaqus/Architecture/Juanjo_with_cython')

import Modules.spin_glass as sg
import Modules.mc_cython as mc_c
import Modules.monte_carlo_gjg as mc_n
import numpy as np
import matplotlib.pyplot as plt
import time

# %%
seed = 132212
rng = np.random.default_rng(seed)
size = 200
N_term = 1000
adjacency = 'random_regular_3'
distribution = 'gaussian_EA'

J = sg.connectivity_matrix(size, adjacency, distribution, seed=seed)

T0 = 0.001
Tf = 0.1
copies = 6
T = np.geomspace(T0, Tf, copies)

#%%
_, _, E_vs_t = mc_n.mc_evolution(1 / T, J, steps=1000, start=None, rng=rng)

# %%
t_points = 100
t_intervals = 50

E_t = np.zeros([t_points, copies])

rng = np.random.default_rng(seed)
# s0, E0, *_ = mc_n.mc_evolution(1 / T, J, steps=1, start=None, eq_points=0, rng=rng)

s0 = np.ones((copies, size), dtype='int8')
E0 = mc_n.cost(J, s0)

s = s0.copy()
E_t[0, :] = E0
ti = time.perf_counter()
for t in range(1, t_points):
    s, E_t[t, :], *_ = mc_n.mc_evolution(1 / T, J, steps=t_intervals, start=[s, E_t[t - 1, :].copy()], rng=rng)
ti = time.perf_counter() - ti
print(ti)
# %%
plt.figure(dpi=500)
plt.plot(E_t, linewidth=0.5) #, color='k', marker='.', markersize = 0.5)
plt.ylim([-220, -150])
plt.show()



