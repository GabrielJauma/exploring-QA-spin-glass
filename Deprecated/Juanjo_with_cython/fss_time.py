import sys
sys.path.insert(1, '/home/gabriel/OneDrive/2021/Avaqus/Architecture/Juanjo_with_cython')

import Modules.spin_glass as sg
import Modules.monte_carlo_new as mc
import Modules.monte_carlo_new2 as mc2
import Modules.monte_carlo_new3 as mc3
import Modules.mc_cython as mc4
import numpy as np
import time

sizes = np.array([200])  # , 250, 400])
adjacencies = ['random_regular_3']  # , 'random_regular_5']

threads = 24
T0 = 0.2
Tf = 1.5
copies = 50
N_configs = threads * 10
N_term = 1000
distribution = 'gaussian_EA'

T = np.zeros(copies * 2)
T[0::2] = np.geomspace(T0, Tf, copies)
T[1::2] = T[0::2].copy()
B = np.zeros([len(adjacencies), len(sizes), copies])
fname = f'Data/Model Random_regular_graphs - {distribution}, sizes = {sizes}, T in [{T[0]}, {T[-1]}], ' \
        f'N_configs = {N_configs}, N_term = {N_term}'

rng1 = np.random.default_rng(13212)
rng2 = np.random.default_rng(13212)
for i, adjacency in enumerate(adjacencies):
    print('Adjacency', i + 1, 'of', adjacencies)
    for j, size in enumerate(sizes):
        print('Size', size, 'of', sizes)
        J = sg.connectivity_matrix(size, adjacency, distribution, 17 * i)
        if True:
            if i == j == 0:
                s, E, *_ = mc.mc_evolution(1 / T, J, steps=None, start=None, eq_points=10)
            t = time.perf_counter()
            s, E, *_ = mc.mc_evolution(1 / T, J, steps=None, start=None, eq_points=10, rng=rng1)
            t = time.perf_counter() - t
        print(f'Total time Numba {t}s')
        print(np.min(E))
        if True:
            if i == j == 0:
                s, E, *_ = mc2.mc_evolution(1 / T, J, steps=None, start=None, eq_points=10)
            t = time.perf_counter()
            s, E, *_ = mc2.mc_evolution(1 / T, J, steps=None, start=None, eq_points=10, rng=rng1)
            t = time.perf_counter() - t
        print(f'Total time Numba2 {t}s')
        print(np.min(E))
        if True:
            if i == j == 0:
                s, E, *_ = mc2.mc_evolution(1 / T, J, steps=None, start=None, eq_points=10)
            t = time.perf_counter()
            s, E, *_ = mc2.mc_evolution(1 / T, J, steps=None, start=None, eq_points=10, rng=rng1)
            t = time.perf_counter() - t
        print(f'Total time Numba3 {t}s')
        print(np.min(E))
        if True:
            t = time.perf_counter()
            s, E = mc4.mc_evolution(1 / T, J, steps=None, start=None, eq_points=10, rng=rng2)
            t = time.perf_counter() - t
        print(f'Total time Cython {t}s')
        print(np.min(E))
