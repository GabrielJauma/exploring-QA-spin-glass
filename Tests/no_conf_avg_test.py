import Modules.spin_glass as sg
import Modules.monte_carlo as mc
import numpy as np

sizes = np.array([10])
adjacencies = ['random_regular_3']
threads = 96
T0 = 0.1
Tf = 1.5
copies = 50
eq_steps = int(1e3)
N_term = int(1e6)
distribution = 'gaussian_EA'

N_configs = 1
T = np.zeros(copies * 2)
T[0::2] = np.geomspace(T0, Tf, copies)
T[1::2] = T[0::2].copy()
B = np.zeros([len(adjacencies), len(sizes), copies])
fname = f'Data/Model Random_regular_graphs - {distribution}, sizes = {sizes}, T in [{T[0]}, {T[-1]}], ' \
        f'N_configs = {N_configs}, N_term = {N_term}, eq_steps = {eq_steps}'

seeds = np.random.SeedSequence().spawn(N_configs)

for i, adjacency in enumerate(adjacencies):
    print('Adjacency', i + 1, 'of', adjacencies)
    for j, size in enumerate(sizes):
        print('Size', size, 'of', sizes)
        q2_q4_c = [
            mc.equilibrate_and_average(1 / T, sg.connectivity_matrix(size, adjacency, distribution, rng=rng), N_term, eq_steps=eq_steps, rng=rng)
            for i in range(N_configs)
            for rng in [np.random.default_rng(seed=seeds[i])]]
        q2 = np.array([q2_q4_c[i][0] for i in range(N_configs)])
        q4 = np.array([q2_q4_c[i][1] for i in range(N_configs)])

        B[i, j, :] = 0.5 * (3 - np.mean(q4, 0) / np.mean(q2, 0) ** 2)
        np.savez(fname, sizes=sizes, adjacencies=adjacencies, T=T, B=B)
