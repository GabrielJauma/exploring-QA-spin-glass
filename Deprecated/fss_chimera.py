from tqdm import tqdm
import Modules.spin_glass as sg
import Modules.monte_carlo as mc
import numpy as np
from joblib import Parallel, delayed

sizes = np.array([3, 4, 5]) ** 2 * 8
adjacencies = ['chimera']

threads = 96
T0 = 0.1
Tf = 0.8
copies = 20
N_configs = threads * 100 * 2
N_term = 10000
distribution = 'binary'

T = np.zeros(copies * 2)
T[0::2] = np.geomspace(T0, Tf, copies)
T[1::2] = T[0::2].copy()
B = np.zeros([len(adjacencies), len(sizes), copies])
fname = f'Data/Model chimera - {distribution}, sizes = {sizes}, T in [{T[0]}, {T[-1]}], ' \
        f'N_configs = {N_configs}, N_term = {N_term}'

for i, adjacency in enumerate(adjacencies):
    print('Adjacency', i + 1, 'of', adjacencies)
    for j, size in enumerate(sizes):
        print('Size', size, 'of', sizes)
        q2_q4_c = Parallel(n_jobs=threads)(
            delayed(mc.equilibrate_and_average)
            (1 / T, sg.connectivity_matrix(size, adjacency, distribution, 17 * i), N_term)
            for i in tqdm(range(N_configs)))
        q2 = np.array([q2_q4_c[i][0] for i in range(N_configs)])
        q4 = np.array([q2_q4_c[i][1] for i in range(N_configs)])

        B[i, j, :] = 0.5 * (3 - np.mean(q4, 0) / np.mean(q2, 0) ** 2)
        np.savez(fname, sizes=sizes, adjacencies=adjacencies, T=T, B=B)
