from tqdm import tqdm
import Modules.spin_glass as sg
import Modules.monte_carlo as mc
import numpy as np
from joblib import Parallel, delayed

# sizes = np.array([i ** 2 for i in range(8, 16, 4)])
# sizes = np.array([i ** 3 for i in range(2, 6, 1)])
# sizes = np.array([60, 180, 540])
sizes = np.array([72, 128, 200, 288])
# sizes = np.array([(2 * i) ** 2 for i in range(2, 5, 1)])
adjacencies = ['hexagonal', 'hexagonal_np_1', 'hexagonal_np_2']

threads = 96
T0 = 0.1
Tf = 0.8
copies = 30
N_configs = threads * 100 * 2
N_term = 10000
distribution = 'binary'

T = np.zeros(copies * 2)
T[0::2] = np.geomspace(T0, Tf, copies)
T[1::2] = T[0::2].copy()
B = np.zeros([len(adjacencies), len(sizes), copies])
fname = f'Data/Model hexagonal_graphs - {distribution}, sizes = {sizes}, T in [{T[0]}, {T[-1]}], ' \
        f'N_configs = {N_configs}, N_term = {N_term}'

for i, adjacency in enumerate(adjacencies):
    print('Adjacency', i+1, 'of', adjacencies)
    for j, size in enumerate(sizes):
        print('Size', size, 'of', sizes)
        q2_q4_c = Parallel(n_jobs=threads)(
            delayed(mc.q2_q4_evolution)
            (1 / T, sg.connectivity_matrix(size, adjacency, distribution, 17 * i), N_term)
            for i in tqdm(range(N_configs)))
        q2 = np.array([q2_q4_c[i][0] for i in range(N_configs)])
        q4 = np.array([q2_q4_c[i][1] for i in range(N_configs)])

        B[i, j, :] = 0.5 * (3 - np.mean(q4, 0) / np.mean(q2, 0) ** 2)
        np.savez(fname, sizes=sizes, adjacencies=adjacencies, T=T, B=B)


