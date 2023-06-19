from tqdm import tqdm
import Modules.spin_glass as sg
import Modules.monte_carlo as mc
import numpy as np
from joblib import Parallel, delayed

# sizes = np.array([i ** 2 for i in range(8, 16, 4)])
# sizes = np.array([i ** 3 for i in range(2, 6, 1)])
# sizes = np.array([60, 180, 540])
sizes = np.array([50, 100, 150])  # , 250, 400])
# sizes = np.array([(2 * i) ** 2 for i in range(2, 5, 1)])
adjacencies = ['random_regular_3', 'random_regular_4'] #, 'random_regular_5']
# adjacencies = ['random_regular_3']
threads = 96
T0 = 0.2
Tf = 1.5
copies = 50
N_configs = threads * 100
N_term = 10000
distribution = 'gaussian_EA'

T = np.zeros(copies * 2)
T[0::2] = np.geomspace(T0, Tf, copies)
T[1::2] = T[0::2].copy()
B = np.zeros([len(adjacencies), len(sizes), copies])
fname = f'Data/Model Random_regular_graphs - {distribution}, sizes = {sizes}, T in [{T[0]}, {T[-1]}], ' \
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


