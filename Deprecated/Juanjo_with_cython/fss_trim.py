from tqdm import tqdm
import Modules.spin_glass as sg
import numpy as np
from joblib import Parallel, delayed

# sizes = np.array([i ** 2 for i in range(8, 16, 4)])
# sizes = np.array([i ** 3 for i in range(2, 6, 1)])
sizes = np.array([60, 180, 540])
# sizes = np.array([(2 * i) ** 2 for i in range(2, 5, 1)])
trims = np.array([0.1, 0.3, 0.6])
threads = 96
T0 = 0.1
Tf = 1
copies = 30
N_configs = threads * 100
N_term = 1000
adjacency = 'SK'
distribution = 'gaussian_SK'

T = np.zeros(copies * 2)
T[0::2] = np.geomspace(T0, Tf, copies)
T[1::2] = T[0::2].copy()
B = np.zeros([len(trims), len(sizes), copies])
fname = f'Data/Model {adjacency}-{distribution}, sizes = {sizes}, trims = {trims}, T in [{T[0]}, {T[-1]}], ' \
        f'N_configs = {N_configs}, N_term = {N_term}'

for i, trim in enumerate(trims):
    print('Trim', trim, 'of', trims)
    for j, size in enumerate(sizes):
        print('Size', size, 'of', sizes)
        q2_q4_c = Parallel(n_jobs=threads)(
            delayed(sg.q2_q4_evolution)
            (1 / T, sg.connectivity_matrix(size, adjacency, distribution, 17 * i,  trim), N_term)
            for i in tqdm(range(N_configs)))
        q2 = np.array([q2_q4_c[i][0] for i in range(N_configs)])
        q4 = np.array([q2_q4_c[i][1] for i in range(N_configs)])

        B[i, j, :] = 0.5 * (3 - np.mean(q4, 0) / np.mean(q2, 0) ** 2)

np.savez(fname, sizes=sizes, trims=trims, T=T, B=B)
