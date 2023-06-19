import sys

sys.path.append('../')
from tqdm import tqdm
from Modules import spin_glass as sg
import numpy as np
from joblib import Parallel, delayed

@profile
def binder_cumulant_multiple_sizes( ):
    sizes = np.array([i ** 2 for i in range(4, 9, 2)])
    T0 = 0.01
    Tf = 1
    copies = 20
    N_configs = 24 * 100
    N_term = 100
    T = np.zeros(copies * 2)
    T[0::2] = np.geomspace(T0, Tf, copies)
    T[1::2] = T[0::2].copy()
    adjacency = '2D'
    distribution = 'binary'
    B_m = []

    for size in sizes:
        print(size, 'of', sizes)
        if size > 250:
            sparse = True
        else:
            sparse = False
        q2_q4_c = Parallel(n_jobs=24)(
            delayed(sg.q2_q4_evolution)
            (1 / T, sg.connectivity_matrix(size, adjacency, distribution, 17 * i, sparse), N_term)
            for i in tqdm(range(N_configs)))
        q2 = np.array([q2_q4_c[i][0] for i in range(N_configs)])
        q4 = np.array([q2_q4_c[i][1] for i in range(N_configs)])

        B = 0.5 * (3 - np.mean(q4, 0) / np.mean(q2, 0) ** 2)
        B_m.append(B)

    fname = f'Data/Model {adjacency}-{distribution}, sizes = {sizes}, T in [{T[0]}, {T[-1]}], N_configs = {N_configs}, N_term = {N_term}'
    np.savez(fname, sizes=sizes, T=T, B_m=B_m)
    return B_m

if __name__ == '__main__':
    binder_cumulant_multiple_sizes( )
