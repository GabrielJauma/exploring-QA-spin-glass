from tqdm import tqdm
import Modules.spin_glass as sg
# import Deprecated.monte_carlo_v2 as mc
import Modules.monte_carlo as mc
import numpy as np
from joblib import Parallel, delayed

# sizes = np.array([3, 4, 5]) ** 2 * 8
sizes = np.array([3, 4]) ** 2 * 8
# sizes = np.array([3, 4])**3
adjacencies = ['hexagonal_np_1']
threads = 80
T0 = 0.1
Tf = 2
copies = 20
N_configs = threads * 10
eq_steps = int(1e3)
N_term = int(1e5 * 2)
distribution = 'gaussian_EA'

Ti = np.linspace(T0, Tf, copies)
# init_steps = int(200e3)
# avg_steps = int(200e3)
# rng = np.random.default_rng(3251)
# J = sg.connectivity_matrix(sizes[0], adjacencies[0], distribution, rng=rng)
# Ti, P = mc.optimal_temperature_distribution(Ti, J, rng, init_steps=init_steps, avg_steps=avg_steps,
#                                            accept_prob_min=0.3, accept_prob_max=0.6, plot=False)
# copies = len(Ti)
T = np.zeros(copies * 2)
T[0::2] = Ti.copy()
T[1::2] = Ti.copy()

B = np.zeros([len(adjacencies), len(sizes), copies])
fname = f'Data/Model {adjacencies}- {distribution}, sizes = {sizes}, T in [{T[0]}, {T[-1]}], ' \
        f'N_configs = {N_configs}, N_term = {N_term}, eq_steps = {eq_steps}'

seeds = np.random.SeedSequence().spawn(N_configs)
for i, adjacency in enumerate(adjacencies):
    print('Adjacency', i + 1, 'of', adjacencies)
    for j, size in enumerate(sizes):
        print('Size', size, 'of', sizes)
        q2_q4_c = Parallel(n_jobs=threads)(
            delayed(mc.equilibrate_and_average)
            (1 / T, sg.connectivity_matrix(size, adjacency, distribution, rng=np.random.default_rng(seed=seeds[i])),
             N_term, eq_steps=eq_steps, rng=np.random.default_rng(seed=seeds[i]))  # , tempering=False)
            for i in tqdm(range(N_configs)))
        q2 = np.array([q2_q4_c[i][0] for i in range(N_configs)])
        q4 = np.array([q2_q4_c[i][1] for i in range(N_configs)])

        # B[i, j, :] = 0.5 * (3 - np.mean(q4, 0) / np.mean(q2, 0) ** 2)
        B[i, j, :] = 0.5 * (3 - np.mean(q4 / q2 ** 2, 0))
        np.savez(fname, sizes=sizes, adjacencies=adjacencies, T=T, B=B)


