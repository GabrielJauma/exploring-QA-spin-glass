from tqdm import tqdm
import Modules.spin_glass as sg
import Modules.monte_carlo as mc
import numpy as np
from joblib import Parallel, delayed
import time
import os

from importlib import reload

sg = reload(sg)
mc = reload(mc)

adjacencies = ['2D']
add = 0
sizes_adj = [np.array([10 ** 2])]
distribution = 'gaussian_EA'
threads = 8
N_configs = threads * 1
T0 = 0.01
Tf = 3.0
MCS_eq = 10001
MCS_avg = 10001
max_MCS = int(1e6)
error = None

# %% Simulations
t = time.perf_counter()
for adjacency, sizes in zip(adjacencies, sizes_adj):
    print(f'Adjacency {adjacency} of {adjacencies}')
    for size in sizes:
        print('Size', size, 'of', sizes)
        copies = 30
        Ti = np.linspace(T0, Tf, copies)

        # rng = np.random.default_rng(12)
        # J = sg.connectivity_matrix(size, adjacency, distribution, rng=rng)
        # Ti, _ = mc.optimal_temperature_distribution(T0, Tf, J, rng, init_steps=int(1e6), avg_steps=int(1e6),
        #                                             accept_prob_min=0.3, accept_prob_max=0.6, plot=False)
        # copies = len(Ti)

        T = np.zeros(copies * 2)
        T[0::2] = Ti.copy()
        T[1::2] = Ti.copy()

        pre_seed = int((time.time() % 1) * 1e8)
        seeds = np.random.SeedSequence(pre_seed).spawn(N_configs)
        # Create file dir and file names
        if add == 0:
            fdir = f'Data/{adjacency}_{distribution},n={size},T={T0}_{Tf},' \
                  f'MCS_eq={MCS_eq},MCS_avg={MCS_avg},max_MCS={max_MCS},error={error}'
        else:
            fdir = f'Data/{adjacency}_{distribution},n={size},T={T0}_{Tf},' \
                   f'MCS_eq={MCS_eq},MCS_avg={MCS_avg},max_MCS={max_MCS},error={error},add={add}'
        fname = fdir + '/thermal_means_and_vars,seed=' + str(pre_seed) + '.dat'
        try:
            os.mkdir(fdir)
        except:
            pass
        fname_T = fdir + '/T.dat'
        file_T = open(fname_T, 'w')
        np.savetxt(file_T, Ti, fmt='%.7e')
        file_T.close()

        q2_q4_c = Parallel(n_jobs=threads)(
            delayed(mc.equilibrate_and_average)
            (1 / T,
             sg.connectivity_matrix(size, adjacency, distribution, rng=np.random.default_rng(seed=seeds[i]), add=add),
             MCS_eq, MCS_avg, max_MCS, error, rng=np.random.default_rng(seed=seeds[i]))
            for i in tqdm(range(N_configs)))

        for i in range(N_configs):
            µ_q2 = q2_q4_c[i][0]
            µ_q4 = q2_q4_c[i][1]
            σ2_q2 = q2_q4_c[i][2]
            σ2_q4 = q2_q4_c[i][3]
            file = open(fname, 'a+')
            np.savetxt(file, (µ_q2, µ_q4, σ2_q2, σ2_q4), fmt='%.7e')
            file.close()

print('Elapsed time', time.perf_counter() - t)
