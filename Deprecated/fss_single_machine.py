from tqdm import tqdm
import Modules.spin_glass as sg
import Modules.monte_carlo as mc
import numpy as np
from joblib import Parallel, delayed
import time
import os
import matplotlib.pyplot as plt

from importlib import reload

sg = reload(sg)
mc = reload(mc)

# %% Parameters
adjacencies = ['random_regular_7']
sizes_adj = [[20]]
distribution = 'gaussian_EA'
size = 20
adjacency = 'random_regular_7'
add = 0
threads = 8
N_configs = threads * 2
T0 = 0.1
Tf = 3.0
MCS_avg = 2 ** 10
max_MCS = 2 ** 14
copies = 30

pre_seed = 218943322
seeds = np.random.SeedSequence(pre_seed).spawn(N_configs)

Ti = np.linspace(T0, Tf, copies)
T = np.zeros(copies * 2)
T[0::2] = Ti.copy()
T[1::2] = Ti.copy()


# %% Simulations with multiple cores
t = time.perf_counter()
for adjacency, sizes in zip(adjacencies, sizes_adj):
    print(f'Adjacency {adjacency} of {adjacencies}')
    for size in sizes:

        print('Size', size, 'of', sizes)
        # # Create file dir and file names
        # if add == 0:
        #     fdir = f'Data/{adjacency}_{distribution},n={size},T={T0}_{Tf},' \
        #                        f'MCS_avg={MCS_avg},max_MCS={max_MCS}'
        # else:
        #     fdir = f'Data/{adjacency}_{distribution},n={size},T={T0}_{Tf},' \
        #                        f'MCS_avg={MCS_avg},max_MCS={max_MCS},add={add}'
        # try:
        #     os.mkdir(fdir)
        # except:
        #     pass
        # # Save temperature in a file just in case
        # try:
        #     fname_T = fdir + '/T.dat'  # The T.dat files will be used as temperatures when reading data
        #     file_T = open(fname_T, 'x')
        # except:
        #     pass
        #
        # index_fname_T = 0
        # success_fname_T = False
        # while success_fname_T == False:
        #     try:
        #         fname_T = fdir + f'/T_{index_fname_T}.dat'  # The T_index.dat files will be used to avoid overwriting info
        #         file_T = open(fname_T, 'x')
        #         success_fname_T = True
        #     except:
        #         index_fname_T += 1
        #
        # np.savetxt(file_T, Ti, fmt='%.7e')
        # file_T.close()

        data = Parallel(n_jobs=threads)(
            delayed(mc.equilibrate_and_average_bin)
            (1 / T,
             sg.connectivity_matrix(size, adjacency, distribution, rng=np.random.default_rng(seed=seeds[i]), add=add),
             MCS_avg, max_MCS, rng=np.random.default_rng(seed=seeds[i]))
            for i in tqdm(range(N_configs)))

print('Elapsed time', time.perf_counter() - t)
