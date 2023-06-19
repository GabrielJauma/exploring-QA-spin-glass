import sys
import os

home_dir = '/lustre/home/iff/gjauma'
store_dir = '/lustre/home/iff/gjauma'
sys.path.extend([store_dir])
sys.path.extend([home_dir])
import numpy as np
import pandas as pd
from mpi4py import MPI
from numpy.random import SeedSequence
import Modules.monte_carlo as mc
import Modules.spin_glass as sg

# In data
adjacency = str(sys.argv[1])
distribution = str(sys.argv[2])
size = int(sys.argv[3])
T0 = float(sys.argv[4])
Tf = float(sys.argv[5])
MCS_avg = int(sys.argv[6])
max_MCS = int(sys.argv[7])
N_configs = int(sys.argv[8])
pre_seed = int(sys.argv[9])
if len(sys.argv) == 11:
    add = float(sys.argv[10])
else:
    add = 0

# Parallel definitions and files for every node:
comm = MPI.COMM_WORLD
process_index = comm.Get_rank()
number_of_processes = comm.Get_size()
configs_per_process = int(N_configs / number_of_processes + 1)

# Create seed for each process
process_seed = SeedSequence(pre_seed).spawn(number_of_processes)[process_index].generate_state(1)[0]
config_seeds = SeedSequence(process_seed).spawn(configs_per_process)


# Create file dir and file names
if add == 0:
    fdir = store_dir + f'/Data/{adjacency}_{distribution},n={size},T={T0}_{Tf},' \
                       f'MCS_avg={MCS_avg},max_MCS={max_MCS},binned'
else:
    fdir = store_dir + f'/Data/{adjacency}_{distribution},n={size},T={T0}_{Tf},' \
                       f'MCS_avg={MCS_avg},max_MCS={max_MCS},add={add},binned'
try:
    os.mkdir(fdir)
except:
    pass
print(fdir)

# Read T from Model / Size / T0,Tf
if add == 0:
    dir_T_dist = home_dir + f'/Cluster/temperature_distributions/{adjacency}_{distribution},n={size},T={T0}_{Tf}.dat'
else:
    dir_T_dist = home_dir + f'/Cluster/temperature_distributions/{adjacency}_{distribution},n={size},T={T0}_{Tf},add={add}.dat'
Ti = np.loadtxt(dir_T_dist)
copies = len(Ti)
T = np.zeros(copies * 2)
T[0::2] = Ti.copy()
T[1::2] = Ti.copy()

# Save the temperature in the folder
fname_T = fdir + '/T.dat'  # The T.dat files will be used as temperatures when reading data
file_T = open(fname_T, 'w')
np.savetxt(file_T, Ti, fmt='%.12e')
file_T.close()

# Perform the actual computation
for config_index in range(configs_per_process):
    print(f'{(config_index + 1) * 100 / configs_per_process} % completed')

    rng = np.random.default_rng(config_seeds[config_index])
    J = sg.connectivity_matrix(size, adjacency, distribution, rng, add=add)

    µ_q2_vs_MCS, µ_q4_vs_MCS, σ2_q2_bin_vs_MCS, σ2_q4_bin_vs_MCS, µ_ql_vs_MCS, µ_U_vs_MCS, µ_U2_vs_MCS, \
    q_dist_vs_MCS, MCS_avg_s = mc.equilibrate_and_average_bin(1 / T, J, MCS_avg, max_MCS, rng)

    print(f'Config_seed = {config_seeds[config_index]}, MCS_avg = {MCS_avg_s[-1]}')

    for i, MCS_avg_i in enumerate(MCS_avg_s):
        fname = fdir + f'/MCS_avg={MCS_avg_i},seed={process_seed}.csv'
        df = pd.DataFrame(np.concatenate((config_seeds[config_index].generate_state(1)[0]*np.ones([1,len(µ_q2_vs_MCS[i])]),
                                          np.array([µ_q2_vs_MCS[i]]), np.array([µ_q4_vs_MCS[i]]),
                                          σ2_q2_bin_vs_MCS[i], σ2_q4_bin_vs_MCS[i],
                                          np.array([µ_ql_vs_MCS[i]]),
                                          np.array([µ_U_vs_MCS[i]]),np.array([µ_U2_vs_MCS[i]]),
                                          q_dist_vs_MCS[i])))
        df.to_csv(fname,  mode='a', index=False)

quit()