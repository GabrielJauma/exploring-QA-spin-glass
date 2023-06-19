import sys
import os

home_dir = '/home/csic/qia/gjg'
store_dir = '/mnt/netapp1/Store_CSIC/home/csic/qia/gjg'
sys.path.extend([store_dir])
sys.path.extend([home_dir])
from mpi4py import MPI
import numpy as np
from numpy.random import SeedSequence, default_rng
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

# From Model / Size / T0,Tf read T
if add == 0:
    dir_T_dist = home_dir + f'/Cluster/temperature_distributions/{adjacency}_{distribution},n={size},T={T0}_{Tf}.dat'
else:
    dir_T_dist = home_dir + f'/Cluster/temperature_distributions/{adjacency}_{distribution},n={size},T={T0}_{Tf},add={add}.dat'

try:
    Ti = np.loadtxt(dir_T_dist)
except:
    Ti = np.linspace(T0, Tf, 30)

copies = len(Ti)
T = np.zeros(copies * 2)
T[0::2] = Ti.copy()
T[1::2] = Ti.copy()

# Create file dir and file names
if add == 0:
    fdir = store_dir + f'/Data/{adjacency}_{distribution},n={size},T={T0}_{Tf},' \
                       f'MCS_avg={MCS_avg},max_MCS={max_MCS}'
else:
    fdir = store_dir + f'/Data/{adjacency}_{distribution},n={size},T={T0}_{Tf},' \
                       f'MCS_avg={MCS_avg},max_MCS={max_MCS},add={add}'

try:
    os.mkdir(fdir)
except:
    pass
print(fdir)

# Save temperature in a file just in case
try:
    fname_T = fdir + '/T.dat'  # The T.dat files will be used as temperatures when reading data
    file_T = open(fname_T, 'x')
except:
    pass

index_fname_T = 0
success_fname_T = False
while success_fname_T == False:
    try:
        fname_T = fdir + f'/T_{index_fname_T}.dat'  # The T_index.dat files will be used to avoid overwriting info
        file_T = open(fname_T, 'x')
        success_fname_T = True
    except:
        index_fname_T += 1

np.savetxt(file_T, Ti, fmt='%.7e')
file_T.close()

# Perform the actual computation
for config_index in range(configs_per_process):
    print(f'{(config_index + 1) * 100 / configs_per_process} % completed')
    rng = np.random.default_rng(config_seeds[config_index])
    J = sg.connectivity_matrix(size, adjacency, distribution, rng, add=add)
    µ_q2_s, µ_q4_s, σ2_q2_s, σ2_q4_s, ql_s, U_s, U2_s, MCS_avg_s = mc.equilibrate_and_average(1 / T, J, MCS_avg,
                                                                                              max_MCS, rng)
    print(f'Config_seed = {config_seeds[config_index]}, MCS_avg = {MCS_avg_s[-1]}')

    for i, MCS_avg_i in enumerate(MCS_avg_s):
        fname = fdir + f'/MCS_avg={MCS_avg_i},seed={process_seed}.dat'
        file = open(fname, 'a+')
        np.savetxt(file, np.array([config_seeds[config_index].generate_state(1)[0]]), fmt='%i')
        np.savetxt(file, (µ_q2_s[i], µ_q4_s[i], σ2_q2_s[i], σ2_q4_s[i], ql_s[i], U_s[i], U2_s[i]), fmt='%.7e')
        file.close()

quit()
