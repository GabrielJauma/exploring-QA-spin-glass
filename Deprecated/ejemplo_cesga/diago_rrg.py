from mpi4py import MPI
import numpy as np
import sys
from numpy.random import SeedSequence, default_rng
import rrg_library as model

n = int(sys.argv[1])  # number of nodes
wmin = np.double(sys.argv[2])  # min and max disorder
wmax = np.double(sys.argv[3])
nsteps = int(sys.argv[4])
w = np.linspace(wmin, wmax, nsteps)
nsamp = int(sys.argv[5])  # num of samples
pre_seed = int(sys.argv[6])

# Paralell definitions and files for every node:
comm = MPI.COMM_WORLD
process_index = comm.Get_rank()
number_of_processes = comm.Get_size()

# Initialize random generator:
ss = SeedSequence(pre_seed)
# Spawn off number_of_processes child SeedSequences to pass to child processes.
child_seeds = ss.spawn(number_of_processes)
streams = [default_rng(s) for s in child_seeds]
xseed = streams[process_index].integers(2 ** 32 - 1)
np.random.seed(xseed)

# number of nodes and number of task per node:
samples_per_process = int(nsamp / number_of_processes + 1)
# The file where we will write results which is node dependant:
# filename = 'n'+str(n)+'rrg_d'+str(d)+'seed'+str(xseed)+'proc'+str(process_index)+'.dat'
filename = 'eig-RRG' + '_size' + str(n) + '_seed' + str(xseed) + '.dat'

f = open(filename, 'w')
f.write("disorder, r, er, S, eS, log(psi**2), elog(psi**2),  I2, eI\n")
f.close()

#
# And finally we perform computations:
for ispr in range(0, samples_per_process):
    # this is toy computation:
    npseed = np.random.get_state()[1][0]
    xdat = model.onesample_mpi(n, w, nvec=1, wst=wst, de=None)
    f = open(filename, 'a+')
    np.savetxt(f, np.c_[w, xdat], header='samples  ' + str(ispr + 1) + '  seed ' + str(npseed))
    f.close()

quit()
