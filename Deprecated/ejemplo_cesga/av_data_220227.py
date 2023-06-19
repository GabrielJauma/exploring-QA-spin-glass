import numpy as np
import scipy.sparse
import os, sys, petsc4py
from petsc4py import PETSc
petsc4py.init(sys.argv)
from slepc4py import SLEPc
Print = PETSc.Sys.Print
import re

def average(res):
    '''input:
    res  -----> list, res[i][0] contains data to be average, res[i][1] are the number of samples
    Output:
    xave  -----> a np.array with the 
     '''
    tsam = sum([res[i][1] for i in range(len(res))])
    xave = sum([res[i][0]*res[i][1] for i in range(len(res))])
    return xave, tsam

def average_data(filed):
    '''input:
    f  -----> path to file with data
    Output:
    A  -----> np array with the averagd data. Each line conrresponds to a disorder w value
    nsamp --> number of samples used contained in the file.'''
    f =open(filed,'r')
    lines = [line for line in f]
    ind = []
    for j in range(len(lines)):
        if lines[j].startswith("#"):
            ind.append(j)
    B = [np.loadtxt(lines[ind[i]:ind[i+1]]) for i in range(len(ind)-1) ]
    nsamp = len(B)
    A = sum(B)/nsamp
    return A, nsamp
res = []
# Path where we are going to search for files, which is the location of this program
rootdir = os.getcwd()
# Name  of file to store data
fr ='rrg_smallsizes.av'
# Scan all files that finish in .bin to search and add disorders W and sizes L values. Each line of W and L corresponds
# to a file with data 
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".dat"):
            pattern2 = "eig-RRG_size(.*?)_seed"
            siz = re.search(pattern2, file).group(1)
            A, samp = average_data(filepath)
            res.append([A, samp])
r, tsam = average(res)
#col = [1,3,5,7, 9]
#for i in col:
    #r[:, i+1] = (r[:,i]**2-r[:, i+1])/np.sqrt(tsam)
np.savetxt(fr, np.column_stack((r[:,0]/tsam, int(siz) * np.ones(r.shape[0]), r[:,1:], tsam * np.ones(r.shape[0]))),  header="disorder, Size, sum(S), sum(S**2),sum(I_2), sum(I_2**2),sum(I_inf), sum(I_inf**2),sum(log(|psi|)**2), sum(log(|psi|**2)**2), Nsamples")



