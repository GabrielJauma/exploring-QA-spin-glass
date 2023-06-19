from Deprecated import monte_carlo_v1 as mc
from Modules import spin_glass as sg

import numpy as np
import scipy.sparse as sp
from numba import njit

import matplotlib.pyplot as plt
import time
import timeit
from importlib import reload
import sys

sys.path.append('../')
sg = reload(sg)

plt.rcParams.update({
    "text.usetex": True})


# %%
def cost1(s, J):
    return np.diag(s @ J)  # @ s.T)


@njit("f8[:](i1[:,:], f8[:,:])", fastmath=True)
def cost2(s, J):
    # copies = np.shape(s)[0]
    # E = np.zeros(copies)
    # # for i in range(copies):
    # #     si = s[i].astype(np.float64)
    # #     E[i] = np.dot(si, np.dot(J, si.T))
    result = [[sum(a * b for a, b in zip(X_row, Y_col)) for Y_col in zip(*s)] for X_row in J]

    return result


# %%

rng = np.random.default_rng()
size = 100
copies = 100

J = 2 * (rng.random([size, size]) - 0.5)
s = rng.integers(0, 2, (copies, size), np.int8)
s[s == 0] = -1

# %%
start = time.time()
E1 = cost1(s, J)
end = time.time()
print('cost1 = ', end - start)

start = time.time()
E2 = cost2(s, J)
end = time.time()
print('cost2 = ', end - start)

start = time.time()
E2 = cost2(s, J)
end = time.time()
print('cost2 = ', end - start)

#%%
start = time.time()
s, E = mc.mc_evolution(β, J, steps=size, start=[s, E], tempering=False, trajectories=False)
end = time.time()
print(end - start)
#%%
start = time.time()
for i in range(10*size):
    s, E = mc.mc_evolution(β, J, steps=size, start=[s, E], tempering=False, trajectories=False)
end = time.time()
print(end - start)

#%%
start = time.time()
s_t, E_t = mc.mc_thermal_eq(β, J, s, E, term_steps=size, N_term=10)
end = time.time()
print(end - start)


#%% Sparse test
size = 30**2
copies = 100

J = 2 * (rng.random([size, size]) - 0.5)
s = rng.integers(0, 2, (copies, size), np.int8)
s[s == 0] = -1

L = round(size ** (1 / 2))
A1d = np.zeros([L, L]) + np.diag(np.ones(L - 1), 1) + np.diag(np.ones(L - 1), -1)
A1d[0, L - 1] = 1
A1d[L - 1, 0] = 1
I = np.eye(L, L)
A2d = np.kron(I, A1d) + np.kron(A1d, I)
A2d[A2d > 0] = 1
A = A2d

mu = 0
sigma = 1
J = np.random.default_rng().normal(mu, sigma, [size, size])

Jm = A*J
Jms = sp.csr_matrix(A * J)

change = rng.integers(0, size, copies)

#%%
print('Dense E')
# %timeit np.diag(s @ Jm @ s.T)

#%%
print('Sparse E')
# %timeit np.diag(s @ Jms @ s.T)

#%%
print('Dense dE')
# %timeit np.diag(s[:, change]) * np.diag(Jm[change, :] @ s.T)

#%%
print('Sparse dE')
# %timeit np.diag(s[:, change]) * np.diag(Jms[change, :] @ s.T)

#%%
rng = np.random.default_rng()
copies = 100

sizes = [i**3 for i in range(4,11)]

for size in sizes:

    s = rng.integers(0, 2, (copies, size), np.int8)
    s[s == 0] = -1
    change = rng.integers(0, size, copies)

    Jm =  sg.adjacency_matrix(size, '3D', 'gaussian_EA', seed=912384, sparse=False)
    Jms = sg.adjacency_matrix(size, '3D', 'gaussian_EA', seed=912384, sparse=True)

    print(size)
    print('Dense')
    # %timeit mc.delta_cost(s, Jm, change)
    print('Sparse')
    # %timeit mc.delta_cost(s, Jms, change)
    print('\n')



