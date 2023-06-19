import Modules.monte_carlo as mc
import numpy as np
from numba import njit, prange
import numba as nb
# %%
def all_bit_strings(size):
    """Return a matrix of shape (2**N, N) of all bit strings that
    can be constructed using 'N' bits. Each row is a different
    configuration, corresponding to the integers 0, 1, 2 up to (2**N)-1"""
    confs = np.arange(2 ** size, dtype=np.int32)
    return np.array([(confs >> i) & 1 for i in range(size)], dtype=int) * 2 - 1

def min_energy(J):
    size = len(J[0])
    ss = all_bit_strings(size).T
    return mc.cost(J, ss).min()

@njit(fastmath=True, boundscheck=False, parallel=True)
def numba_min_energy_parallel(J):
    size = len(J[0])
    N = 2 ** size
    threads = nb.np.ufunc.parallel.get_num_threads()
    N_thread = N / threads
    E_mins = np.zeros(threads)
    for thread in prange(threads):
        s = np.zeros(size)
        for conf in range(thread * N_thread, (thread+1)*N_thread):
            for i in range(size):
                s[i] = ((conf >> i) & 1) * 2 - 1
            E = 0
            for i in range(size):
                for j in range(size):
                    E = E + s[i] * J[i, j] * s[j]
            E *= -0.5
            if E < E_mins[thread]:
                E_mins[thread] = E
    return E_mins.min()

@njit(fastmath=True, boundscheck=False)
def numba_min_energy(J):
    size = len(J[0])
    N = 2 ** size
    s = np.zeros(size)
    E_min = 0
    for conf in range(N):
        for i in range(size):
            s[i] = ((conf >> i) & 1) * 2 - 1
        E = 0
        for i in range(size):
            for j in range(size):
                E = E + s[i] * J[i, j] * s[j]
        E *= -0.5
        if E < E_min:
            E_min = E
    return E_min

def partition_function(T, size, J):
    ss = all_bit_strings(size).astype('int').T
    Es = mc.cost(J, ss)
    Z_T = [np.sum(np.exp(-Es / T_z)) for T_z in T]
    return Z_T


@njit()
def steps_to_min(E_vs_t,E_min):
    steps = E_vs_t.shape[0]
    copies = E_vs_t.shape[1]
    MC_step_to_min = steps * np.ones(copies, 'int32')
    for b in range(copies):
        for i in range(steps):
            if np.abs(E_vs_t[i, b] - E_min)<1e-10:
                MC_step_to_min[b] = i
                break
    return MC_step_to_min

@njit()
def p_to_GS_vs_steps(MC_step_to_min_vs_configs, steps):
    N_configs = MC_step_to_min_vs_configs.shape[0]
    copies = MC_step_to_min_vs_configs.shape[1]
    P_to_have_reached_GS_vs_MC_step = np.ones((copies, steps))
    for b in range(copies):
        for i in range(steps):
            P_to_have_reached_GS_vs_MC_step[b, i] = np.sum( MC_step_to_min_vs_configs[:,b] < i ) / N_configs
            if P_to_have_reached_GS_vs_MC_step[b, i] == 1:
                break
    return P_to_have_reached_GS_vs_MC_step

@njit()
def p_to_GS_vs_steps_no_T(steps_to_GS_vs_configs, steps):
    N_configs = steps_to_GS_vs_configs.shape[0]
    P_to_have_reached_GS_vs_MC_step = np.ones(steps)
    for i in range(steps):
        P_to_have_reached_GS_vs_MC_step[i] = np.sum( steps_to_GS_vs_configs < i ) / N_configs
        if P_to_have_reached_GS_vs_MC_step[i] == 1:
            break
    return P_to_have_reached_GS_vs_MC_step

