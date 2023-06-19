from numba import njit
import time
import numpy as np
import math


@njit("i4[:](i4[:], f8[:], f8[:], f8[:])", fastmath=True)
def tempering_step(ndx, p, β, E):
    Ei, βi, i = E[0], β[0], 0
    for j, (Ej, βj, pj) in enumerate(zip(E, β, p)):
        if j:
            r = (βi - βj) * (Ei - Ej)
            if r >= 0 or pj <= math.exp(r):
                ndx[j - 1] = j
                Ej, βj, j = Ei, βi, i
                continue
            else:
                ndx[j - 1] = i
        Ei, βi, i = Ej, βj, j
    ndx[-1] = i
    return ndx


@njit("void(f8[:], f8[:], f8[:], i1[:,:], f8[:], i1[:,:])", fastmath=True)
def mc_step(p, βE, E, s, newE, news):
    for i, (βEi, pi) in enumerate(zip(βE, p)):
        if βEi < 0 or pi <= math.exp(-βEi):
            s[i, :] = news[i, :]
            E[i] = newE[i]


# @njit('i8[:,:](i8[:,:],i1[:,:])', fastmath=True)
# def matmul(matrix1, matrix2):
#     A = np.zeros((np.shape(matrix1)[0], np.shape(matrix2)[1]), dtype='int64')
#     for i in range(len(matrix1)):
#         for j in range(len(matrix2[0])):
#             for k in range(len(matrix2)):
#                 A[i][j] += matrix1[i][k] * matrix2[k][j]
#     return A
#
#
# @njit('i8[:](i1[:,:],i8[:,:])', fastmath=True)
# def matmul_diag(matrix1, matrix2):
#     A = np.zeros((np.shape(matrix1)[0]), dtype='int64')
#     for i in range(len(matrix1)):
#         for k in range(len(matrix2)):
#             A[i] += matrix1[i][k] * matrix2[k][i]
#     return A
#
#
# @njit('i8[:](i8[:,:],i1[:,:])', fastmath=True)
# def matmul_diag2(matrix1, matrix2):
#     A = np.zeros((np.shape(matrix1)[0]d)
#     for i in range(len(matrix1)):
#         for k in range(len(matrix2)):
#             A[i] += matrix1[i][k] * matrix2[k][i]
#     return A


#
# def cost(s, J):
#     E = -0.5 * matmul_diag(s, matmul(J, s.T))
#     return E

# def delta_cost(s, J, change):
#     dE = 2 * np.diag(s[:, change]) * matmul_diag2(J[change, :], s.T)
#     return dE


def cost(s, J):
    E = -0.5 * np.diag(s @ J @ s.T)
    return E


# def delta_cost(s, J, change):
#     dE = 2 * np.diag(s[:, change]) * np.diag(J[change, :] @ s.T)
#     return dE


@njit('f8[:](i1[:,:], i8[:,:], i8[:])', fastmath=True)
def delta_cost(s, J, change):
    copies = np.shape(s)[0]
    size = np.shape(s)[1]
    dE = np.zeros(copies)
    for c in range(copies):
        a = change[c]
        for i in range(size):
            dE[c] += J[a, i] * s[c, i]
        dE[c] *= 2 * s[c, a]
        dE[c] += 2 * J[a, a]
    return dE


def mc_evolution(β, J, steps=None, start=None, tempering=False, trajectories=False):
    """Solve a QUBO problem by classical annealing.

    Input:
    cost =          Energy function to minimize/maximize
    size =          Size of the binary vector that is an argument to the cost function
    β =             Vector of real nonnegative values of the inverse temperature,
                    to be used one after for the parallel tempering.
    steps =         Number of MC steps to use. If steps == None run the Monte Carlo until thermal equilibrium.
    trajectories =  If True, return a matrix of energies explored.

    Output:
    s =       list/vector of integers that minimize the problem.
    E =       minimum energy that was found.
    theE =    matrix of energies explored.

    """
    size = np.shape(J)[0]
    copies = len(β)
    rng = np.random.default_rng()

    # Generate the original instances that we are modifying
    if start is not None:
        s = start[0]
        E = start[1]
    else:
        # s = rng.integers(0, 2, (copies, size), np.int8)
        s = np.array([rng.integers(0, 2, size, np.int8)] * copies)  # Same initial conditions for everyone
        s[s == 0] = -1
        E = cost(s, J)

    thermal_equilibrium = False
    if steps is None:
        thermalization = False
        steps = size * 5
        Emin = np.nan
        eq_points_0 = 10
        eq_points = eq_points_0
    else:
        thermalization = True

    if trajectories:
        theE = [E.copy()]

    N_mc = 0
    while not thermal_equilibrium and N_mc < size * 10000:
        # Monte Carlo algorithm
        swap = np.arange(copies, dtype=np.int32)
        for n, change in enumerate(rng.integers(0, size, (steps, copies))):
            N_mc = N_mc + 1
            #
            # Find which bit to change on each instance and flip it
            news = s.copy()
            news[np.arange(copies, dtype=np.int32), change] = - news[np.arange(copies, dtype=np.int32), change]
            newE = E + delta_cost(s, J, change)
            #
            # Accept the change with a probability determined by the
            # instance's temperature and change of energy
            mc_step(rng.random(copies), β * (newE - E), E, s, newE, news)
            #
            # If tempering, randomly swap the instances' temperatures
            # based on the tempering probability formula
            if tempering:
                swap = tempering_step(swap, rng.random(β.size), β, E)
                E = E[swap]
                s = s[swap, :]
            if trajectories:
                theE = np.concatenate((theE, [E]), axis=0)

        if thermalization:
            thermal_equilibrium = True
            continue

        if eq_points == 0:
            thermal_equilibrium = True
        elif Emin < np.min(E) or np.isclose(Emin, np.min(E)):
            eq_points -= 1
        else:
            Emin = np.min(E)
            eq_points = eq_points_0

    if trajectories:
        return s, E, theE
    else:
        return s, E


def mc_thermal_eq(β, J, s, E, N_term):
    size = np.shape(J)[0]
    copies = len(β)
    term_steps = size

    s_t = np.zeros([int(N_term), copies, size])
    E_t = np.zeros([int(N_term), copies])
    t = 0

    rng = np.random.default_rng()
    # Monte Carlo algorithm
    swap = np.arange(copies, dtype=np.int32)
    for n, change in enumerate(rng.integers(0, size, (N_term * term_steps, copies))):
        news = s.copy()
        news[np.arange(copies, dtype=np.int32), change] = - news[np.arange(copies, dtype=np.int32), change]
        newE = E + delta_cost(s, J, change)

        mc_step(rng.random(copies), β * (newE - E), E, s, newE, news)

        if not bool(np.remainder(n, term_steps)):
            s_t[t, :, :] = s.copy()
            E_t[t, :] = E.copy()
            t = t + 1

    return s_t, E_t


# def binder_cumulant_thermal_av(β, J, s, E, N_term):
#     size = np.shape(J)[0]
#     copies = len(β)
#     term_steps = size
#
#     q2 = np.zeros((int(len(β) / 2)), dtype='float64')
#     q4 = np.zeros((int(len(β) / 2)), dtype='float64')
#
#     rng = np.random.default_rng()
#     # Monte Carlo algorithm
#     for n, change in enumerate(rng.integers(0, size, (N_term * term_steps, copies))):
#         news = s.copy()
#         news[np.arange(copies, dtype=np.int32), change] = - news[np.arange(copies, dtype=np.int32), change]
#         newE = E + delta_cost(s, J, change)
#
#         mc_step(rng.random(copies), β * (newE - E), E, s, newE, news)
#
#         if not bool(np.remainder(n, term_steps)):
#             q_ab = np.mean(s[0::2, :] * s[1::2, :], 1)
#             q2 += q_ab ** 2
#             q4 += q_ab ** 4
#     q2 /= N_term
#     q4 /= N_term
#     return 0.5 * (3 - q4 / (q2 ** 2))
monte_carlo.py


def binder_cumulant_thermal_av(β, J, s, E, N_term):
    size = np.shape(J)[0]
    copies = len(β)
    term_steps = size

    rng = np.random.default_rng()
    random_sites = rng.integers(0, size, (N_term * term_steps, copies))
    random_chances = rng.random((N_term * term_steps, copies))

    return binder_cumulant_thermal_av_mc_loop(β, J, s, E, N_term, term_steps, random_sites, random_chances)


@njit('f8[:](f8[:], i8[:,:], i1[:,:], f8[:], i8, i8, i8[:,:], f8[:,:] )', fastmath=True)
def binder_cumulant_thermal_av_mc_loop(β, J, s, E, N_term, term_steps, random_sites, random_chances):
    size = np.shape(J)[0]
    copies = len(β)
    q2 = np.zeros((int(len(β) / 2)), dtype='float64')
    q4 = np.zeros((int(len(β) / 2)), dtype='float64')
    q_ab = np.zeros((int(len(β) / 2)), dtype='float64')

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        news = s.copy()
        for i in range(copies):
            news[i, flip_sites[i]] = - news[i, flip_sites[i]]
        newE = E + delta_cost(s, J, flip_sites)
        mc_step(flip_chances, β * (newE - E), E, s, newE, news)

        if not bool(np.remainder(n, term_steps)):
            for c in range(int(copies / 2)):
                q_ab[c] = np.sum(s[0 + 2 * c, :] * s[1 + 2 * c, :])
            q_ab /= size
            q2 += q_ab ** 2
            q4 += q_ab ** 4
    q2 /= N_term
    q4 /= N_term
    return 0.5 * (3 - q4 / (q2 ** 2))
