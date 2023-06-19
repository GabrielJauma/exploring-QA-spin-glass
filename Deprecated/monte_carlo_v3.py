from numba import njit
import numpy as np
import math


def custom_sparse(J):
    size = J.shape[0]
    J = (J + J.T) / 2.0
    Jlil = J.tolil()
    for i in range(J.shape[0]):
        Jlil[i, i] = 0
    connections = max(len(elem) for elem in Jlil.rows)
    Jrows = np.zeros([size, connections], dtype='int64')
    Jvals = np.zeros([size, connections])
    for i in range(size):
        Jrows[i, 0:len(Jlil.rows[i])] = Jlil.rows[i]
        Jvals[i, 0:len(Jlil.data[i])] = Jlil.data[i]
    return Jrows, Jvals


@njit(fastmath=True, boundscheck=False)
def mc_step(p, βE, E, s, dE, changes):
    for i, (βEi, pi, c) in enumerate(zip(βE, p, changes)):
        if βEi < 0 or pi <= math.exp(-βEi):
            s[i, c] = -s[i, c]
            E[i] += dE[i]


# @njit("i4[:](i4[:], f8[:], f8[:], f8[:])", boundscheck=False, fastmath=True)
# def tempering_step(ndx, p, β, E):
#     Ei, βi, i = E[0], β[0], 0
#     for j, (Ej, βj, pj) in enumerate(zip(E, β, p)):
#         if j:
#             r = (βi - βj) * (Ei - Ej)
#             if r >= 0 or pj <= math.exp(r):
#                 ndx[j - 1] = j
#                 Ej, βj, j = Ei, βi, i
#                 continue
#             else:
#                 ndx[j - 1] = i
#         Ei, βi, i = Ej, βj, j
#     ndx[-1] = i
#     return ndx

@njit("i4[:](i4[:], f8[:], f8[:], f8[:])", boundscheck=False, fastmath=True)
def tempering_step(ndx, p, β, E):
    swapped = []
    for j, (Ej, βj, pj) in enumerate(zip(E, β, p)):
        if np.any(np.array(swapped) == j):
            continue
        else:
            for i, (Ei, βi) in enumerate(zip(E, β)):
                if not np.any(np.array(swapped) == i) and i != j:
                    r = (βi - βj) * (Ei - Ej)
                    if r >= 0 or pj <= math.exp(r):
                        ndx[i], ndx[j] = ndx[j], ndx[i]
                        swapped.append(j)
                        swapped.append(i)
                        break
    return ndx


def cost(s, J):
    # E = -0.5 * np.diag(s @ J @ s.T)
    s = s.T
    E = - 0.5 * np.sum(s * (J @ s), 0)
    return E


@njit(fastmath=True, boundscheck=False)
def delta_cost(dE, s, Jrows, Jvals, change):
    for c, a in enumerate(change):
        columns = Jrows[a]  # nonzero column indices of row a, that is, of J[a,:]
        values = Jvals[a]  # corresponding values, that is, J[a,columns]
        E = 0.0
        for j, J_aj in zip(columns, values):
            E += J_aj * s[c, j]
        dE[c] = 2 * (E * s[c, a])
    return dE


@njit(fastmath=True)
def mc_loop_trajectories(β, Jrows, Jvals, s, E, steps, random_sites, random_chances, random_tempering, tempering):
    size = Jrows.shape[0]
    copies = len(β)
    dE = np.zeros(E.size, dtype=np.double)
    swap = np.arange(copies, dtype=np.int32)

    E_vs_t_loop = np.zeros((steps, copies), dtype=np.double)
    swap_vs_t_loop = np.zeros((int(steps / 10), copies), dtype=np.int32)

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        delta_cost(dE, s, Jrows, Jvals, flip_sites)
        mc_step(flip_chances, β * dE, E, s, dE, flip_sites)
        mc_step(flip_chances, β * dE, E, s, dE, flip_sites)
        if tempering and n % 10 * size == 0:# and n != 0:
            swap = tempering_step(swap, random_tempering[int(n / 10), :], β, E)
            E = E[swap]
            s = s[swap, :]
            swap_vs_t_loop[int(n / 10), :] = swap

        E_vs_t_loop[n, :] = E

    return s, E, E_vs_t_loop, swap_vs_t_loop


@njit(fastmath=True)
def mc_loop(β, Jrows, Jvals, s, E, steps, random_sites, random_chances, random_tempering, tempering):
    size = Jrows.shape[0]
    copies = len(β)
    dE = np.zeros(E.size, dtype=np.double)
    swap = np.arange(copies, dtype=np.int32)

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        delta_cost(dE, s, Jrows, Jvals, flip_sites)
        mc_step(flip_chances, β * dE, E, s, dE, flip_sites)
        if tempering and n % 10 * size == 0:# and n != 0:
            swap = tempering_step(swap, random_tempering[int(n / 10), :], β, E)
            E = E[swap]
            s = s[swap, :]

    return s, E


def mc_evolution(β, J, steps=None, start=None, eq_steps=1000, rng=None, trajectories=False, tempering=True):
    size = np.shape(J)[0]
    copies = len(β)
    Jrows, Jvals = custom_sparse(J)

    if rng is None:
        rng = np.random.default_rng()

    if start is not None:
        s = start[0]
        E = start[1]
    else:
        s = rng.integers(0, 2, (copies, size), np.int8)
        s[s == 0] = -1
        E = cost(s, J)

    if steps is None:
        steps = int(size * 100)

    if trajectories:
        E_vs_t = np.array([E])
        swap_vs_t = np.array([np.arange(copies, dtype=np.int32)])

    for _ in range(eq_steps):
        random_sites = rng.integers(0, size, (steps, copies))
        random_chances = rng.random((steps, copies))
        random_tempering = rng.random((int(steps / 10), copies))

        if trajectories:
            s, E, E_vs_t_loop, swap_vs_t_loop = mc_loop_trajectories(β, Jrows, Jvals, s, E, steps, random_sites, random_chances,
                                                  random_tempering, tempering)
            E_vs_t = np.concatenate((E_vs_t, E_vs_t_loop))
            swap_vs_t = np.concatenate((swap_vs_t, swap_vs_t_loop))
        else:
            s, E = mc_loop(β, Jrows, Jvals, s, E, steps, random_sites, random_chances, random_tempering, tempering)

    if trajectories:
        return s, E, E_vs_t, swap_vs_t
    else:
        return s, E, Jrows, Jvals


def q2_q4_thermal_av(β, Jrows, Jvals, s, E, N_term, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    size = np.shape(Jrows)[0]
    copies = len(β)
    term_steps = size
    total_steps = int(N_term * term_steps)

    # I have to divide this process in blocks because the random numbers occupy too much memory=
    if total_steps > 1e6:
        blocks = int(total_steps / 1e6)
        steps_per_block = int(total_steps / blocks)
    else:
        blocks = 1
        steps_per_block = total_steps

    q2_b = np.zeros([blocks, int(copies / 2)])
    q4_b = np.zeros([blocks, int(copies / 2)])

    for i in range(blocks):
        random_sites = rng.integers(0, size, (steps_per_block, copies))
        random_chances = rng.random((steps_per_block, copies))
        q2_b[i, :], q4_b[i, :], s, E = q2_q4_thermal_av_mc_loop(β, Jrows, Jvals, s, E,
                                                                steps_per_block / term_steps, term_steps, random_sites,
                                                                random_chances)

    return np.mean(q2_b, 0), np.mean(q4_b, 0)


@njit(fastmath=True)
def q2_q4_thermal_av_mc_loop(β, Jrows, Jvals, s, E, N_term, term_steps, random_sites, random_chances):
    size = np.shape(Jrows)[0]
    copies = len(β)
    q2 = np.zeros((int(len(β) / 2)), dtype='float64')
    q4 = np.zeros((int(len(β) / 2)), dtype='float64')
    q_ab = np.zeros((int(len(β) / 2)), dtype='float64')
    dE = np.zeros(E.size, dtype='float64')
    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        delta_cost(dE, s, Jrows, Jvals, flip_sites)
        mc_step(flip_chances, β * dE, E, s, dE, flip_sites)

        if not bool(np.remainder(n, term_steps)):
            for c in range(int(copies / 2)):
                q_ab[c] = np.sum(s[0 + 2 * c, :] * s[1 + 2 * c, :])
            q_ab /= size
            q2 += q_ab ** 2
            q4 += q_ab ** 4
    q2 /= N_term
    q4 /= N_term
    return q2, q4, s, E


def q2_q4_evolution(β, J, N_term, eq_steps=1000, rng=None):
    s, E, Jrows, Jvals = mc_evolution(β, J, steps=None, start=None, eq_steps=eq_steps, rng=rng)
    B = q2_q4_thermal_av(β, Jrows, Jvals, s, E, N_term, rng=rng)
    return B
