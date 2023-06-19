from numba import njit
import numpy as np
import math


@njit("i4[:](i4[:], f8[:], f8[:], f8[:])", fastmath=True)
def tempering_step(ndx, p, β, E):
    Ei, βi, i = E[0], β[0], 0
    for j, (Ej, βj, pj) in enumerate(zip(E, β, p)):
        if j:
            r = (βi - βj) * (Ei - Ej)
            # The second part of the if ensures no tempering between different replicas
            if r >= 0 or pj <= math.exp(r):  # and (i % 2 == 0 and j != i + 1) or (i % 2 != 0 and j != i - 1):
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


def cost(s, J):
    # E = -0.5 * np.diag(s @ J @ s.T)
    s = s.T
    E = - 0.5 * np.sum(s * (J @ s), 0)
    return E


@njit(fastmath=True)
def delta_cost(s, Jrows, Jvals, change):
    copies = np.shape(s)[0]
    size = np.shape(s)[1]
    dE = np.zeros(copies)

    for c, a in enumerate(change):
        columns = Jrows[a]  # nonzero column indices of row a, that is, of J[a,:]
        values = Jvals[a]  # corresponding values, that is, J[a,columns]
        J_aa = 0
        for j, J_aj in zip(columns, values):
            dE[c] += J_aj * s[c, j]
            if a == j:
                J_aa = J_aj
        dE[c] *= 2 * s[c, a]
        dE[c] += 2 * J_aa

    return dE


def custom_sparse(J):
    Jlil = J.tolil()
    size = np.shape(J)[0]
    connections = max(len(elem) for elem in Jlil.rows)
    Jrows = np.zeros([size, connections], dtype='int64')
    Jvals = np.zeros([size, connections])
    for i in range(size):
        Jrows[i, 0:len(Jlil.rows[i])] = Jlil.rows[i]
        Jvals[i, 0:len(Jlil.data[i])] = Jlil.data[i]
    return Jrows, Jvals


def mc_evolution(β, J, steps=None, start=None, eq_points=1000, rng=None):
    size = np.shape(J)[0]
    copies = len(β)
    if rng is None:
        rng = np.random.default_rng()

    # Generate the original instances that we are modifying
    if start is not None:
        s = start[0]
        E = start[1]
    else:
        s = rng.integers(0, 2, (copies, size), np.int8)
        # s = np.array([rng.integers(0, 2, size, np.int8)] * copies)  # Same initial conditions for everyone
        s[s == 0] = -1
        E = cost(s, J)

    thermal_equilibrium = False
    if steps is None:
        thermalization = False
        steps = int(size ** 2)  # change to int(size**2 * 10)
        Emin = np.nan
        eq_points_0 = eq_points
    else:
        thermalization = True

    Jrows, Jvals = custom_sparse(J)

    while not thermal_equilibrium:
        random_sites = rng.integers(0, size, (steps, copies))
        random_chances = rng.random((steps, copies))
        random_tempering = rng.random((int(steps / 10), copies))
        s, E = mc_loop(β, Jrows, Jvals, s, E, steps, random_sites, random_chances, random_tempering)

        if eq_points == 0:
            thermal_equilibrium = True
        elif Emin < np.min(E) or np.isclose(Emin, np.min(E)):
            eq_points -= 1
        else:
            Emin = np.min(E)
            eq_points = eq_points_0

    return s, E, Jrows, Jvals


@njit(fastmath=True)
def mc_loop(β, Jrows, Jvals, s, E, steps, random_sites, random_chances, random_tempering):
    size = np.shape(Jrows)[0]
    copies = len(β)

    swap = np.arange(copies, dtype=np.int32)
    for n, (flip_sites, flip_chances) in enumerate(
            zip(random_sites, random_chances)):
        news = s.copy()
        for i in range(copies):
            news[i, flip_sites[i]] = - news[i, flip_sites[i]]
        newE = E + delta_cost(s, Jrows, Jvals, flip_sites)
        mc_step(flip_chances, β * (newE - E), E, s, newE, news)
        # if n % 10 == 0:
        #     swap = tempering_step(swap, random_tempering[int(n / 10), :], β, E)
        #     E = E[swap]
        #     s = s[swap, :]

    return s, E


def q2_q4_thermal_av(β, Jrows, Jvals, s, E, N_term):
    size = np.shape(Jrows)[0]
    copies = len(β)
    term_steps = size

    rng = np.random.default_rng()
    random_sites = rng.integers(0, size, (N_term * term_steps, copies))
    random_chances = rng.random((N_term * term_steps, copies))

    return q2_q4_thermal_av_mc_loop(β, Jrows, Jvals, s, E, N_term, term_steps, random_sites, random_chances)


@njit(fastmath=True)
def q2_q4_thermal_av_mc_loop(β, Jrows, Jvals, s, E, N_term, term_steps, random_sites, random_chances):
    size = np.shape(Jrows)[0]
    copies = len(β)
    q2 = np.zeros((int(len(β) / 2)), dtype='float64')
    q4 = np.zeros((int(len(β) / 2)), dtype='float64')
    q_ab = np.zeros((int(len(β) / 2)), dtype='float64')

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        news = s.copy()
        for i, fi in enumerate(flip_sites):  # range(copies):
            news[i, fi] = - news[i, fi]
        newE = E + delta_cost(s, Jrows, Jvals, flip_sites)
        mc_step(flip_chances, β * (newE - E), E, s, newE, news)

        if not bool(np.remainder(n, term_steps)):
            for c in range(int(copies / 2)):
                q_ab[c] = np.sum(s[0 + 2 * c, :] * s[1 + 2 * c, :])
            q_ab /= size
            q2 += q_ab ** 2
            q4 += q_ab ** 4
    q2 /= N_term
    q4 /= N_term
    return q2, q4


def q2_q4_evolution(β, J, N_term):
    s, E, Jrows, Jvals = mc_evolution(β, J, steps=None, start=None)
    B = q2_q4_thermal_av(β, Jrows, Jvals, s, E, N_term)
    return B
