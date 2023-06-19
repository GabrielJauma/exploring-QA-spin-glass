from numba import njit
import numpy as np
import scipy.sparse as sp
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


def cost(s, J):
    s = s.T
    return - 0.5 * np.sum(s * (J @ s), 0)


# @njit(fastmath=True, boundscheck=False)
# def delta_cost(Jrv, s, dE, flip_sites):
#     for c, a in enumerate(flip_sites):
#         E = 0.0
#         for j, J_aj in zip(Jrv[0][a], Jrv[1][a]):
#             E += J_aj * s[c, j]
#         dE[c] = 2 * (E * s[c, a])


# @njit(fastmath=True, boundscheck=False)
# def mc_step(beta, s, E, dE, flip_chances, flip_sites):
#     for i, (beta_dEi, pi, c) in enumerate(zip(beta * dE, flip_chances, flip_sites)):
#         if beta_dEi < 0 or pi <= math.exp(-beta_dEi):
#             s[i, c] = -s[i, c]
#             E[i] += dE[i]


@njit(fastmath=True, boundscheck=False)
def delta_cost(Jrows, Jvals, s, dE, flip_sites):
    for c, a in enumerate(flip_sites):
        columns = Jrows[a]  # nonzero column indices of row a, that is, of J[a,:]
        values = Jvals[a]  # corresponding values, that is, J[a,columns]
        E = 0.0
        for j, J_aj in zip(columns, values):
            E += J_aj * s[c, j]
        dE[c] = 2 * (E * s[c, a])


@njit(fastmath=True, boundscheck=False)
def mc_step(p, βdE, E, s, dE, changes):
    for i, (βdEi, pi, c) in enumerate(zip(βdE, p, changes)):
        if βdEi < 0 or pi <= math.exp(-βdEi):
            s[i, c] = -s[i, c]
            E[i] += dE[i]


@njit("(i4[:], f8[:], f8[:], f8, i4)", boundscheck=False, fastmath=True)
def maybe_temper(ndx, beta, E, pj, j):
    # We compare the states with temperatures betai < betaj
    ndxi = ndx[j - 1]
    betai, Ei = beta[ndxi], E[ndxi]
    ndxj = ndx[j]
    betaj, Ej = beta[ndxj], E[ndxj]
    # According to the temperature probability, we may
    # exchange the configurations associated to those
    # temperatures
    r = (betai - betaj) * (Ei - Ej)
    if r >= 0 or pj <= math.exp(r):
        ndx[j - 1], ndx[j] = ndxj, ndxi
        beta[ndxi], beta[ndxj] = betaj, betai


@njit("(i4[:], f8[:], f8[:], f8[:])", boundscheck=False, fastmath=True)
def tempering_step(ndx, p, beta, E):
    """Possibly beta_order temperatures between states.

    We implement a tempering phase in which we run over all temperatures
    from hot (small beta) to cold (large beta), investigating whether we
    beta_order states. In practice `ndx` is the order of the temperatures,
    meaning that state beta[ndx[i]] <= beta[ndx[i+1]]. We run over the
    temperatures in order, computing the probabilities that two configurations
    are exchanged."""
    n = beta.size
    for j in range(1, n, 2):
        maybe_temper(ndx, beta, E, p[j], j)
    for j in range(2, n, 2):
        maybe_temper(ndx, beta, E, p[j], j)


# Tempering alternatives

# @njit("(i4[:], f8[:], f8[:], f8[:])", boundscheck=False, fastmath=True)
# def tempering_step(ndx, p, beta, E):
#     for j in range(1, len(E)):
#         # We compare the states with temperatures betai < betaj
#         ndxi = ndx[j-1]
#         betai, Ei = beta[ndxi], E[ndxi]
#         ndxj = ndx[j]
#         betaj, Ej = beta[ndxj], E[ndxj]
#         # According to the temperature probability, we may
#         # exchange the configurations associated to those
#         # temperatures
#         r = (betai - betaj) * (Ei - Ej)
#         if r >= 0 or p[j] <= math.exp(r):
#             ndx[j-1], ndx[j] = ndxj, ndxi
#             beta[ndxi], beta[ndxj] = betaj, betai


# @njit("(i4[:], f8[:], f8[:], f8[:], i4[:])", boundscheck=False, fastmath=True)
# def tempering_step_prob_accept(ndx, p, beta, E, prob_accept):
#     for j in range(1, len(E)):
#         # We compare the states with temperatures betai < betaj
#         ndxi = ndx[j-1]
#         betai, Ei = beta[ndxi], E[ndxi]
#         ndxj = ndx[j]
#         betaj, Ej = beta[ndxj], E[ndxj]
#         # According to the temperature probability, we may
#         # exchange the configurations associated to those
#         # temperatures
#         r = (betai - betaj) * (Ei - Ej)
#         if r >= 0 or p[j] <= math.exp(r):
#             ndx[j-1], ndx[j] = ndxj, ndxi
#             beta[ndxi], beta[ndxj] = betaj, betai
#             prob_accept[j - 1] += 1
#             prob_accept[j] += 1


# @njit("(i4[:], f8[:], f8[:], f8[:])", boundscheck=False, fastmath=True)
# def tempering_step(ndx, p, beta, E):
#     swapped = []
#     for j in range(len(E)):
#         if np.any(np.array(swapped) == j):
#             continue
#         else:
#             for i in range(j + 1, len(E)):
#                 if not np.any(np.array(swapped) == i) and i != j:
#                     r = (beta[i] - beta[j]) * (E[i] - E[j])
#                     if r >= 0 or p[j] <= math.exp(r):
#                         ndx[i], ndx[j] = ndx[j], ndx[i]
#                         beta[i], beta[j] = beta[j], beta[i]
#                         swapped.append(i)
#                         swapped.append(j)
#                         break


@njit(boundscheck=False, fastmath=True)
def mc_loop(beta, Jrows, Jvals, s, E, steps_until_temp, random_sites, random_chances, random_tempering, tempering):
    dE = np.zeros(E.size, dtype=np.double)
    beta_order = np.argsort(beta).astype(np.int32)

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        delta_cost(Jrows, Jvals, s, dE, flip_sites)
        mc_step(flip_chances, beta * dE, E, s, dE, flip_sites)
        if n % steps_until_temp == 0 and tempering:
            tempering_step(beta_order, random_tempering[int(n / steps_until_temp), :], beta, E)


def mc_evolution(beta, J, steps=None, start=None, steps_until_temp=10, rng=np.random.default_rng(), tempering=True):
    size = J.shape[0]
    copies = beta.size
    Jrows, Jvals = custom_sparse(J)

    if start is None:
        s = rng.integers(0, 2, (copies, size), np.int8) * 2 - 1
        E = cost(s, J)
    else:
        s = start[0]
        E = start[1]

    if steps is None:
        steps = int(size * 1e5)

    max_array_len = 4e8
    if steps * copies > max_array_len:  # Hardcoded value to ensure that the random arrays (random_sites, ...) fit in Paulas PC.
        blocks = int(steps * copies / max_array_len)
        steps = int(steps / blocks)
    else:
        blocks = 1

    for _ in range(blocks):
        random_sites = rng.integers(0, size, (steps, copies), dtype='int32')
        random_chances = rng.random((steps, copies))
        random_tempering = rng.random((int(steps / steps_until_temp), copies))

        mc_loop(beta, Jrows, Jvals, s, E, steps_until_temp, random_sites, random_chances, random_tempering, tempering)

        del random_sites, random_chances, random_tempering
    return s, E, Jrows, Jvals


@njit(boundscheck=False, fastmath=True)
def q2_q4_thermal_av_mc_loop(beta, Jrows, Jvals, s, E, steps_until_term, random_sites, random_chances):
    if beta[0] != beta[1]:
        'WARNING: beta should be a 2N array with N identical values'

    size = s.shape[1]
    copies = beta.size
    steps = random_sites.shape[0]
    N_term_block = steps / steps_until_term

    q2 = np.zeros(copies // 2)
    q4 = np.zeros(copies // 2)
    q_ab = np.zeros(copies // 2)
    dE = np.zeros(copies)

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        delta_cost(Jrows, Jvals, s, dE, flip_sites)
        mc_step(flip_chances, beta * dE, E, s, dE, flip_sites)

        if n % steps_until_term == 0:
            for c in range(copies // 2):
                for i in range(size):
                    q_ab[c] += s[0 + 2 * c, i] * s[1 + 2 * c, i]
            q_ab /= size
            q2 += q_ab ** 2
            q4 += q_ab ** 4

    q2 /= N_term_block
    q4 /= N_term_block
    return q2, q4


def q2_q4_thermal_av(beta, Jrows, Jvals, s, E, N_term, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    size = s.shape[1]
    copies = beta.size
    steps_until_term = size
    steps = int(steps_until_term * N_term)

    max_array_len = 4e8
    if steps * copies > max_array_len:  # Hardcoded value to ensure that the random arrays (random_sites, ...) fit in Paulas PC.
        blocks = int(steps * copies / max_array_len)
        steps = int(steps / blocks)
    else:
        blocks = 1

    q2_b = np.zeros([blocks, int(copies / 2)])
    q4_b = np.zeros([blocks, int(copies / 2)])

    for i in range(blocks):
        random_sites = rng.integers(0, size, (steps, copies), dtype='int32')
        random_chances = rng.random((steps, copies))
        q2_b[i, :], q4_b[i, :] = q2_q4_thermal_av_mc_loop(beta, Jrows, Jvals, s, E, steps_until_term, random_sites, random_chances)
        del random_sites, random_chances

    return np.mean(q2_b, 0), np.mean(q4_b, 0)


def q2_q4_evolution(beta, J, N_term, steps_until_temp=10, rng=None, tempering=True):
    s, E, Jrows, Jvals = mc_evolution(beta, J, steps=None, start=None, steps_until_temp=steps_until_temp, rng=rng,
                             tempering=tempering)
    beta_order = np.argsort(beta)[::-1].astype(np.int32)
    beta, s, E = beta[beta_order], s[beta_order, :], E[beta_order]
    B = q2_q4_thermal_av(beta, Jrows, Jvals, s, E, N_term, rng=rng)
    return B


# %% Test functions

@njit("(i4[:], f8[:], f8[:], f8, i4, i4[:])", boundscheck=False, fastmath=True)
def maybe_temper_prob_accept(ndx, beta, E, pj, j, prob_accept):
    # We compare the states with temperatures betai < betaj
    ndxi = ndx[j - 1]
    betai, Ei = beta[ndxi], E[ndxi]
    ndxj = ndx[j]
    betaj, Ej = beta[ndxj], E[ndxj]
    # According to the temperature probability, we may
    # exchange the configurations associated to those
    # temperatures
    r = (betai - betaj) * (Ei - Ej)
    if r >= 0 or pj <= math.exp(r):
        ndx[j - 1], ndx[j] = ndxj, ndxi
        beta[ndxi], beta[ndxj] = betaj, betai
        # prob_accept[j - 1] += 1
        prob_accept[j] += 1


@njit("(i4[:], f8[:], f8[:], f8[:], i4[:])", boundscheck=False, fastmath=True)
def tempering_step_prob_accept(ndx, p, beta, E, prob_accept):
    """Possibly beta_order temperatures between states.
    We implement a tempering phase in which we run over all temperatures
    from hot (small beta) to cold (large beta), investigating whether we
    beta_order states. In practice `ndx` is the order of the temperatures,
    meaning that state beta[ndx[i]] <= beta[ndx[i+1]]. We run over the
    temperatures in order, computing the probabilities that two configurations
    are exchanged."""
    n = beta.size
    for j in range(1, n, 2):
        maybe_temper_prob_accept(ndx, beta, E, p[j], j, prob_accept)
    for j in range(2, n, 2):
        maybe_temper_prob_accept(ndx, beta, E, p[j], j, prob_accept)


@njit(fastmath=True)
def mc_loop_trajectories(beta, Jrows, Jvals, s, E, steps, steps_until_temp, random_sites, random_chances,
                         random_tempering, tempering):
    copies = beta.size
    dE = np.zeros(E.size, dtype=np.double)
    beta_order = np.argsort(beta).astype(np.int32)

    E_vs_t_loop = np.zeros((steps, copies), dtype=np.double)
    T_vs_t_loop = np.zeros((int(steps / steps_until_temp), copies), dtype=np.double)

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        delta_cost(Jrows, Jvals, s, dE, flip_sites)
        mc_step(flip_chances, beta * dE, E, s, dE, flip_sites)
        if tempering and n % steps_until_temp == 0:
            tempering_step(beta_order, random_tempering[int(n / steps_until_temp), :], beta, E)
            T_vs_t_loop[int(n / steps_until_temp), :] = 1 / beta

        E_vs_t_loop[n, :] = E

    return E_vs_t_loop, T_vs_t_loop


@njit(fastmath=True)
def mc_loop_prob_accept(beta, Jrows, Jvals, s, E, steps_until_temp, random_sites, random_chances,
                        random_tempering, tempering):
    copies = beta.size
    dE = np.zeros(E.size, dtype=np.double)
    beta_order = np.argsort(beta).astype(np.int32)
    prob_accept = np.zeros(copies).astype(np.int32)

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        delta_cost(Jrows, Jvals, s, dE, flip_sites)
        mc_step(flip_chances, beta * dE, E, s, dE, flip_sites)
        if n > tempering and n % steps_until_temp == 0:
            tempering_step_prob_accept(beta_order, random_tempering[int(n / steps_until_temp), :], beta, E,
                                       prob_accept)

    return prob_accept / ((n - tempering) / steps_until_temp)


def mc_evolution_tests(beta, J, steps=None, start=None, steps_until_temp=10, rng=np.random.default_rng(),
                       tempering=True, trajectories=False, tempering_probabilities=False):
    """
    If 'tempering' is (True / False), (use / don't use) the parallel tempering algorithm.
    If 'tempering' is int then the tempering algorithm starts after 'tempering' MC steps
    """

    size = np.shape(J)[0]
    copies = beta.size
    Jrows, Jvals = custom_sparse(J)

    if start is None:
        s = rng.integers(0, 2, (copies, size), np.int8) * 2 - 1
        E = cost(s, J)
    else:
        s = start[0]
        E = start[1]

    if steps is None:
        steps = int(size * 1e5)

    max_array_len = 1e10
    if steps * copies > max_array_len:  # Hardcoded value to ensure that the random arrays (random_sites, ...) fit in Paulas PC.
        blocks = int(steps * copies / max_array_len)
        steps = int(steps / blocks)
    else:
        blocks = 1

    if trajectories:
        E_vs_t = np.array([E])
        T_vs_t = np.array([1 / beta])

    if tempering_probabilities:
        prob_accept = np.zeros(copies)

    for _ in range(blocks):
        random_sites = rng.integers(0, size, (steps, copies), dtype='int32')
        random_chances = rng.random((steps, copies))
        random_tempering = rng.random((int(steps / steps_until_temp), copies))

        if trajectories:
            E_vs_t_loop, T_vs_t_loop = mc_loop_trajectories(beta, Jrows, Jvals, s, E, steps, steps_until_temp, random_sites,
                                                            random_chances, random_tempering, tempering)
            E_vs_t = np.concatenate((E_vs_t, E_vs_t_loop))
            T_vs_t = np.concatenate((T_vs_t, T_vs_t_loop))

        elif tempering_probabilities:
            prob_accept += mc_loop_prob_accept(beta, Jrows, Jvals, s, E, steps_until_temp, random_sites,
                                               random_chances, random_tempering, tempering)

    if trajectories:
        return E_vs_t, T_vs_t

    elif tempering_probabilities:
        return prob_accept / blocks


# %% Optimal temperature distribution

def optimal_temperature_distribution(T0,Tf, J, rng, init_steps=None, avg_steps=None, accept_prob_min=0.2,
                                     accept_prob_max=0.6, plot=False):
    if plot:
        import matplotlib.pyplot as plt
    '''
    Returns an optimal temperature distribution 'T_opt' for parallel tempering where the acceptance probability
    of a temper swap is accept_prob_target for each temperature.
    :param T0: Initial value of temperature distribution.
    :param Tf: Final value for temperature distribution.
    :param J: Adjacency matrix of the problem.
    :param init_steps: Number of MC steps not used for the calculation of accept probability.
    :param avg_steps:
    :param accept_prob_target: Target acceptance probability.
    :param error_accept: Admissible error for the measured acceptance probability.
    '''

    total_steps = init_steps + avg_steps

    if plot:
        T_plot = []
        accept_prob_meas_plot = []

    copies = 20
    T = np.concatenate((np.geomspace(T0, Tf / 2, copies // 2, endpoint=False), np.linspace(Tf / 2, Tf, copies // 2)))
    accept_prob_meas = np.zeros([len(T)])
    check = 0

    while np.any(accept_prob_meas[:-1] < accept_prob_min / 2) or check < 2:
        accept_prob_meas = mc_evolution_tests(1 / T, J, steps=total_steps, start=None, steps_until_temp=10, rng=rng,
                                              tempering=avg_steps, tempering_probabilities=True)[::-1]
        if np.all(accept_prob_meas[:-1] > accept_prob_min / 2):
            check += 1
        else:
            check = 0

        if plot:
            print(f'T =  ' + np.array2string(T, precision=3, floatmode='fixed').replace('\n', '') + ', \nP =  ' +
                  np.array2string(accept_prob_meas * 100, precision=2, floatmode='fixed').replace('\n', '') + ' \n')
            T_plot.append(T)
            accept_prob_meas_plot.append(accept_prob_meas)

        dont_insert = []
        inserted_temps = 0
        for c in range(len(T) - 1):
            if accept_prob_meas[c] < accept_prob_min / 2 and c not in dont_insert:
                T = np.insert(T, c + 1 + inserted_temps, (T[c + inserted_temps] + T[c + 1 + inserted_temps]) / 2)
                inserted_temps += 1
                dont_insert.append(c + 1)

    if plot:
        fig, ax = plt.subplots(ncols=2, dpi=300, figsize=[16 / 2, 9 / 2])
        for i, (Ts, Ps) in enumerate(zip(T_plot, accept_prob_meas_plot)):
            ax[0].plot(np.linspace(0, 1, len(Ts)), Ts, color='k', alpha=(i + 1) / (len(T_plot) + 1), marker='.')
            ax[1].plot(Ts[:-1], Ps[:-1], color='k', alpha=(i + 1) / (len(T_plot) + 1), marker='.')
        ax[0].set_yscale('log')
        ax[1].set_xscale('log')
        plt.show()

    accept_prob_meas = mc_evolution_tests(1 / T, J, steps=total_steps, start=None, steps_until_temp=10, rng=rng,
                                          trajectories=False, tempering=avg_steps, tempering_probabilities=True)[::-1]
    T_kill = []
    accept_prob_meas_kill = []
    c = 0
    for trim in range(len(T)):
        T0 = T.copy()
        # while c < len(T) - 2:
        while c < len(T) - 4:
            # while c < len(T) - 3:
            c += 1
            # if accept_prob_meas[c] > accept_prob_max/2 and accept_prob_meas[c + 1] > accept_prob_max/2:
            #     T[c] = (T[c] + T[c + 1]) / 2
            #     T = np.delete(T, c + 1, 0)
            #     c -= 1
            #     accept_prob_meas = mc_evolution(1 / T, J, steps=total_steps, start=None, eq_steps=1, rng=rng,
            #                                     trajectories=False, tempering=avg_steps, tempering_probabilities=True)[::-1]
            if np.all(accept_prob_meas[c:c + 4] > accept_prob_max):
                T[c + 1] = (T[c + 1] + T[c + 2]) / 2
                T = np.delete(T, c + 2, 0)
                c -= 1
                accept_prob_meas = mc_evolution_tests(1 / T, J, steps=total_steps, start=None, steps_until_temp=10,
                                                      rng=rng, trajectories=False, tempering=avg_steps,
                                                      tempering_probabilities=True)[::-1]
                # if np.all(accept_prob_meas[c:c + 3] > accept_prob_max):
                #     T = np.delete(T, c + 1 , 0)
                #     c -= 1
                #     accept_prob_meas = mc_evolution(1 / T, J, steps=total_steps, start=None, eq_steps=1, rng=rng,
                #                                     trajectories=False, tempering=avg_steps, tempering_probabilities=True)[::-1]

                if plot:
                    print(
                        f'T =  ' + np.array2string(T, precision=3, floatmode='fixed').replace('\n', '') + ', \nP =  ' +
                        np.array2string(accept_prob_meas * 100, precision=2, floatmode='fixed').replace('\n',
                                                                                                        '') + ' \n')
                    T_kill.append(T)
                    accept_prob_meas_kill.append(accept_prob_meas)
        if len(T0) == len(T) and np.all(np.isclose(T0, T)):
            break

    if plot and len(T_kill) > 1:
        fig, ax = plt.subplots(ncols=2, dpi=300, figsize=[16 / 2, 9 / 2])
        for i, (Ts, Ps) in enumerate(zip(T_kill, accept_prob_meas_kill)):
            ax[0].plot(np.linspace(0, 1, len(Ts)), Ts, color='k', alpha=(i + 1) / (len(T_kill) + 1), marker='.',
                       linewidth=0.5)
            ax[1].plot(Ts[:-1], Ps[:-1], color='k', alpha=(i + 1) / (len(T_kill) + 1), marker='.', linewidth=0.5)
        ax[0].set_yscale('log')
        ax[1].set_xscale('log')
        plt.show()

    return T, accept_prob_meas
