from numba import njit
import numpy as np
import math
from numpy.polynomial import Polynomial
from scipy.stats import norm
from scipy.optimize import fmin
import sympy as s


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

@njit(fastmath=True, boundscheck=False)
def mc_step(p, βE, E, s, dE, changes):
    for i, (βEi, pi, c) in enumerate(zip(βE, p, changes)):
        if βEi < 0 or pi <= math.exp(-βEi):
            s[i, c] = -s[i, c]
            E[i] += dE[i]



# #Numba 1
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


# Numba 2
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
    n = len(beta)
    for j in range(1, n, 2):
        maybe_temper(ndx, beta, E, p[j], j)
    for j in range(2, n, 2):
        maybe_temper(ndx, beta, E, p[j], j)


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
    n = len(beta)
    for j in range(1, n, 2):
        maybe_temper_prob_accept(ndx, beta, E, p[j], j, prob_accept)
    for j in range(2, n, 2):
        maybe_temper_prob_accept(ndx, beta, E, p[j], j, prob_accept)


# Numba 3
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


@njit(fastmath=True)
def mc_loop_trajectories(beta, Jrows, Jvals, s, E, steps, random_sites, random_chances, random_tempering, tempering):
    size = Jrows.shape[0]
    copies = len(beta)
    dE = np.zeros(E.size, dtype=np.double)
    beta_order = np.argsort(beta).astype(np.int32)

    E_vs_t_loop = np.zeros((steps, copies), dtype=np.double)
    T_vs_t_loop = np.zeros((int(steps / 10), copies), dtype=np.double)

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        delta_cost(dE, s, Jrows, Jvals, flip_sites)
        mc_step(flip_chances, beta * dE, E, s, dE, flip_sites)
        if tempering and n % 10 == 0:
            tempering_step(beta_order, random_tempering[int(n / 10), :], beta, E)
            T_vs_t_loop[int(n / 10), :] = 1 / beta

        E_vs_t_loop[n, :] = E

    return s, E, E_vs_t_loop, T_vs_t_loop


@njit(fastmath=True)
def mc_loop_prob_accept(beta, Jrows, Jvals, s, E, steps, random_sites, random_chances, random_tempering, tempering):
    size = Jrows.shape[0]
    copies = len(beta)
    dE = np.zeros(E.size, dtype=np.double)
    beta_order = np.argsort(beta).astype(np.int32)
    prob_accept = np.zeros(copies).astype(np.int32)

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        delta_cost(dE, s, Jrows, Jvals, flip_sites)
        mc_step(flip_chances, beta * dE, E, s, dE, flip_sites)
        if n > tempering and n % 10 == 0:
            tempering_step_prob_accept(beta_order, random_tempering[int(n / 10), :], beta, E, prob_accept)

    return prob_accept / ((n - tempering) / 10)


@njit(fastmath=True)
def mc_loop(beta, Jrows, Jvals, s, E, steps, random_sites, random_chances, random_tempering, tempering):
    size = Jrows.shape[0]
    copies = len(beta)
    dE = np.zeros(E.size, dtype=np.double)
    beta_order = np.argsort(beta).astype(np.int32)

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        delta_cost(dE, s, Jrows, Jvals, flip_sites)
        mc_step(flip_chances, beta * dE, E, s, dE, flip_sites)
        if tempering and n % 10 == 0:
            tempering_step(beta_order, random_tempering[int(n / 10), :], beta, E)

    return s, E


def mc_evolution(beta, J, steps=None, start=None, eq_steps=1000, rng=None, trajectories=False, tempering=True,
                 prob_accept=False):
    size = np.shape(J)[0]
    copies = len(beta)
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
        T_vs_t = np.array([1 / beta])

    for _ in range(eq_steps):
        random_sites = rng.integers(0, size, (steps, copies))
        random_chances = rng.random((steps, copies))
        random_tempering = rng.random((int(steps / 10), copies))

        if trajectories:
            s, E, E_vs_t_loop, T_vs_t_loop = mc_loop_trajectories(beta, Jrows, Jvals, s, E, steps, random_sites,
                                                                  random_chances,
                                                                  random_tempering, tempering)
            E_vs_t = np.concatenate((E_vs_t, E_vs_t_loop))
            T_vs_t = np.concatenate((T_vs_t, T_vs_t_loop))
        elif prob_accept:
            prob_accept = mc_loop_prob_accept(beta, Jrows, Jvals, s, E, steps, random_sites, random_chances,
                                              random_tempering, tempering)
        else:
            s, E = mc_loop(beta, Jrows, Jvals, s, E, steps, random_sites, random_chances, random_tempering, tempering)

    if trajectories:
        return s, E, E_vs_t, T_vs_t
    elif np.any(prob_accept):
        return prob_accept
    else:
        return s, E, Jrows, Jvals


def q2_q4_thermal_av(beta, Jrows, Jvals, s, E, N_term, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    size = np.shape(Jrows)[0]
    copies = len(beta)
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
        q2_b[i, :], q4_b[i, :], s, E = q2_q4_thermal_av_mc_loop(beta, Jrows, Jvals, s, E,
                                                                steps_per_block / term_steps, term_steps, random_sites,
                                                                random_chances)

    return np.mean(q2_b, 0), np.mean(q4_b, 0)


@njit(fastmath=True)
def q2_q4_thermal_av_mc_loop(beta, Jrows, Jvals, s, E, N_term, term_steps, random_sites, random_chances):
    size = np.shape(Jrows)[0]
    copies = len(beta)
    q2 = np.zeros((int(len(beta) / 2)), dtype='float64')
    q4 = np.zeros((int(len(beta) / 2)), dtype='float64')
    q_ab = np.zeros((int(len(beta) / 2)), dtype='float64')
    dE = np.zeros(E.size, dtype='float64')
    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        delta_cost(dE, s, Jrows, Jvals, flip_sites)
        mc_step(flip_chances, beta * dE, E, s, dE, flip_sites)

        if not bool(np.remainder(n, term_steps)):
            for c in range(int(copies / 2)):
                q_ab[c] = np.sum(s[0 + 2 * c, :] * s[1 + 2 * c, :])
            q_ab /= size
            q2 += q_ab ** 2
            q4 += q_ab ** 4
    q2 /= N_term
    q4 /= N_term
    return q2, q4, s, E


def q2_q4_evolution(beta, J, N_term, eq_steps=1000, rng=None, tempering=True):
    s, E, Jrows, Jvals = mc_evolution(beta, J, steps=None, start=None, eq_steps=eq_steps, rng=rng, tempering=tempering)
    beta_order = np.argsort(beta)[::-1].astype(np.int32)
    beta, s, E = beta[beta_order], s[beta_order, :], E[beta_order]
    B = q2_q4_thermal_av(beta, Jrows, Jvals, s, E, N_term, rng=rng)
    return B


# def optimal_temperature_distribution(T, E_vs_t, T_vs_t, steps_for_avg=50000, target=1.5):
#     '''
#     Returns an optimal temperature distribution 'T_opt' for parallel tempering.
#     :param T: Initial temperature distribution.
#     :param E_vs_t: Energy of each copy for at least 'steps_for_avg' steps.
#     :param T_vs_t: Temperature of each copy ... If there is no tempering 'T_vs_t' will be constant.
#     :param steps_for_avg: Steps used to calculate mu[E](T) and sigma[E](T).
#     :return T_opt: Optimal temperature distribution between T_opt[0]=T[0] and T_opt[-1]=T[-1].
#     '''
#     copies = len(T)
#     T_s = s.symbols('T')#, real=True)
#     mu_s = s.Function('mu')(T)
#     sigma_s = s.Function('sigma')(T)
#
#     if len(E_vs_t) != (len(T_vs_t) - 1) * 10:
#         E_vs_t = np.delete(E_vs_t, 0, 0)
#     T_index_vs_t = np.argsort(np.argsort(T_vs_t))
#     T_E_vs_t = np.zeros([E_vs_t.shape[0], E_vs_t.shape[1]])
#     for copy in range(copies):
#         T_E_vs_t[:, copy] = np.array([[T_vs_t[i, copy]] * 10 for i in range(1, T_vs_t.shape[0])]).flatten()
#     T_E_index_vs_t = np.argsort(T_E_vs_t)
#     E_fixed_T_vs_t = np.zeros_like(E_vs_t)
#     for i in range(E_vs_t.shape[0]):
#         E_fixed_T_vs_t[i, :] = E_vs_t[i, T_E_index_vs_t[i, :]]
#     mu = np.array([norm.fit(E_fixed_T_vs_t[-steps_for_avg:-1, i])[0] for i in range(copies)])
#     sigma = np.array([norm.fit(E_fixed_T_vs_t[-steps_for_avg:-1, i])[1] for i in range(copies)])
#     mu_fit = Polynomial.fit(T, mu, 10)
#     sigma_fit = Polynomial.fit(T, sigma, 10)
#
#     mu_s = s.Poly(mu_fit.coef[::-1], T_s)
#     sigma_s = s.Poly(sigma_fit.coef[::-1], T_s)
#
#     i = 0
#     T_opt = []
#     T_opt.append(T[0])
#     while T_opt[i] < T[-1]:
#         eq = 2 * (mu_s - mu_s(T_opt[i])) - target * (sigma_s + sigma_s(T_opt[i]))
#
#        #  # Idea 1
#        #  Ti = s.solve((2 / target) * (mu_s - mu_s(T_opt[i])) / (sigma_s + sigma_s(T_opt[i])) - 1)
#        #  Ti = np.array(Ti, dtype=complex)
#        #  if np.isclose(Ti,np.real(Ti)):
#        #      Ti = np.real(Ti)[ np.isclose(Ti,np.real(Ti)) ].astype(float)
#        #  else:
#        #      print('Warning: casting complex part of solution')
#        #  Ti = Ti.min()
#        #
#        # # Idea 2
#        #  f = s.lambdify(T_s, s.Abs(eq.simplify()), "numpy")
#        #  Ti = fmin(f, T_opt[i])
#        #  T_opt.append(Ti[0].astype('float'))
#        #  # Idea 3
#        #  Ti = root_scalar(eq, fprime=eq.diff(), x0=T_opt[i], method='newton')  # bracket=[T_opt[i], T[-1]],
#        #  T_opt.append(Ti.root)
#
#         i += 1
#         if i > 100:
#             print(f'Ouch, copies > {i} ...')
#             break
#     T_opt.pop()
#     T_opt.append(T[-1])
#
#     return np.array(T_opt, dtype=float)

def optimal_temperature_distribution(T, J, rng, init_steps=None, avg_steps=None, accept_prob_min=0.2,
                                     accept_prob_max=0.6, plot=False):
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

    T0 = T[0]
    Tf = T[-1]
    accept_prob_meas = np.zeros([len(T)])

    check = 0
    while np.any(accept_prob_meas[:-1] < accept_prob_min / 2) or check < 2:
        accept_prob_meas = mc_evolution(1 / T, J, steps=total_steps, start=None, eq_steps=1, rng=rng,
                                        trajectories=False,
                                        tempering=avg_steps, prob_accept=True)[::-1]
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

    accept_prob_meas = mc_evolution(1 / T, J, steps=total_steps, start=None, eq_steps=1, rng=rng,
                                    trajectories=False,
                                    tempering=avg_steps, prob_accept=True)[::-1]
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
            #                                     trajectories=False, tempering=avg_steps, prob_accept=True)[::-1]
            if np.all(accept_prob_meas[c:c + 4] > accept_prob_max):
                T[c + 1] = (T[c + 1] + T[c + 2]) / 2
                T = np.delete(T, c + 2, 0)
                c -= 1
                accept_prob_meas = mc_evolution(1 / T, J, steps=total_steps, start=None, eq_steps=1, rng=rng,
                                                trajectories=False, tempering=avg_steps, prob_accept=True)[::-1]
                # if np.all(accept_prob_meas[c:c + 3] > accept_prob_max):
                #     T = np.delete(T, c + 1 , 0)
                #     c -= 1
                #     accept_prob_meas = mc_evolution(1 / T, J, steps=total_steps, start=None, eq_steps=1, rng=rng,
                #                                     trajectories=False, tempering=avg_steps, prob_accept=True)[::-1]

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
