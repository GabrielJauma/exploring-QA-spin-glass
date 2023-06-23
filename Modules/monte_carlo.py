from numba import njit
import numpy as np
import math

max_array_len = 1e7
cost_functions = 'spin'  # 'qubo'
overlap_variables = 'full'  # 'partial_2D'

if overlap_variables == 'partial':
    print('WARNING: Using partial overlap variables')

# %% Spin glass cost functions
def cost(J, s):
    # Calculate the energy or cost of a spin glass problem where the variables are binary with possible values +-1
    if len(J) == 2:
        Jrows, Jvals = J[0], J[1]
        size = s.shape[1]
        J = np.zeros([size, size])
        for i in range(size):
            J[i, Jrows[i]] = Jvals[i]

    return - 0.5 * np.sum(s.T * (s @ J).T, 0)


@njit(fastmath=True, boundscheck=False)
def delta_cost_total(Jrows, Jvals, s, dE):
    """
    This function calculates the total change in the cost function ('dE') for a spin-glass problem.
    The inputs are: 'Jrows' and 'Jvals' (the indices and corresponding values of non-zero elements in J),
    's' (the spin configuration), and 'dE' (an array to store the calculated cost change for each spin).
    """

    copies = s.shape[0]
    size = s.shape[1]
    for c in range(copies):
        for i in range(size):
            columns = Jrows[i]  # nonzero column indices of row i, that is, of J[i,:]
            values = Jvals[i]  # corresponding values, that is, J[i,columns]
            E0 = 0.0
            for j, J_ij in zip(columns, values):
                E0 += J_ij * s[c, j]
            dE[c, i] = 2 * E0 * s[c, i]


# %% Monte Carlo
@njit(fastmath=True, boundscheck=False)
def mc_step(beta, Jrows, Jvals, s, E, dE, flip_chances, flip_sites):
    """
    This function performs a step of the Monte Carlo simulation for a spin glass system using the Metropolis algorithm.
    The inputs are: `beta` (the inverse temperature), `Jrows` and `Jvals` (indices and values of non-zero elements in J),
    `s` (the spin configuration), `E` (the current energy), `dE` (the energy change for each spin),
    `flip_chances` (the probability of each spin to flip), and `flip_sites` (the indices of the spins to consider flipping).
    """
    for c, (pi, i) in enumerate(zip(flip_chances, flip_sites)):
        # Determine whether to accept the flip or not according to the Metropolis algorithm
        # If the energy change resulting from the flip is negative (i.e., energy decreases), the flip is accepted
        # If the energy change is positive, the flip is accepted with a probability e^(-beta*dE), where beta is the inverse temperature
        accept_swap = beta[c] * dE[c, i] < 0 or pi <= math.exp(-beta[c] * dE[c, i])
        if accept_swap:
            s[c, i] = -s[c, i]
            E[c] += dE[c, i]
            dE[c, i] = -dE[c, i]
            columns = Jrows[i]  # nonzero column indices of row a, that is, of J[a,:]
            values = Jvals[i]  # corresponding values, that is, J[a,columns]
            for j, J_ij in zip(columns, values):
                dE[c, j] += 4 * J_ij * s[c, j] * s[c, i]

# %% Parallel tempering
@njit("(f8[:], f8[:], i4[:], f8, i4)", boundscheck=False, fastmath=True)
def maybe_temper(beta, E, beta_order, pj, j):
    # We compare the states with temperatures betai < betaj
    ndxi = beta_order[j - 2]
    betai, Ei = beta[ndxi], E[ndxi]
    ndxj = beta_order[j]
    betaj, Ej = beta[ndxj], E[ndxj]

    if betai == betaj:
        raise ValueError("Attempting to swap copies with the same temperature")
    # According to the temperature probability, we may
    # exchange the configurations associated to those
    # temperatures
    r = (betai - betaj) * (Ei - Ej)
    if r >= 0 or pj <= math.exp(r):
        beta_order[j - 2], beta_order[j] = ndxj, ndxi
        beta[ndxi], beta[ndxj] = betaj, betai


@njit("(f8[:], f8[:], i4[:], f8[:])", boundscheck=False, fastmath=True)
def tempering_step(beta, E, beta_order, p):
    """Possibly swap temperatures between states.
    We implement a tempering phase in which we run over all temperatures
    from hot (small beta) to cold (large beta), investigating whether we
    beta_order states. In practice `beta_order` is the order of the temperatures,
    meaning that state beta[beta_order[i]] <= beta[beta_order[i+2]]. We run over the
    temperatures in order, computing the probabilities that two configurations
    are exchanged."""
    n = beta.size

    # Swaps for the first set of replicas
    for j in range(2, n, 2):
        maybe_temper(beta, E, beta_order, p[j], j)

    for j in range(3, n, 2):
        maybe_temper(beta, E, beta_order, p[j], j)


# %% Monte Carlo loops
@njit(boundscheck=False, fastmath=True)
def mc_loop(beta, Jrows, Jvals, s, E, dE, random_sites, random_chances, random_tempering):
    """
    This function performs a loop of Monte Carlo simulation steps for a spin glass system, including tempering steps.
    The inputs are: `beta` (the inverse temperature), `Jrows` and `Jvals` (indices and values of non-zero elements in J),
    `s` (the spin configuration), `E` (the current energy), `dE` (the energy change for each spin),
    `random_sites` (randomly chosen indices of spins to consider flipping),
    `random_chances` (randomly chosen probabilities for each spin to flip), and
    `random_tempering` (random values used for the tempering step).
    """
    size = s.shape[1]
    beta_order = np.argsort(beta).astype(np.int32)
    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        mc_step(beta, Jrows, Jvals, s, E, dE, flip_chances, flip_sites)
        if n % size == 0:
            tempering_step(beta, E, beta_order, random_tempering[n // size, :])


def mc_evolution(beta, J, steps=None, start=None, rng=np.random.default_rng()):
    """
    Function to perform a full Monte Carlo simulation (evolution) for a spin-glass problem.
    `beta`: array of inverse temperatures.
    `J`: a 2D list or array [Jrows, Jvals] representing a matrix with indices and values of non-zero elements.
    `steps`: number of steps in the Monte Carlo simulation, by default, it's size*1e5
    `start`: initial configuration of the system (spin and energy), if None, a random configuration is generated
    `rng`: random number generator (default is np.random.default_rng())
    """
    global max_array_len
    size = J[0].shape[0]
    copies = beta.size
    Jrows, Jvals = J[0], J[1]

    if start is None:
        s = rng.integers(0, 2, (copies, size), np.int8) * 2 - 1
        E = cost(J, s)
    else:
        s = start[0]
        E = start[1]

    if steps is None:
        steps = int(size * 1e5)
    else:
        steps = int(size * steps)

    if steps * copies > max_array_len:
        blocks = int(steps * copies / max_array_len)
        steps = int(steps / blocks)
    else:
        blocks = 1

    dE = np.empty([copies, size])
    delta_cost_total(Jrows, Jvals, s, dE)
    for _ in range(blocks):
        random_sites = rng.integers(0, size, (steps, copies), dtype='int32')
        random_chances = rng.random((steps, copies))
        random_tempering = rng.random((int(steps / size), copies))

        mc_loop(beta, Jrows, Jvals, s, E, dE, random_sites, random_chances, random_tempering)

        del random_sites, random_chances, random_tempering

    return s, E


# %% Thermal averages
@njit(boundscheck=False, fastmath=True)
def thermal_average_mc_loop_bin(beta, Jrows, Jvals, s, E, dE,
                                random_sites, random_chances, random_tempering, Nb,
                                N, dist, q_dist, µ_q2_bin, µ_q4_bin, σ2_q2_bin, σ2_q4_bin,
                                ql, U, U2, µ_ql, µ_U, µ_U2):
    "See the function below"
    size = s.shape[1]
    copies = beta.size
    beta_order = np.argsort(beta).astype(np.int32)
    n_bins = np.shape(σ2_q2_bin)[0]

    n_distributon = dist.shape[0]

    if np.any(beta[beta_order[0::2]] != beta[beta_order[1::2]]):
        raise ValueError('WARNING: beta should be a 2N array with N identical values')

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        mc_step(beta, Jrows, Jvals, s, E, dE, flip_chances, flip_sites)

        if n % size == 0:
            tempering_step(beta, E, beta_order, random_tempering[n // size, :])

            q = np.zeros(copies // 2)
            for c in range(0, copies, 2):
                index_replica_1 = beta_order[c]
                index_replica_2 = beta_order[c + 1]

                # Check if the two replicas are in the same temperature
                if beta[index_replica_1] != beta[index_replica_2]:
                    raise ValueError('Wrong replicas')

                for i in range(size):
                    q[c // 2] += s[index_replica_1, i] * s[index_replica_2, i]
                    for j in Jrows[i][Jrows[i] > i]:
                        ql[c // 2] += s[index_replica_1, i] * s[index_replica_1, j] * s[index_replica_2, i] * s[
                            index_replica_2, j]

                U[c // 2] = E[index_replica_1]
                U2[c // 2] = E[index_replica_1]  ** 2
                q[c // 2] /= size

                if q[c // 2] <= 0:
                    for d in range(n_distributon // 2):
                        if dist[d] < q[c // 2] < dist[d + 1]:
                            q_dist[d, c // 2] += 1
                            break
                else:
                    for d in range(n_distributon - 1, n_distributon // 2 - 1, -1):
                        if dist[d - 1] < q[c // 2] < dist[d]:
                            q_dist[d - 1, c // 2] += 1
                            break

            σ2_q2_bin[0] = (σ2_q2_bin[0] + (µ_q2_bin[0] - q ** 2) ** 2 / (N + 1)) * N / (N + 1)
            σ2_q4_bin[0] = (σ2_q4_bin[0] + (µ_q4_bin[0] - q ** 4) ** 2 / (N + 1)) * N / (N + 1)
            µ_q2_bin[0] = (µ_q2_bin[0] * N + q ** 2) / (N + 1)
            µ_q4_bin[0] = (µ_q4_bin[0] * N + q ** 4) / (N + 1)

            for bin in range(1, n_bins):
                if (N + 1) < 2 ** bin:
                    break
                if (N + 1) % 2 ** bin == 0:
                    N_bin = (N + 1) // 2 ** bin - 1
                    new_point_µ_q2 = µ_q2_bin[bin - 1] * (N_bin + 1) - µ_q2_bin[bin] * N_bin
                    new_point_µ_q4 = µ_q4_bin[bin - 1] * (N_bin + 1) - µ_q4_bin[bin] * N_bin
                    σ2_q2_bin[bin] = (σ2_q2_bin[bin] + (µ_q2_bin[bin] - new_point_µ_q2) ** 2 / (N_bin + 1)) * \
                                     N_bin / (N_bin + 1)
                    σ2_q4_bin[bin] = (σ2_q4_bin[bin] + (µ_q4_bin[bin] - new_point_µ_q4) ** 2 / (N_bin + 1)) * \
                                     N_bin / (N_bin + 1)
                    µ_q2_bin[bin] = µ_q2_bin[bin - 1]
                    µ_q4_bin[bin] = µ_q4_bin[bin - 1]

            ql /= Nb
            U /= size
            U2 /= size ** 2
            µ_ql = (µ_ql * N + ql) / (N + 1)
            µ_U = (µ_U * N + U) / (N + 1)
            µ_U2 = (µ_U2 * N + U2) / (N + 1)
            N += 1
    return N, dist, q_dist, µ_q2_bin, µ_q4_bin, σ2_q2_bin, σ2_q4_bin, ql, U, U2, µ_ql, µ_U, µ_U2


def thermal_average_bin(beta, J, s, E, MCS, rng=np.random.default_rng(), n_distributon=51):
    """
    This function computes the thermal averages and statistical uncertainties of several observables in a spin system.
    These quantities include averages of overlaps and link overlaps, energy and energy square, and overlap distribution.

    Parameters:
    beta (numpy.ndarray): An array of inverse temperatures.
    J (list): A list of two arrays, where J[0] contains the row indices of non-zero elements in the J matrix
              and J[1] contains the corresponding non-zero values.
    s (numpy.ndarray): The current spin configurations.
    E (numpy.ndarray): The current energy configurations.
    MCS (int): The number of Monte Carlo sweeps.
    rng (numpy.random.Generator, optional): A numpy random number generator. Defaults to numpy.random.default_rng().
    n_distributon (int, optional): The number of bins for the overlap distribution. Defaults to 51.

    Returns:
    numpy.ndarray: Returns several arrays containing thermal averages and statistical uncertainties
                   of the spin overlap squared, spin overlap to the fourth power,
                   link overlap, energy, energy squared, and overlap distribution.
    """
    global max_array_len
    size = s.shape[1]
    copies = beta.size
    Jrows, Jvals = J[0], J[1]
    steps = int(size * MCS)
    n_bins = np.log2(MCS).astype('int')

    Nb = 0
    for i in range(len(Jrows)):
        Nb += np.sum((Jrows[i] > 0) * (Jrows[i] > i))

    if steps * copies > max_array_len:
        blocks = int(steps * copies / max_array_len)
        steps = int(steps / blocks)
    else:
        blocks = 1

    N = 0
    dist = np.linspace(-1, 1, n_distributon)
    q_dist = np.zeros([n_distributon - 1, copies // 2], dtype='int32')
    µ_q2_bin = np.zeros([n_bins, copies // 2])
    µ_q4_bin = np.zeros([n_bins, copies // 2])
    σ2_q2_bin = np.zeros([n_bins, copies // 2])
    σ2_q4_bin = np.zeros([n_bins, copies // 2])
    ql = np.zeros(copies // 2)
    U = np.zeros(copies // 2)
    U2 = np.zeros(copies // 2)
    µ_ql = np.zeros(copies // 2)
    µ_U = np.zeros(copies // 2)
    µ_U2 = np.zeros(copies // 2)

    dE = np.empty([copies, size])
    delta_cost_total(Jrows, Jvals, s, dE)

    for i in range(blocks):
        random_sites = rng.integers(0, size, (steps, copies), dtype='int32')
        random_chances = rng.random((steps, copies))
        random_tempering = rng.random((int(steps / size), copies))
        N, dist, q_dist, µ_q2_bin, µ_q4_bin, σ2_q2_bin, σ2_q4_bin, ql, U, U2, µ_ql, µ_U, µ_U2 = \
            thermal_average_mc_loop_bin(beta, Jrows, Jvals, s, E, dE, random_sites, random_chances, random_tempering,
                                        Nb,
                                        N, dist, q_dist, µ_q2_bin, µ_q4_bin, σ2_q2_bin, σ2_q4_bin,
                                        ql, U, U2, µ_ql, µ_U, µ_U2)
        del random_sites, random_chances, random_tempering

    return µ_q2_bin[0, ::-1], µ_q4_bin[0, ::-1], σ2_q2_bin[:, ::-1], σ2_q4_bin[:, ::-1], \
           µ_ql[::-1], µ_U[::-1], µ_U2[::-1], q_dist[::-1]


def equilibrate_and_average_bin(beta, J, MCS_avg, max_MCS, rng=None, start=None):
    """
    This function calculates the thermal averages and statistical uncertainties of several observables
    in a spin system, over an increasing number of Monte Carlo steps.

    Parameters:
    beta (numpy.ndarray): An array of inverse temperatures.
    J (list): A list of two arrays, where J[0] contains the row indices of non-zero elements in the J matrix
              and J[1] contains the corresponding non-zero values.
    MCS_avg (int): The initial number of Monte Carlo steps for averaging.
    max_MCS (int): The maximum number of Monte Carlo steps.
    rng (numpy.random.Generator, optional): A numpy random number generator. Defaults to numpy.random.default_rng().
    start (numpy.ndarray, optional): The initial spin configuration. If None, new spin configurations
                                     will be generated using mc_evolution function.

    Returns:
    list of numpy.ndarray: Returns several lists containing the observables computed over an increasing
                           number of Monte Carlo steps. The observables include averages of overlaps and
                           link overlaps, energy and energy square, overlap distribution and the number
                           of Monte Carlo steps used for each computation.
    """
    µ_q2_vs_MCS, µ_q4_vs_MCS, σ2_q2_bin_vs_MCS, σ2_q4_bin_vs_MCS, \
    µ_ql_vs_MCS, µ_U_vs_MCS, µ_U2_vs_MCS, q_dist_vs_MCS, MCS_avg_s = [[] for _ in range(9)]

    if start is None:
        s, E = mc_evolution(beta, J, steps=MCS_avg, start=None, rng=rng)
    else:
        s = start
        E = cost(J, s)

    while MCS_avg <= max_MCS:
        µ_q2, µ_q4, σ2_q2_bin, σ2_q4_bin, µ_ql, µ_U, µ_U2, q_dist = thermal_average_bin(beta, J, s, E, MCS_avg, rng=rng)

        µ_q2_vs_MCS.append(µ_q2), µ_q4_vs_MCS.append(µ_q4)
        σ2_q2_bin_vs_MCS.append(σ2_q2_bin), σ2_q4_bin_vs_MCS.append(σ2_q4_bin)
        q_dist_vs_MCS.append(q_dist)
        µ_ql_vs_MCS.append(µ_ql), µ_U_vs_MCS.append(µ_U), µ_U2_vs_MCS.append(µ_U2), MCS_avg_s.append(MCS_avg)

        MCS_avg *= 2

    return µ_q2_vs_MCS, µ_q4_vs_MCS, σ2_q2_bin_vs_MCS, σ2_q4_bin_vs_MCS, \
           µ_ql_vs_MCS, µ_U_vs_MCS, µ_U2_vs_MCS, q_dist_vs_MCS, MCS_avg_s


"""
The functions that follow, named as "fast" versions of the ones above, have the difference that they only 
calculate the thermal avgs. and variances of the second and fourth power of the spin overlap, neglecting
the rest of the variables. We do this because these are the only variables neccesary to caclculate the binder cumulant.
"""

@njit(boundscheck=False, fastmath=True)
def thermal_average_mc_loop_fast(beta, Jrows, Jvals, s, E, dE,
                                 random_sites, random_chances, random_tempering,
                                 N, µ_q2, µ_q4):
    size = s.shape[1]
    copies = beta.size
    beta_order = np.argsort(beta).astype(np.int32)

    if np.any(beta[beta_order[0::2]] != beta[beta_order[1::2]]):
        raise ValueError('WARNING: beta should be a 2N array with N identical values')

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        mc_step(beta, Jrows, Jvals, s, E, dE, flip_chances, flip_sites)

        if n % size == 0:
            q = np.zeros(copies // 2)
            tempering_step(beta, E, beta_order, random_tempering[n // size, :])
            for c in range(0, copies, 2):
                index_replica_1 = beta_order[c]
                index_replica_2 = beta_order[c + 1]
                # Check if the two replicas are in the same temperature
                if beta[index_replica_1] != beta[index_replica_2]:
                    raise ValueError('Wrong replicas')
                for i in range(size):
                    q[c // 2] += s[index_replica_1, i] * s[index_replica_2, i]

            q /= size
            µ_q2 = (µ_q2 * N + q ** 2) / (N + 1)
            µ_q4 = (µ_q4 * N + q ** 4) / (N + 1)
            N += 1

    return N, µ_q2, µ_q4

def thermal_average_fast(beta, J, s, E, MCS, rng=np.random.default_rng()):
    global max_array_len
    size = s.shape[1]
    copies = beta.size
    Jrows, Jvals = J[0], J[1]
    steps = int(size * MCS)

    if steps * copies > max_array_len:
        blocks = int(steps * copies / max_array_len)
        steps = int(steps / blocks)
    else:
        blocks = 1

    N_term_samples = 0
    µ_q2 = np.zeros(copies // 2)
    µ_q4 = np.zeros(copies // 2)

    dE = np.empty([copies, size])
    delta_cost_total(Jrows, Jvals, s, dE)

    for i in range(blocks):
        random_sites = rng.integers(0, size, (steps, copies), dtype='int32')
        random_chances = rng.random((steps, copies))
        random_tempering = rng.random((int(steps / size), copies))
        N_term_samples, µ_q2, µ_q4 = thermal_average_mc_loop_fast(beta, Jrows, Jvals, s, E, dE,
                                                                  random_sites, random_chances, random_tempering,
                                                                  N_term_samples, µ_q2, µ_q4)
        del random_sites, random_chances, random_tempering

    return µ_q2, µ_q4


def equilibrate_and_average_fast(beta, J, max_MCS, rng=None):
    s, E = mc_evolution(beta, J, steps=max_MCS, start=None, rng=rng)
    µ_q2, µ_q4 = thermal_average_fast(beta, J, s, E, max_MCS, rng=rng)

    return µ_q2[::-1], µ_q4[::-1]
