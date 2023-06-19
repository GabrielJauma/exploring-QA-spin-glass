from numba import njit
import numpy as np
import math

# max_array_len = 4e8  # Paula's PC
max_array_len = 1e7
cost_functions = 'spin'  # 'qubo'
overlap_variables = 'full'  # 'partial_2D'

if overlap_variables == 'partial':
    print('WARNING: Using partial overlap variables')

# %% Cost functions

if cost_functions == 'spin':

    # Spin glass cost functions
    def cost(J, s):
        if len(J) == 2:
            Jrows, Jvals = J[0], J[1]
            size = s.shape[1]
            J = np.zeros([size, size])
            for i in range(size):
                J[i, Jrows[i]] = Jvals[i]

        return - 0.5 * np.sum(s.T * (s @ J).T, 0)


    @njit(fastmath=True, boundscheck=False)
    def delta_cost_total(Jrows, Jvals, s, dE):
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


    @njit(fastmath=True, boundscheck=False)
    def mc_step(beta, Jrows, Jvals, s, E, dE, flip_chances, flip_sites):
        # Temperatures with T < T_Metropolis_or_HB should be below the critical temperature
        # T_Metropolis_or_HB = 0.25 * ((1 / beta).max() - (1 / beta).min())
        for c, (pi, i) in enumerate(zip(flip_chances, flip_sites)):
            # if 1 / beta[c] > T_Metropolis_or_HB:  # Metropolis Algorithm.
            #     accept_swap = beta[c] * dE[c, i] < 0 or pi <= math.exp(-beta[c] * dE[c, i])
            # else:  # Heat Bath Algorithm
            #     accept_swap = pi <= math.exp(-beta[c] * dE[c, i]) / (math.exp(-beta[c] * dE[c, i]) + 1)
            # accept_swap = pi <= math.exp(-beta[c] * dE[c, i]) / (math.exp(-beta[c] * dE[c, i]) + 1) # Heat Bath Algorithm
            accept_swap = beta[c] * dE[c, i] < 0 or pi <= math.exp(-beta[c] * dE[c, i])  # Metropolis Algorithm.

            if accept_swap:
                s[c, i] = -s[c, i]
                E[c] += dE[c, i]

                dE[c, i] = -dE[c, i]
                columns = Jrows[i]  # nonzero column indices of row a, that is, of J[a,:]
                values = Jvals[i]  # corresponding values, that is, J[a,columns]
                for j, J_ij in zip(columns, values):
                    dE[c, j] += 4 * J_ij * s[c, j] * s[c, i]


elif cost_functions == 'qubo':

    # QUBO const functions
    def cost(J, s):
        µ = J[0]
        Q = J[1]
        c = J[2]

        n = (s + 1) // 2

        return n @ µ + np.diag(n @ Q @ n.T) + c


    @njit(fastmath=True, boundscheck=False)
    def delta_cost_total(µ, Q, s, dE):
        copies = s.shape[0]
        size = s.shape[1]
        for c in range(copies):
            for i in range(size):
                dE[c, i] = µ[i] * (-s[c, i]) + 2 * (-s[c, i]) * np.sum(Q[i, :] * ((s[c, :] + 1) // 2)) + Q[i, i]
                # n = (s[c, :] + 1) // 2
                # ni = n[i] # Old value in binary
                # d = (ni + 1) % 2 - ni # Distance between old and new values in binary
                # dE[c, i] = µ[i] * d + 2 * d * np.sum(Q[i,:]*n) + Q[i, i]


    @njit(fastmath=True, boundscheck=False)
    def mc_step(beta, µ, Q, s, E, dE, flip_chances, flip_sites):
        size = s.shape[1]
        for c, (pi, i) in enumerate(zip(flip_chances, flip_sites)):
            if beta[c] * dE[c, i] < 0 or pi <= math.exp(-beta[c] * dE[c, i]):
                s[c, i] = -s[c, i]
                E[c] += dE[c, i]

                dE[c, i] = -dE[c, i]
                for j in range(size):
                    if j == i:
                        continue
                    dE[c, j] -= 2 * (s[c, j]) * Q[j, i] * ((-s[c, i] + 1) // 2)


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


# %% Monte Carlo
@njit(boundscheck=False, fastmath=True)
def mc_loop(beta, Jrows, Jvals, s, E, dE, random_sites, random_chances, random_tempering):
    size = s.shape[1]
    beta_order = np.argsort(beta).astype(np.int32)

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        mc_step(beta, Jrows, Jvals, s, E, dE, flip_chances, flip_sites)
        if n % size == 0:
            tempering_step(beta, E, beta_order, random_tempering[n // size, :])


def mc_evolution(beta, J, steps=None, start=None, rng=np.random.default_rng()):
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
def thermal_average_mc_loop(beta, Jrows, Jvals, s, E, dE, random_sites, random_chances, random_tempering):
    global overlap_variables

    size = s.shape[1]
    copies = beta.size
    beta_order = np.argsort(beta).astype(np.int32)

    if np.any(beta[beta_order[0::2]] != beta[beta_order[1::2]]):
        raise ValueError('WARNING: beta should be a 2N array with N identical values')

    µ_q2 = np.zeros(copies // 2)
    µ_q4 = np.zeros(copies // 2)
    σ2_q2 = np.zeros(copies // 2)
    σ2_q4 = np.zeros(copies // 2)
    N = 0

    ql = np.zeros(copies // 2)
    U = np.zeros(copies // 2)
    U2 = np.zeros(copies // 2)

    µ_ql = np.zeros(copies // 2)
    µ_U = np.zeros(copies // 2)
    µ_U2 = np.zeros(copies // 2)

    Nb = 0
    for i in range(len(Jrows)):
        Nb += np.sum((Jrows[i] > 0) * (Jrows[i] > i))

    if overlap_variables == 'full':
        total_variables = size
    elif overlap_variables == 'partial_2D':
        total_variables = int(size ** (1 / 2))

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
                for i in range(total_variables):
                    q[c // 2] += s[index_replica_1, i] * s[index_replica_2, i]
                    for j in Jrows[i][Jrows[i] > i]:
                        ql[c // 2] += s[index_replica_1, i] * s[index_replica_1, j] * s[index_replica_2, i] * s[
                            index_replica_2, j]

                U[c // 2] = E[index_replica_1]
                U2[c // 2] = E[index_replica_1] ** 2
            q /= total_variables
            ql /= Nb
            U /= total_variables
            U2 /= total_variables ** 2

            σ2_q2 = (σ2_q2 + (µ_q2 - q ** 2) ** 2 / (N + 1)) * N / (N + 1)
            σ2_q4 = (σ2_q4 + (µ_q4 - q ** 4) ** 2 / (N + 1)) * N / (N + 1)

            µ_q2 = (µ_q2 * N + q ** 2) / (N + 1)
            µ_q4 = (µ_q4 * N + q ** 4) / (N + 1)
            µ_ql = (µ_ql * N + ql) / (N + 1)
            µ_U = (µ_U * N + U) / (N + 1)
            µ_U2 = (µ_U2 * N + U2) / (N + 1)

            N += 1

    return µ_q2, µ_q4[::-1], σ2_q2[::-1], σ2_q4[::-1], µ_ql[::-1], µ_U[::-1], µ_U2[::-1]


def thermal_average(beta, J, s, E, MCS, rng=np.random.default_rng()):
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

    µ_q2_b = np.zeros([blocks, copies // 2])
    µ_q4_b = np.zeros([blocks, copies // 2])
    σ2_q2_b = np.zeros([blocks, copies // 2])
    σ2_q4_b = np.zeros([blocks, copies // 2])

    µ_ql_b = np.zeros([blocks, copies // 2])
    µ_U_b = np.zeros([blocks, copies // 2])
    µ_U2_b = np.zeros([blocks, copies // 2])

    dE = np.empty([copies, size])
    delta_cost_total(Jrows, Jvals, s, dE)

    for i in range(blocks):
        random_sites = rng.integers(0, size, (steps, copies), dtype='int32')
        random_chances = rng.random((steps, copies))
        random_tempering = rng.random((int(steps / size), copies))
        µ_q2_b[i, :], µ_q4_b[i, :], σ2_q2_b[i, :], σ2_q4_b[i, :], µ_ql_b[i, :], µ_U_b[i, :], µ_U2_b[i, :] = \
            thermal_average_mc_loop(beta, Jrows, Jvals, s, E, dE, random_sites, random_chances, random_tempering)
        del random_sites, random_chances, random_tempering

    µ_q2 = np.mean(µ_q2_b, 0)
    µ_q4 = np.mean(µ_q4_b, 0)
    σ2_q2 = (1 / blocks) * np.sum(σ2_q2_b + µ_q2_b ** 2, 0) - µ_q2 ** 2
    σ2_q4 = (1 / blocks) * np.sum(σ2_q4_b + µ_q4_b ** 2, 0) - µ_q4 ** 2
    µ_ql = np.mean(µ_ql_b, 0)
    µ_U = np.mean(µ_U_b, 0)
    µ_U2 = np.mean(µ_U2_b, 0)
    return µ_q2, µ_q4, σ2_q2, σ2_q4, µ_ql, µ_U, µ_U2


def equilibrate_and_average(beta, J, MCS_avg, max_MCS, rng=None, start=None):
    µ_q2_vs_MCS, µ_q4_vs_MCS, σ2_q2_vs_MCS, σ2_q4_vs_MCS, \
    µ_ql_vs_MCS, µ_U_vs_MCS, µ_U2_vs_MCS, MCS_avg_s = [[] for _ in range(8)]

    if start is None:
        s, E = mc_evolution(beta, J, steps=MCS_avg, start=None, rng=rng)
    else:
        s = start
        E = cost(J, s)

    while MCS_avg <= max_MCS:
        print(MCS_avg)
        µ_q2, µ_q4, σ2_q2, σ2_q4, µ_ql, µ_U, µ_U2 = thermal_average(beta, J, s, E, MCS_avg, rng=rng)

        µ_q2_vs_MCS.append(µ_q2), µ_q4_vs_MCS.append(µ_q4), σ2_q2_vs_MCS.append(σ2_q2), σ2_q4_vs_MCS.append(σ2_q4)
        µ_ql_vs_MCS.append(µ_ql), µ_U_vs_MCS.append(µ_U), µ_U2_vs_MCS.append(µ_U2), MCS_avg_s.append(MCS_avg)

        MCS_avg *= 2

    return µ_q2_vs_MCS, µ_q4_vs_MCS, σ2_q2_vs_MCS, σ2_q4_vs_MCS, µ_ql_vs_MCS, µ_U_vs_MCS, µ_U2_vs_MCS, MCS_avg_s


@njit(boundscheck=False, fastmath=True)
def thermal_average_mc_loop_bin(beta, Jrows, Jvals, s, E, dE,
                                random_sites, random_chances, random_tempering, Nb,
                                N, dist, q_dist, µ_q2_bin, µ_q4_bin, σ2_q2_bin, σ2_q4_bin,
                                ql, U, U2, µ_ql, µ_U, µ_U2):
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


# %% Test functions
@njit(boundscheck=False, fastmath=True)
def maybe_temper_prob_accept(beta, E, beta_order, pj, j, prob_accept):
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
        prob_accept[j] += 1


@njit(boundscheck=False, fastmath=True)
def tempering_step_prob_accept(beta, E, beta_order, p, prob_accept):
    """Possibly beta_order temperatures between states.
    We implement a tempering phase in which we run over all temperatures
    from hot (small beta) to cold (large beta), investigating whether we
    beta_order states. In practice `ndx` is the order of the temperatures,
    meaning that state beta[ndx[i]] <= beta[ndx[i+1]]. We run over the
    temperatures in order, computing the probabilities that two configurations
    are exchanged."""
    n = beta.size
    for j in range(2, n, 2):
        maybe_temper_prob_accept(beta, E, beta_order, p[j], j, prob_accept)
    for j in range(3, n, 2):
        maybe_temper_prob_accept(beta, E, beta_order, p[j], j, prob_accept)


@njit(fastmath=True)
def mc_loop_trajectories(beta, Jrows, Jvals, s, E, dE, steps, random_sites, random_chances, random_tempering,
                         tempering, only_E=False):
    size = s.shape[1]
    copies = beta.size
    beta_order = np.argsort(beta).astype(np.int32)

    E_vs_t_loop = np.zeros((steps, copies), dtype=np.double)
    T_vs_t_loop = np.zeros((int(steps / size), copies), dtype=np.double)

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        mc_step(beta, Jrows, Jvals, s, E, dE, flip_chances, flip_sites)

        if not only_E:
            E_vs_t_loop[n, :] = E
        else:
            for i, c in enumerate(beta_order[::-1]):
                E_vs_t_loop[n, i] = E[c]

        if tempering and n % size == 0:
            tempering_step(beta, E, beta_order, random_tempering[n // size, :])
            if not only_E:
                T_vs_t_loop[n // size, :] = 1 / beta

    return E_vs_t_loop, T_vs_t_loop


@njit(fastmath=True)
def mc_loop_prob_accept(beta, Jrows, Jvals, s, E, dE, random_sites, random_chances, random_tempering, tempering):
    copies = beta.size
    size = s.shape[1]
    beta_order = np.argsort(beta).astype(np.int32)
    prob_accept = np.zeros(copies).astype(np.int32)

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        mc_step(beta, Jrows, Jvals, s, E, dE, flip_chances, flip_sites)
        if n > tempering and n % size == 0:
            tempering_step_prob_accept(beta, E, beta_order, random_tempering[n // size, :], prob_accept)

    return prob_accept / (n // size)


def mc_evolution_tests(beta, J, steps=None, start=None, rng=np.random.default_rng(),
                       tempering=True, trajectories=False, tempering_probabilities=False, only_E=False,
                       return_s_E=False):
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

    dE = np.empty([copies, size])
    delta_cost_total(Jrows, Jvals, s, dE)

    for _ in range(blocks):
        random_sites = rng.integers(0, size, (steps, copies), dtype='int32')
        random_chances = rng.random((steps, copies))
        random_tempering = rng.random((steps // size, copies))

        if trajectories:
            E_vs_t_loop, T_vs_t_loop = mc_loop_trajectories(beta, Jrows, Jvals, s, E, dE, steps, random_sites,
                                                            random_chances, random_tempering, tempering, only_E)
            E_vs_t = np.concatenate((E_vs_t, E_vs_t_loop))
            T_vs_t = np.concatenate((T_vs_t, T_vs_t_loop))


        elif tempering_probabilities:
            prob_accept += mc_loop_prob_accept(beta, Jrows, Jvals, s, E, dE, random_sites,
                                               random_chances, random_tempering, tempering)

    if trajectories:
        if only_E:
            if return_s_E:
                return s, E, np.delete(E_vs_t, 0, 0)
            else:
                return np.delete(E_vs_t, 0, 0)
        else:
            return E_vs_t, T_vs_t

    elif tempering_probabilities:
        return prob_accept / blocks


# %% Compare with QAOA

@njit(fastmath=True)
def mc_loop_probability_ground_state(beta, Jrows, Jvals, s, E, dE, random_sites, random_chances,
                                     random_tempering, tempering, E_min):
    size = s.shape[1]
    copies = beta.size
    beta_order = np.argsort(beta).astype(np.int32)
    E0 = E.copy()

    P_vs_t_loop = np.zeros(copies, dtype=np.double)

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        mc_step(beta, Jrows, Jvals, s, E, dE, flip_chances, flip_sites)

        if n % size == 0:
            if tempering:
                tempering_step(beta, E, beta_order, random_tempering[n // size, :])
            for i, c in enumerate(beta_order[::-1]):
                P_vs_t_loop[i] += int(np.abs(E[c] - E_min) < 1e-10)

        E0 = E.copy()

    return P_vs_t_loop


def probability_ground_state(beta, J, steps, s, E, rng, E_min, tempering):
    global max_array_len
    size = J[0].shape[0]
    copies = beta.size
    Jrows, Jvals = J[0], J[1]

    if steps * copies > max_array_len:  # Hardcoded value to ensure that the random arrays (random_sites, ...) fit in Paulas PC.
        blocks = int(steps * copies / max_array_len)
        steps = int(steps / blocks)
    else:
        blocks = 1

    dE = np.empty([copies, size])
    delta_cost_total(Jrows, Jvals, s, dE)
    P_vs_t = np.zeros([len(E)])

    for _ in range(blocks):
        random_sites = rng.integers(0, size, (steps, copies), dtype='int32')
        random_chances = rng.random((steps, copies))
        random_tempering = rng.random((steps // size, copies))

        P_vs_t_loop = mc_loop_probability_ground_state(beta, Jrows, Jvals, s, E, dE, random_sites,
                                                       random_chances, random_tempering, tempering, E_min)
        P_vs_t += P_vs_t_loop

    return P_vs_t


def probability_ground_state_vs_MCS(beta, J, MCS, max_MCS, rng, E_min, tempering):
    size = J[0].shape[0]
    copies = beta.size
    s = rng.integers(0, 2, (copies, size), np.int8) * 2 - 1
    E = cost(J, s)

    P_sample_GS_vs_t = []
    MCS_total = 0
    steps = int(MCS * size)
    while MCS_total <= max_MCS:
        P = probability_ground_state(beta, J, steps, s, E, rng, E_min, tempering) / MCS
        P_sample_GS_vs_t.append(P)
        MCS_total += MCS

    return P_sample_GS_vs_t


@njit(fastmath=True)
def mc_loop_steps_to_ground_state(beta, Jrows, Jvals, s, E, dE, random_sites, random_chances, random_tempering,
                                  tempering, E_min, steps_to_GS):
    size = s.shape[1]
    copies = beta.size
    beta_order = np.argsort(beta).astype(np.int32)
    E0 = E.copy()

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        mc_step(beta, Jrows, Jvals, s, E, dE, flip_chances, flip_sites)

        if tempering and n % size == 0:
            tempering_step(beta, E, beta_order, random_tempering[n // size, :])
            if np.all(steps_to_GS > 0):
                break

        for i, c in enumerate(beta_order[::-1]):
            if steps_to_GS[i] == 0 and np.abs(E[c] - E_min) < 1e-10:
                steps_to_GS[i] = n


def steps_to_ground_state(beta, J, max_MCS, rng, E_min, tempering=True):
    global max_array_len
    size = J[0].shape[0]
    copies = beta.size
    Jrows, Jvals = J[0], J[1]
    s = rng.integers(0, 2, (copies, size), np.int8) * 2 - 1
    E = cost(J, s)
    dE = np.empty([copies, size])
    delta_cost_total(Jrows, Jvals, s, dE)

    steps = int(size * max_MCS)

    if steps * copies > max_array_len:  # Hardcoded value to ensure that the random arrays (random_sites, ...) fit in Paulas PC.
        blocks = int(steps * copies / max_array_len)
        steps = int(steps / blocks)
    else:
        blocks = 1

    steps_to_GS = np.zeros([len(E)], dtype=np.int32)

    for _ in range(blocks):
        random_sites = rng.integers(0, size, (steps, copies), dtype='int32')
        random_chances = rng.random((steps, copies))
        random_tempering = rng.random((steps // size, copies))

        mc_loop_steps_to_ground_state(beta, Jrows, Jvals, s, E, dE, random_sites, random_chances, random_tempering,
                                      tempering, E_min, steps_to_GS)
        if np.all(steps_to_GS > 0):
            return steps_to_GS

    return steps_to_GS


@njit(fastmath=True)
def mc_loop_steps_to_ground_state_no_T(beta, Jrows, Jvals, s, E, dE, random_sites, random_chances, random_tempering,
                                       tempering, E_min):
    size = s.shape[1]
    copies = beta.size
    beta_order = np.argsort(beta).astype(np.int32)
    E0 = E.copy()

    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        mc_step(beta, Jrows, Jvals, s, E, dE, flip_chances, flip_sites)

        if tempering and n % size == 0:
            tempering_step(beta, E, beta_order, random_tempering[n // size, :])

        if np.any(np.abs(E - E_min) < 1e-10):
            return n

    return None


def steps_to_ground_state_no_T(beta, J, max_MCS, rng, E_min, tempering=True):
    global max_array_len
    size = J[0].shape[0]
    copies = beta.size
    Jrows, Jvals = J[0], J[1]
    s = rng.integers(0, 2, (copies, size), np.int8) * 2 - 1
    E = cost(J, s)
    dE = np.empty([copies, size])
    delta_cost_total(Jrows, Jvals, s, dE)

    steps = int(size * max_MCS)

    if steps * copies > max_array_len:  # Hardcoded value to ensure that the random arrays (random_sites, ...) fit in Paulas PC.
        blocks = int(steps * copies / max_array_len)
        steps = int(steps / blocks)
    else:
        blocks = 1

    for block_index in range(blocks):
        random_sites = rng.integers(0, size, (steps, copies), dtype='int32')
        random_chances = rng.random((steps, copies))
        random_tempering = rng.random((steps // size, copies))

        steps_to_GS = mc_loop_steps_to_ground_state_no_T(beta, Jrows, Jvals, s, E, dE, random_sites, random_chances,
                                                         random_tempering,
                                                         tempering, E_min)
        if steps_to_GS is None:
            pass
        else:
            return block_index * steps + steps_to_GS

    return steps_to_GS


def estimate_ground_state(beta, J, MCS, rng, tempering=True):
    counter = 0
    check = 0
    s, E, E_vs_t = mc_evolution_tests(beta, J, MCS, None, rng, tempering, trajectories=True,
                                      tempering_probabilities=False, only_E=True, return_s_E=True)
    E_min_0 = np.min(E_vs_t)
    while check < 5:
        s, E, E_vs_t = mc_evolution_tests(beta, J, MCS, (s, E), rng,
                                          tempering, trajectories=True, tempering_probabilities=False,
                                          only_E=True, return_s_E=True)
        E_min = np.min(E_vs_t)

        if np.isclose(E_min, E_min_0):
            check += 1
        else:
            check = 0
        E_min_0 = E_min
        counter += 1
        if counter >= 10:
            RuntimeError('E_min has not been obtained in the given number of MCS, increase it')

    return E_min


# %% Optimal temperature distribution
def optimal_temperature_distribution_constant_acceptance(T0, Tf, J, rng, MCS_eq, MCS_avg, accept_prob_min=0.2,
                                                         accept_prob_max=0.7, plot=False, copies0=30):
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

    if plot:
        T_plot = []
        accept_prob_meas_plot = []

    copies = copies0
    T = np.concatenate((np.geomspace(T0, Tf / 2, copies // 2, endpoint=False), np.linspace(Tf / 2, Tf, copies // 2)))
    accept_prob_meas = np.zeros([len(T)])
    check = 0

    while np.any(accept_prob_meas[:-1] < accept_prob_min / 2) or check < 2:
        s, E = mc_evolution(1 / T, J, steps=MCS_eq, start=None, rng=rng)
        accept_prob_meas = mc_evolution_tests(1 / T, J, steps=MCS_avg, start=[s, E], rng=rng,
                                              tempering=True, tempering_probabilities=True)[::-1]
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

    s, E = mc_evolution(1 / T, J, steps=MCS_eq, start=None, rng=rng)
    accept_prob_meas = mc_evolution_tests(1 / T, J, steps=MCS_avg, start=[s, E], rng=rng,
                                          tempering=True, tempering_probabilities=True)[::-1]
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
                s, E = mc_evolution(1 / T, J, steps=MCS_eq, start=None, rng=rng)
                accept_prob_meas = mc_evolution_tests(1 / T, J, steps=MCS_avg, start=[s, E], rng=rng,
                                                      tempering=True, tempering_probabilities=True)[::-1]
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

# def optimal_temperature_distribution_constant_entropy(T, CvT, copies, ):
#     copies = 2000
#     dS = np.trapz(CvT, T) / copies
#     error = dS / 1000
#     dT = (Tf - T0) / (copies - 1)
#     T_n = np.zeros(copies)
#     T_n[0] = T0
#
#     for c in range(copies - 1):
#         T_0 = T_n[c]
#         T_1 = T_0 + dT
#         CvT_0 = np.interp(T_0, T, CvT)
#         CvT_1 = np.interp(T_1, T, CvT)
#         dS_01 = np.trapz([CvT_0, CvT_1], [T_0, T_1])
#         while np.abs(dS_01 - dS) > error:
#             dT /= 2
#             if dS_01 < dS:
#                 T_1 += dT
#             else:
#                 T_1 -= dT
#             CvT_1 = np.interp(T_1, T, CvT)
#             dS_01 = np.trapz([CvT_0, CvT_1], [T_0, T_1])
#         T_n[c + 1] = T_1
#         dT = T_1 - T_0
#
# # Tempering alternatives
#
# # @njit("(i4[:], f8[:], f8[:], f8[:])", boundscheck=False, fastmath=True)
# # def tempering_step(ndx, p, beta, E):
# #     for j in range(1, len(E)):
# #         # We compare the states with temperatures betai < betaj
# #         ndxi = ndx[j-1]
# #         betai, Ei = beta[ndxi], E[ndxi]
# #         ndxj = ndx[j]
# #         betaj, Ej = beta[ndxj], E[ndxj]
# #         # According to the temperature probability, we may
# #         # exchange the configurations associated to those
# #         # temperatures
# #         r = (betai - betaj) * (Ei - Ej)
# #         if r >= 0 or p[j] <= math.exp(r):
# #             ndx[j-1], ndx[j] = ndxj, ndxi
# #             beta[ndxi], beta[ndxj] = betaj, betai
#
#
# # @njit("(i4[:], f8[:], f8[:], f8[:], i4[:])", boundscheck=False, fastmath=True)
# # def tempering_step_prob_accept(ndx, p, beta, E, prob_accept):
# #     for j in range(1, len(E)):
# #         # We compare the states with temperatures betai < betaj
# #         ndxi = ndx[j-1]
# #         betai, Ei = beta[ndxi], E[ndxi]
# #         ndxj = ndx[j]
# #         betaj, Ej = beta[ndxj], E[ndxj]
# #         # According to the temperature probability, we may
# #         # exchange the configurations associated to those
# #         # temperatures
# #         r = (betai - betaj) * (Ei - Ej)
# #         if r >= 0 or p[j] <= math.exp(r):
# #             ndx[j-1], ndx[j] = ndxj, ndxi
# #             beta[ndxi], beta[ndxj] = betaj, betai
# #             prob_accept[j - 1] += 1
# #             prob_accept[j] += 1
#
#
# # @njit("(i4[:], f8[:], f8[:], f8[:])", boundscheck=False, fastmath=True)
# # def tempering_step(ndx, p, beta, E):
# #     swapped = []
# #     for j in range(len(E)):
# #         if np.any(np.array(swapped) == j):
# #             continue
# #         else:
# #             for i in range(j + 1, len(E)):
# #                 if not np.any(np.array(swapped) == i) and i != j:
# #                     r = (beta[i] - beta[j]) * (E[i] - E[j])
# #                     if r >= 0 or p[j] <= math.exp(r):
# #                         ndx[i], ndx[j] = ndx[j], ndx[i]
# #                         beta[i], beta[j] = beta[j], beta[i]
# #                         swapped.append(i)
# #                         swapped.append(j)
# #                         break
