import Modules.monte_carlo as mc
import Modules.chip_architecture as ca

import numpy as np
import scipy.sparse as sp
import networkx as nx


def connectivity_matrix(size, adjacency, distribution, seed=912384, trim=0):
    # Bond distribution
    if distribution == 'gaussian_SK':
        mu = 0
        sigma = 1 / np.sqrt(size)
        J = np.random.default_rng(seed).normal(mu, sigma, [size, size])

    elif distribution == 'gaussian_EA':
        mu = 0
        sigma = 1
        J = np.random.default_rng(seed).normal(mu, sigma, [size, size])
    elif distribution == 'binary':
        J = np.random.default_rng(seed).integers(0, 2, [size, size], dtype='int8')
        J[J == 0] = -1
    elif distribution == 'ising':
        J = np.ones([size, size])

    if trim > 0:
        T = np.ones([size, size])
        T[np.random.random([size, size]) < trim] = 0

    for i in range(1, size):  # Make the bond distribution symmetric
        J[i, :i] = J[:i, i]
        if trim > 0:
            T[i, :i] = T[:i, i]

    # Adjacency
    if adjacency == 'SK':
        A = np.ones([size, size])

    elif adjacency == '2D':
        L = round(size ** (1 / 2))
        A1d = np.zeros([L, L]) + np.diag(np.ones(L - 1), 1) + np.diag(np.ones(L - 1), -1)
        A1d[0, L - 1] = 1
        A1d[L - 1, 0] = 1
        I = np.eye(L, L)
        A2d = np.kron(I, A1d) + np.kron(A1d, I)
        A2d[A2d > 0] = 1
        A = A2d

    elif adjacency == '2Dx':
        L = round(size ** (1 / 2))
        A1d = np.zeros([L, L]) + np.diag(np.ones(L - 1), 1) + np.diag(np.ones(L - 1), -1)
        A1d[0, L - 1] = 1
        A1d[L - 1, 0] = 1
        I = np.eye(L, L)
        A2d = np.kron(I, A1d) + np.kron(A1d, I) + np.kron(A1d, A1d)
        A2d[A2d > 0] = 1
        A = A2d

    elif adjacency == '3D':
        L = round(size ** (1 / 3))
        A1d = np.zeros([L, L]) + np.diag(np.ones(L - 1), 1) + np.diag(np.ones(L - 1), -1)
        A1d[0, L - 1] = 1
        A1d[L - 1, 0] = 1
        I = np.eye(L, L)
        A2d = np.kron(I, A1d) + np.kron(A1d, I)
        A3d = np.kron(I, A2d) + np.kron(A2d, I)
        A3d[A3d > 0] = 1
        A = A3d
    elif adjacency == 'random_regular_3':
        if np.remainder(3*size, 2) != 0:
            size += 1
        A_graph = nx.random_regular_graph(d=3, n=size, seed=seed)
        A = np.array(nx.adjacency_matrix(A_graph).todense())
    elif adjacency == 'random_regular_4':
        if np.remainder(4*size, 2) != 0:
            size += 1
        A_graph = nx.random_regular_graph(d=4, n=size, seed=seed)
        A = np.array(nx.adjacency_matrix(A_graph).todense())
    elif adjacency == 'random_regular_5':
        if np.remainder(5*size, 2) != 0:
            size += 1
        A_graph = nx.random_regular_graph(d=5, n=size, seed=seed)
        A = np.array(nx.adjacency_matrix(A_graph).todense())

    elif adjacency == 'hexagonal':
        L = int((size/2)**(1/2))
        A_graph = nx.hexagonal_lattice_graph(L, L, periodic=True)
        A = np.array(nx.adjacency_matrix(A_graph).todense())

    elif adjacency == 'hexagonal_np_1':
        L = int((size/2)**(1/2))
        A_graph = nx.hexagonal_lattice_graph(L, L, periodic=True)
        for i in range(L - 2):
            for j in range(L * 2 - 2):
                if i % 2 == 0 and j % 2 == 0:
                    A_graph.add_edge((i, j), (i + 2, j + 3))
        A = np.array(nx.adjacency_matrix(A_graph).todense())

    elif adjacency == 'hexagonal_np_2':
        L = int((size / 2) ** (1 / 2))
        A_graph = nx.hexagonal_lattice_graph(L, L, periodic=True)
        for i in range(L - 2):
            for j in range(L * 2 - 3):
                if (i % 2 == 0 and j % 2 == 0) or (i % 2 != 0 and j % 2 != 0):
                    A_graph.add_edge((i, j), (i + 2, j + 3))
        A = np.array(nx.adjacency_matrix(A_graph).todense())
    else:
        A = ca.adjacency_matrix(size, adjacency, draw_unit_cell=False, draw_architecture=False)

    # No self interaction
    np.fill_diagonal(A, 0)
    A = A.astype('int8')

    if trim > 0:
        return sp.csr_matrix(A * J * T)
    else:
        return sp.csr_matrix(A * J)


def order_parameters(β, s_t, E_t):
    size = np.shape(s_t)[-1]
    N_term = np.shape(E_t)[0]

    E = np.mean(E_t, 0)
    E_err = np.std(E_t, 0) / np.sqrt(N_term)

    m_t = np.mean(s_t, 2)
    M = np.mean(m_t, 0)
    M_err = np.std(m_t, 0) / np.sqrt(N_term)

    C_V = β ** 2 * (np.mean(E_t ** 2, 0) - np.mean(E_t, 0) ** 2) / size
    X = β * (np.mean(m_t ** 2, 0) - np.mean(m_t, 0) ** 2) * size

    return E, E_err, M, M_err, C_V, X


def q_EA(s):  # β, J,   N_term, tempering_in_term):
    q = np.mean(s, 1)
    q_t_av_c_av = np.mean(q ** 2, (0, 2))

    return q_t_av_c_av


def q_ab(s):
    s = np.moveaxis(s, -2, 0)
    q_ab_i = np.moveaxis(s[0::2, :] * s[1::2, :], 0, -2)
    return np.mean(q_ab_i, -1)


def binder_cumulant_conf_avg(s):  # β, J,   N_term, tempering_in_term):
    N_configs = np.shape(s)[0]
    q_ab = np.mean(s[:, :, 0::2, :] * s[:, :, 1::2, :], 3)
    q2 = q_ab ** 2
    q4 = q_ab ** 4

    q2_t_av = np.mean(q2, 1)
    q4_t_av = np.mean(q4, 1)

    B_t_av = 0.5 * (3 - q4_t_av / (q2_t_av ** 2))
    B_c_av = np.mean(B_t_av, 0)
    B_c_av_error = np.std(B_t_av, 0) / np.sqrt(N_configs)

    return B_c_av, B_c_av_error


def binder_cumulant_term_avg(s):  # β, J,   N_term, tempering_in_term):
    N_term = np.shape(s)[0]
    q_ab = np.mean(s[:, 0::2, :] * s[:, 1::2, :], 2)
    q2 = q_ab ** 2
    q4 = q_ab ** 4

    q2_t_av = np.mean(q2, 0)
    q4_t_av = np.mean(q4, 0)

    σ_q2 = np.std(q2, 0)
    σ_q4 = np.std(q4, 0)
    dBdq4 = -1 / q2_t_av
    dBdq2 = 2 * q4_t_av / q2_t_av ** 3

    B_t_av = 0.5 * (3 - q4_t_av / (q2_t_av ** 2))
    B_t_av_error = np.sqrt((σ_q2 * dBdq2) ** 2 + (σ_q4 * dBdq4) ** 2) / np.sqrt(N_term)

    return B_t_av, B_t_av_error
