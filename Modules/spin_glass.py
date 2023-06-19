# import Modules.chip_architecture as ca
import numpy as np
import scipy.sparse as sp
import networkx as nx


def connectivity_matrix(size, adjacency, distribution, rng=np.random.default_rng(), trim=0, add=0, sparse=True, periodic=True, bonds='diagonal'):
    # if adjacency == 'chimera':
    #     size = size ** 2 * 8
    # elif adjacency == 'pegasus':
    #     size = 24 * size * (size - 1) - 8 * (size - 1)
    # elif adjacency[:9] == 'hexagonal':
    #     size = size ** 2 * 2

    # Adjacency
    if adjacency == 'SK':
        A = np.ones([size, size])

    elif adjacency == '2D':
        L = round(size ** (1 / 2))
        A1d = np.zeros([L, L]) + np.diag(np.ones(L - 1), 1) + np.diag(np.ones(L - 1), -1)
        if periodic:
            A1d[0, L - 1] = 1
            A1d[L - 1, 0] = 1
        I = np.eye(L, L)
        A2d = np.kron(I, A1d) + np.kron(A1d, I)
        A2d[A2d > 0] = 1
        A = A2d

    elif adjacency == '2Dx':
        L = round(size ** (1 / 2))
        A1d = np.zeros([L, L]) + np.diag(np.ones(L - 1), 1) + np.diag(np.ones(L - 1), -1)
        if periodic:
            A1d[0, L - 1] = 1
            A1d[L - 1, 0] = 1
        I = np.eye(L, L)
        A2d = np.kron(I, A1d) + np.kron(A1d, I) + np.kron(A1d, A1d)
        A2d[A2d > 0] = 1
        A = A2d

    elif adjacency == '3D':
        L = round(size ** (1 / 3))
        A1d = np.zeros([L, L]) + np.diag(np.ones(L - 1), 1) + np.diag(np.ones(L - 1), -1)
        if periodic:
            A1d[0, L - 1] = 1
            A1d[L - 1, 0] = 1
        I = np.eye(L, L)
        A2d = np.kron(I, A1d) + np.kron(A1d, I)
        A3d = np.kron(I, A2d) + np.kron(A2d, I)
        A3d[A3d > 0] = 1
        A = A3d

    elif adjacency == 'random_regular_3':
        if np.remainder(3 * size, 2) != 0:
            size += 1
        A_graph = nx.random_regular_graph(d=3, n=size, seed=np.random.RandomState(seed=rng.bit_generator))
        A = np.array(nx.adjacency_matrix(A_graph).todense())

    elif adjacency == 'random_regular_4':
        if np.remainder(4 * size, 2) != 0:
            size += 1
        A_graph = nx.random_regular_graph(d=4, n=size, seed=np.random.RandomState(seed=rng.bit_generator))
        A = np.array(nx.adjacency_matrix(A_graph).todense())

    elif adjacency == 'random_regular_5':
        if np.remainder(5 * size, 2) != 0:
            size += 1
        A_graph = nx.random_regular_graph(d=5, n=size, seed=np.random.RandomState(seed=rng.bit_generator))
        A = np.array(nx.adjacency_matrix(A_graph).todense())

    elif adjacency == 'random_regular_7':
        if np.remainder(7 * size, 2) != 0:
            size += 1
        A_graph = nx.random_regular_graph(d=7, n=size, seed=np.random.RandomState(seed=rng.bit_generator))
        A = np.array(nx.adjacency_matrix(A_graph).todense())

    elif adjacency == 'random_regular_9':
        if np.remainder(9 * size, 2) != 0:
            size += 1
        A_graph = nx.random_regular_graph(d=9, n=size, seed=np.random.RandomState(seed=rng.bit_generator))
        A = np.array(nx.adjacency_matrix(A_graph).todense())

    elif adjacency == 'hexagonal':
        L = int((size / 2) ** (1 / 2))
        A_graph = nx.hexagonal_lattice_graph(L, L, periodic=periodic)
        A = np.array(nx.adjacency_matrix(A_graph).todense())

    elif adjacency == 'hexagonal_np_1' or adjacency == 'hexagonal_np_2':
        L = int((size / 2) ** (1 / 2))
        A_graph = nx.hexagonal_lattice_graph(L, L, periodic=periodic)
        A = np.array(nx.adjacency_matrix(A_graph).todense())

        # Add the non-planar bonds between qubits 2+jump rows apart. jump must be high enough to ensure that
        # the non-planar bonds create a 3D-like adjacency matrix.
        jump = 0
        while np.max(np.argwhere(A[:, 0] == 1)) <= round(size ** (2 / 3)):
            A_graph = nx.hexagonal_lattice_graph(L, L, periodic=periodic)

            if adjacency == 'hexagonal_np_1':
                for i in range(L - 2):
                    for j in range(L * 2 - 2):
                        if i % 2 == 0 and j % 2 == 0:
                            if (i + 2 + jump, j + 3 + jump) in list(A_graph):
                                A_graph.add_edge((i, j), (i + 2 + jump, j + 3 + jump))

            elif adjacency == 'hexagonal_np_2':
                for i in range(L - 2):
                    for j in range(L * 2 - 3):
                        if (i % 2 == 0 and j % 2 == 0) or (i % 2 != 0 and j % 2 != 0):
                            if (i + 2 + jump, j + 3 + jump) in list(A_graph):
                                A_graph.add_edge((i, j), (i + 2 + jump, j + 3 + jump))
            jump += 1

            A = np.array(nx.adjacency_matrix(A_graph).todense())

    elif adjacency == 'hexagonal_np_3D':
        L = int((size / 2) ** (1 / 2))
        A_graph = nx.hexagonal_lattice_graph(L, L, periodic=periodic)
        A = np.array(nx.adjacency_matrix(A_graph).todense())

        for i in range(0, size - int(size ** (2 / 3)), 2):
            A[i, i + int(size ** (2 / 3))] = 1
            A[i + int(size ** (2 / 3)), i] = 1

    elif adjacency == 'hexagonal_np_3D_full':
        L = int((size / 2) ** (1 / 2))
        A_graph = nx.hexagonal_lattice_graph(L, L, periodic=periodic)
        A = np.array(nx.adjacency_matrix(A_graph).todense())

        for i in range(size - int(size ** (2 / 3))):
            A[i, i + int(size ** (2 / 3))] = 1
            A[i + int(size ** (2 / 3)), i] = 1

    elif adjacency == 'hexagonal_np_3D_half':
        L = int((size / 2) ** (1 / 2))
        A_graph = nx.hexagonal_lattice_graph(L, L, periodic=periodic)
        A = np.array(nx.adjacency_matrix(A_graph).todense())

        for i in range(size - int(size ** (2 / 3))):
            if np.sum(A[i, :]) < 4:
                A[i, i + int(size ** (2 / 3))] = 1
                A[i + int(size ** (2 / 3)), i] = 1

    elif adjacency == 'chimera':
        chimera_size = int((size / 8) ** (1 / 2))
        try:
            import dwave_networkx as dnx
            A_graph = dnx.chimera_graph(chimera_size)
            A = np.array(nx.adjacency_matrix(A_graph).todense())
            # file = f'/home/gabriel/OneDrive/2021/Avaqus/Architecture_v2/Cluster/Adjacency_matrices/{adjacency}_{chimera_size}.npy'
            # np.save(file, A)
        except:
            file = f'/home/csic/qia/gjg/Cluster/Adjacency_matrices/{adjacency}_{chimera_size}.npy'
            A = np.load(file)

    elif adjacency == 'zephyr':
        zephyr_size = int((-0.5 + np.sqrt(0.5**2+4*size/32)) / 2)
        try:
            import dwave_networkx as dnx
            A_graph = dnx.zephyr_graph(zephyr_size)
            A = np.array(nx.adjacency_matrix(A_graph).todense())
            # file = f'/home/gabriel/OneDrive/2021/Avaqus/Architecture_v2/Cluster/Adjacency_matrices/{adjacency}_{zephyr_size}.npy'
            # file = f'/media/gabriel/D/onedrive/2021/Avaqus/Architecture_v1/Cluster/Adjacency_matrices/{adjacency}_{zephyr_size}.npy'
            # np.save(file, A)
        except:
            file = f'/home/csic/qia/gjg/Cluster/Adjacency_matrices/{adjacency}_{zephyr_size}.npy'
            A = np.load(file)

    elif adjacency == 'pegasus':
        pegasus_size = int((32 + np.sqrt(32 ** 2 - 4 * 24 * (8 - size))) / 48) + 1
        try:
            import dwave_networkx as dnx
            A_graph = dnx.pegasus_graph(pegasus_size)
            A = np.array(nx.adjacency_matrix(A_graph).todense())
        except:
            file = f'/home/csic/qia/gjg/Cluster/Adjacency_matrices/{adjacency}_{pegasus_size}.npy'
            A = np.load(file)


    elif adjacency == 'portafolio':
        import Portafolio.get_financial_data as gfd
        chosen_assets = rng.choice(400, size, replace=False)
        λ = distribution[0]
        λN = distribution[1]
        P0 = distribution[2]
        end_date = distribution[3]

        _, S, _, µ, _, prices = gfd.get_returns_and_correlations('Portafolio/SP500_data_from_2008-01-01.csv',
                                                                 date=end_date, ix_assets=chosen_assets)

        nmin = np.zeros(size)
        Nq = np.full((size,), 3)
        µ, Q, c = gfd.make_QUBO(µ, S, λ, λN, prices, P0, nmin, Nq)
        J = [µ, Q, c]
        return J

    elif adjacency == '1D+':
        L = size
        A = np.zeros([L, L]) + np.diag(np.ones(L - 1), 1) + np.diag(np.ones(L - 1), -1)
        if periodic:
            A[0, L - 1] = 1
            A[L - 1, 0] = 1

        if add < 1:
            extra_bonds = 0
            while extra_bonds < (L ** 2 - L) * add / 2:
                bond = rng.choice(L, 2, replace=False)
                if A[bond[0], bond[1]] == 0:
                    A[bond[0], bond[1]] = 1
                    A[bond[1], bond[0]] = 1
                    extra_bonds += 1
        elif add > 2:
            while np.mean(A)*size < add:
                bond = rng.choice(L, 2, replace=False)
                if A[bond[0], bond[1]] == 0:
                    A[bond[0], bond[1]] = 1
                    A[bond[1], bond[0]] = 1

    elif adjacency == '2D_small_world':

        L = round(size ** (1 / 2))
        n = L - 1
        G = nx.grid_2d_graph(L, L, periodic=False)
        D = sp.csgraph.shortest_path(nx.adjacency_matrix(G)).astype('int64')

        n_original_bonds = len(G.edges)

        while D.mean() > np.log(size): #D.max() > size ** (1 / 3) * 3 // 2:  # D_max_rrg or D.mean() > D_mean_rrg:
            c = 0
            max_dist = np.where(D == D.max())
            new_bond = False
            while new_bond == False:
                for k in range(len(max_dist[0])):
                    i, j = max_dist[0][k], max_dist[1][k]
                    node_0, node_1 = list(G)[i], list(G)[j]

                    if bonds == 'both':
                        bond_condition = np.abs(node_0[0]-node_1[0]) == np.abs(node_0[1]-node_1[1]) or node_0[0] == node_1[0] or node_0[1] == node_1[1]
                    elif bonds == 'horizontal_vertical':
                        bond_condition = node_0[0] == node_1[0] or node_0[1] == node_1[1]
                    elif bonds == 'diagonal':
                        bond_condition = np.abs(node_0[0]-node_1[0]) == np.abs(node_0[1]-node_1[1])

                    if bond_condition:
                        new_bond = True
                        break

                c += 1
                max_dist = np.where(D == D.max() - c)

            new_bonds = [(list(G)[i], list(G)[j]),
                         (tuple(np.abs(np.subtract((0, n), list(G)[i]))),
                          tuple(np.abs(np.subtract((0, n), list(G)[j])))),
                         (tuple(np.abs(np.subtract((n, 0), list(G)[i]))),
                          tuple(np.abs(np.subtract((n, 0), list(G)[j])))),
                         (tuple(np.abs(np.subtract((n, n), list(G)[i]))),
                          tuple(np.abs(np.subtract((n, n), list(G)[j]))))]

            G.add_edge(new_bonds[0][0], new_bonds[0][1])
            G.add_edge(new_bonds[1][0], new_bonds[1][1])
            G.add_edge(new_bonds[2][0], new_bonds[2][1])
            G.add_edge(new_bonds[3][0], new_bonds[3][1])

            D = sp.csgraph.shortest_path(nx.adjacency_matrix(G)).astype('int64')

        A = np.array(nx.adjacency_matrix(G).todense())
        n_extra_bonds = len(G.edges)-n_original_bonds

    # else:
    #     A = ca.adjacency_matrix(size, adjacency, draw_unit_cell=False, draw_architecture=False)

    # No self interaction
    np.fill_diagonal(A, 0)
    A = A.astype('int8')

    size = A.shape[0]

    # Distribution (of the bonds)
    if distribution == 'gaussian_SK':
        mu = 0
        sigma = 1 / np.sqrt(size)
        J = rng.normal(mu, sigma, [size, size])
    elif distribution == 'gaussian_EA':
        mu = 0
        sigma = 1
        J = rng.normal(mu, sigma, [size, size])
    elif distribution == 'binary':
        J = rng.integers(0, 2, [size, size], dtype='int8')
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

    if sparse:
        if trim > 0:
            return custom_sparse(A * J * T)
        else:
            return custom_sparse(A * J)

    # if adjacency == '2D_small_world':
    #     return A * J, n_extra_bonds

    return A * J


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


def custom_sparse(J):
    J = sp.csr_matrix(J)
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

def custom_dense(J):
    Jrows, Jvals = J[0], J[1]
    size = Jrows.shape[0]
    J = np.zeros([size, size])
    for i in range(size):
        J[i, Jrows[i]] = Jvals[i]
    return J

