import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def unit_cell_index(M, N, K, L, k, l):
    if k > K - 1:
        print('k must be smaller thank K-1')
        return np.NaN
    if l > L - 1:
        print('l must be smaller thank L-1')
        return np.NaN

    return M * N * (k + K * l)


def periodic_graph(unit_cell, M, N, K, L):
    """
    Receives a rectangular graph "unit_cell" with "M" rows and "N" columns and creates a new graph by tiling
    the "unit_cell" in "K" rows and "L" columns.
    """
    uc_nodes = np.array(list(unit_cell.nodes))
    uc_edges = np.array(list(unit_cell.edges))

    G = nx.Graph()
    for l in range(L):
        for k in range(K):
            index = unit_cell_index(M, N, K, L, k, l)
            nodes = list(uc_nodes + index)
            edges = tuple(map(tuple, uc_edges + index))
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)

            if k > 0:
                for m in range(M):
                    u = (N - 1) + N * m + unit_cell_index(M, N, K, L, k - 1, l)
                    v = N * m + unit_cell_index(M, N, K, L, k, l)
                    G.add_edge(u, v)
            if l > 0:
                for n in range(N):
                    u = (M - 1) + n + 1 + unit_cell_index(M, N, K, L, k, l - 1)
                    v = n + unit_cell_index(M, N, K, L, k, l)
                    G.add_edge(u, v)
    return G


def adjacency_matrix(size, adjacency, draw_unit_cell=False, draw_architecture=False):
    if adjacency == '1':
        unit_cell_A = np.array([[0, 1, 1, 0],
                                [1, 0, 0, 1],
                                [1, 0, 0, 1],
                                [0, 1, 1, 0]])
        M = N = 2
    elif adjacency == '2':
        unit_cell_A = np.array([[0, 1, 1, 1],
                                [1, 0, 0, 1],
                                [1, 0, 0, 1],
                                [1, 1, 1, 0]])
        M = N = 2
    elif adjacency == '3':
        unit_cell_A = np.array([[0, 1, 1, 1],
                                [1, 0, 1, 1],
                                [1, 1, 0, 1],
                                [1, 1, 1, 0]])
        M = N = 2

    K = L = int(np.sqrt(size / (M * N)))
    if np.round(np.sqrt(size / (M * N))) != np.sqrt(size / (M * N)):
        raise ValueError('Error adjacency_matrix')
    unit_cell = nx.Graph(unit_cell_A)
    A_graph = periodic_graph(unit_cell, M, N, K, L)

    if draw_unit_cell:
        plt.figure(dpi=400)
        nx.draw(unit_cell, with_labels=True)
        plt.show()
    if draw_architecture:
        plt.figure(dpi=400)
        nx.draw_kamada_kawai(A_graph, with_labels=True)
        plt.show()

    return np.array(nx.adjacency_matrix(A_graph).todense())
