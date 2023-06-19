from Modules import spin_glass as sg, chip_architecture as ca, monte_carlo as mc

import numpy as np
import networkx as nx
import dwave_networkx as dnx


import matplotlib.pyplot as plt
from importlib import reload
import sys

sys.path.append('../')
ca = reload(ca)
sg = reload(sg)
mc = reload(mc)

# %%
L = 10
A_graph = nx.hexagonal_lattice_graph(L, L, periodic=True)
pos = nx.get_node_attributes(A_graph, 'pos')
size = (L**2)*2
A = np.array(nx.adjacency_matrix(A_graph).todense())


adjacency = 'hexagonal'
distribution = 'ising'
A3D = sg.connectivity_matrix(L, adjacency, distribution, sparse=False, add=0).astype('int8')

for i in range(size):
    for j in range(i):
        if A3D[i, j] == 1 and A[i, j] == 0:
            A_graph.add_edge(list(A_graph)[i], list(A_graph)[j])

#%%
plt.figure(dpi=300, figsize=(10, 10))
nx.draw(A_graph, pos=pos, node_size=50)
plt.show()

# %%
size = 4 ** 3
A = ca.adjacency_matrix(size, adjacency='3', draw_unit_cell=False, draw_architecture=True)

# %%
G = dnx.pegasus_graph(4)
# G = dnx.chimera_graph(3)


#%%
plt.figure(dpi=300)
dnx.draw_pegasus(G, with_labels=True, node_size=50)
plt.show()


# %%
# unit_cell_A = np.array([[0, 1, 1, 0],
#                         [1, 0, 0, 1],
#                         [1, 0, 0, 1],
#                         [0, 1, 1, 0]])
# unit_cell_A = np.ones([4, 4])
# np.fill_diagonal(unit_cell_A, 0)
size = 9
L = round(size ** (1 / 2))
A1d = np.zeros([L, L]) + np.diag(np.ones(L - 1), 1) + np.diag(np.ones(L - 1), -1)
A1d[0, L - 1] = 1
A1d[L - 1, 0] = 1
I = np.eye(L, L)
A2d = np.kron(I, A1d) + np.kron(A1d, I)
A2d[A2d > 0] = 1
A = A2d

A += np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0]])

unit_cell_A = A
unit_cell = nx.Graph(unit_cell_A)

nx.draw_kamada(unit_cell, with_labels=True)
plt.show()

# %%
M = 2
N = 2
K = 3
L = 2

G = ca.periodic_graph(unit_cell, M, N, K, L)

