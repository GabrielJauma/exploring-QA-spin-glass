import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
import networkx as nx
from networkx.algorithms.approximation import diameter

import Modules.spin_glass as sg
from Modules.fractal_dimension import dimension, box_counting, box_counting_numba, compact_box_counting_numba, \
    dist_from_node

import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import curve_fit


# %% Calculate matrices A and D, and dimension, for a given network 1/6
# L = 6
# size = L ** 2  # 30 ** 2 * 2
size = 100
adjacency = '1D+'
# adjacency = '3D'
add = 5.0
distribution = 'gaussian_EA'
periodic = True
if distribution == 'gaussian_EA' or distribution == 'gaussian_SK':
    A = sg.connectivity_matrix(size, adjacency, distribution, sparse=False, add=add, periodic=periodic)
else:
    A = sg.connectivity_matrix(size, adjacency, distribution, sparse=False, add=add, periodic=periodic).astype('int8')

print(np.mean(A[A!=0]), np.var(A[A!=0]))
#%%
D = shortest_path(csr_matrix(A)).astype('int64')
G = nx.Graph(A)

del A
l_max = np.max(D)
# print(dimension(D))
print(np.max(D), np.mean(D))
# print(diameter(G))
D_max_rrg = np.max(D)
D_mean_rrg = np.mean(D)

print( size ** (1/3) * 3/2 )
# %% Add connections to nodes with the largest distance
L = 10
size = L ** 2  # 30 ** 2 * 2
n = L-1
# G = nx.grid_2d_graph(L, L, periodic=False)
G = nx.hexagonal_lattice_graph(L, L, periodic=False)
G_new_bonds = nx.create_empty_copy(G)

pos = {(x, y): (x, y) for x, y in G.nodes()}
D = shortest_path(nx.adjacency_matrix(G)).astype('int64')

original_bonds = list(G.edges)
n_original_bonds = len(G.edges)
extra_bonds = []

# %%
plt.figure(dpi=100, figsize=[10, 10])
nx.draw(G, pos=pos, node_size=50)
plt.suptitle(f'original_bonds = {n_original_bonds}, n_extra_bonds = {len(G.edges)-n_original_bonds}, D_max = {D.max()}, D_mean = {D.mean().round(3)}')
plt.show()

# %%
# while D.mean() > np.log(size) + 1: # D.max() > size ** (1/3) * 3//2 :  # D_max_rrg or D.mean() > D_mean_rrg:
while len(extra_bonds)/4 < n_original_bonds * 0.1 : # D.max() > size ** (1/3) * 3//2 :  # D_max_rrg or D.mean() > D_mean_rrg:
    c = 0
    max_dist = np.where(D == D.max())
    new_bond = False
    while new_bond == False:
        for k in range(len(max_dist[0])):
            i, j = max_dist[0][k], max_dist[1][k]
            node_0, node_1 = list(G)[i], list(G)[j]
            # if np.abs(node_0[0]-node_1[0]) == np.abs(node_0[1]-node_1[1])or node_0[0] == node_1[0] or node_0[1] == node_1[1]: #Vertical and diagonal bonds
            if np.abs(node_0[0]-node_1[0]) == np.abs(node_0[1]-node_1[1]): # Only diagonal bonds
            # if node_0[0] == node_1[0] or node_0[1] == node_1[1]: # Only vertical bonds
            # if True: # Only vertical bonds
                print(node_0, node_1)
                print(c)
                new_bond = True
                break
        c += 1
        max_dist = np.where(D == D.max() - c)

    new_bonds = [(list(G)[i], list(G)[j]),
                 (tuple(np.abs(np.subtract((0, n), list(G)[i]))), tuple(np.abs(np.subtract((0, n), list(G)[j])))),
                 (tuple(np.abs(np.subtract((n, 0), list(G)[i]))), tuple(np.abs(np.subtract((n, 0), list(G)[j])))),
                 (tuple(np.abs(np.subtract((n, n), list(G)[i]))), tuple(np.abs(np.subtract((n, n), list(G)[j]))))]
    extra_bonds.extend(new_bonds)

    for i in range(1):
        G.add_edge(new_bonds[i][0], new_bonds[i][1])
        G_new_bonds.add_edge(new_bonds[i][0], new_bonds[i][1])

    D = shortest_path(nx.adjacency_matrix(G)).astype('int64')
    # plt.figure(dpi=100, figsize=[10, 10])
    # nx.draw(G_new_bonds, pos=pos, node_size=50)
    # plt.show()

plt.figure(dpi=100, figsize=[10, 10])
nx.draw(G_new_bonds, pos=pos, node_size=50)
# nx.draw(G, pos=pos, node_size=50)
plt.suptitle(f'original_bonds = {n_original_bonds}, n_extra_bonds = {len(G_new_bonds.edges)}, D_max = {D.max()}, D_mean = {D.mean().round(3)}')
plt.show()

# %% Erdos Reny 2/6
size = 36  # 30 ** 2 * 2

# p = 1 / size
p = 1 * np.log(size) / size
G = nx.fast_gnp_random_graph(size, p)
print(nx.number_connected_components(G))
print(diameter(G))
print(np.array(list(G.degree()))[:, 1].mean())
D = shortest_path(nx.adjacency_matrix(G)).astype('int16')

# %% Barabasi Albert 3/6
size = 1000
G = nx.barabasi_albert_graph(size, 3)
D = shortest_path(nx.adjacency_matrix(G)).astype('int16')
np.any(D == np.inf)
l_max = np.max(D)
print(l_max)

# %% Real world network 4/6
G = nx.read_adjlist('Data/c_elegans_undir.net', comments='#')
# G = nx.read_adjlist('Data/NDwww.net', comments='#')
A = np.array(nx.adjacency_matrix(G).todense())
size = len(A)

# %% 5/6
D = shortest_path(csr_matrix(A)).astype('int64')
l_max = np.max(D)
print(dimension(D))
print(l_max)

# %% Plot matrices A and D 6/6
fig, (ax1, ax2) = plt.subplots(ncols=2, dpi=150, figsize=(10, 5))

plot_d = ax2.matshow(D)
ax1.matshow(nx.adjacency_matrix(G).todense())
fig.colorbar(plot_d)
fig.suptitle(f'{adjacency}, n={size}, d_max={np.max(D)}, d_mean={np.round(np.mean(D), 3)}')
fig.tight_layout()
plt.show()

# %% l max and l mean vs size for one adjacency 1/3
adjacency = 'random_regular_5'
distribution = 'ising'
periodic = True
add = 0

sizes = np.geomspace(100, 2000, 10).astype('int32')
l_maxs = np.zeros_like(sizes)
l_means = np.zeros_like(sizes)

As = [sg.connectivity_matrix(size, adjacency, distribution, sparse=False, add=add, periodic=periodic).astype('int8') for
      size in sizes]
# %% 2/3
for i, A in enumerate(As):
    D = shortest_path(csr_matrix(A), method='FW').astype('int64')
    l_means[i] = np.mean(D)
    l_maxs[i] = np.max(D)

# %% 3/3
# plt.plot(sizes, l_means, 'x-')
plt.plot(sizes, l_maxs, '.-')
plt.xscale('log')
plt.show()
# %% l_max vs size for multiple parameters 1/2
periodic = True
distribution = 'ising'

adjacencies = ['2D', '3D', 'chimera', 'pegasus', 'random_regular_3', '2D_small_world']
sizes_adj = [np.linspace(20, 30, 20).astype('int') ** 2,
             np.linspace(7, 12, 11).astype('int') ** 3,
             np.arange(6, 12),
             np.arange(4, 10),
             np.linspace(20, 30, 20).astype('int') ** 2,
             np.linspace(20, 30, 20).astype('int') ** 2]
adds = [0, 0, 0, 0, 0, 0]

l_max_size_adj = []
l_mean_size_adj = []
real_sizes_adj = []

for adjacency, sizes, add in zip(adjacencies, sizes_adj, adds):
    l_max_size = []
    l_mean_size = []
    real_sizes = []
    for size in sizes:
        print(adjacency, size)
        A = sg.connectivity_matrix(size, adjacency, distribution, sparse=False, add=add, periodic=periodic).astype(
            'int8')
        D = shortest_path(A, directed=False, unweighted=True).astype('int64')
        real_sizes.append(A.shape[0])
        l_max_size.append(np.max(D))
        l_mean_size.append(np.mean(D))

    real_sizes_adj.append(real_sizes)
    l_max_size_adj.append(l_max_size)
    l_mean_size_adj.append(l_mean_size)


# %% 2/2
# def power_fit(x, a, x0):
#     return (x+x0)**(1/a)
def power_fit(x, a):
    return (x) ** (1 / a)


def log_fit(x, b):
    return b * np.log(x)


x = real_sizes
y = l_mean_size
curve_fit(log_fit, x, y)
# %%
fig, ax = plt.subplots(dpi=150, figsize=(16 / 1.5, 9 / 1.5))

for adjacency, real_sizes, l_max_size, l_mean_size, add in zip(adjacencies, real_sizes_adj, l_max_size_adj,
                                                               l_mean_size_adj, adds):

    x = real_sizes
    y = l_mean_size

    # p = np.polyfit(np.log(x), y, 2)
    print(f'{adjacency}')
    # print(f'power fit, param = {curve_fit(power_fit, x, y)[0][0]}, error = {curve_fit(power_fit, x, y)[1][0][0]}')
    # print(f'log fit, param = {curve_fit(log_fit, x, y)[0][0]}, error = {curve_fit(log_fit, x, y)[1][0][0]}')
    # x = np.log(x)
    if add != 0:
        ax.plot(x, y, '.-', label=f'{adjacency}, add={add}', markersize=10)
    else:
        ax.plot(x, y, '.-', label=f'{adjacency}', markersize=10)
    # ax.plot(x, p[0] * x ** 2 + p[1] * x + p[2], 'k', alpha=0.8, linewidth=0.5)

ax.legend()
# ax.set_xscale('log')
# ax.set_yscale('log')
fig.show()

# %% Estimate dimension of network for size in sizes or add in adds 1/4
adjacency = '2D'
# adjacency = '3'
# sizes = np.arange(10, 46, 2)       # hexagonal
# sizes = np.arange(5, 11) ** 3   # 3D
sizes = np.arange(10, 34, 2) ** 2   # 2D
# sizes = np.array([5, 10, 15])  # chimera
# sizes = np.array([4, 8, 12])  # pegasus

# sizes = np.geomspace(100, 1000, 6).astype('int')
# sizes_fit = np.append(sizes, 1e10)
size = 1000

bonds = 'both'
adds = np.geomspace(2, 3, 10)
periodic = False
add = 0
distribution = 'ising'
ds = []
errs = []
real_sizes = []
threads = 8
N_average = threads * 1

l_n = [[] for _ in sizes]

As = [
    [sg.connectivity_matrix(size, adjacency, distribution, sparse=False, add=add, periodic=periodic, bonds=bonds).astype('int8') for
     _ in range(N_average)] for size in sizes]
# %% 2/4
for k, A in enumerate(tqdm(As)):
    l_n[k] = Parallel(n_jobs=threads)(
        delayed(compact_box_counting_numba)(
            shortest_path(
                csr_matrix(A[i])
            ).astype('int64')
        )
        for i in range(N_average))

# %% 3/4
l_vs_size = []
n_vs_size = []
ls_vs_size = []
ns_vs_size = []

for k in range(len(sizes)):
    lens = [len(l_n[k][i][1]) for i in range(N_average)]
    max_len = np.array(lens).max()
    ns = [l_n[k][i][1] / l_n[k][i][1].max() for i in range(N_average)]
    ls = [l_n[k][i][0] for i in range(N_average)]
    l = ls[np.where(lens == max_len)[0][0]]
    for i in range(N_average):
        while len(ns[i]) < max_len:
            ns[i] = np.append(ns[i], np.nan)
            ls[i] = np.append(ls[i], np.nan)
    n = np.nanmean(ns, 0)
    l_vs_size.append(l)
    n_vs_size.append(n)
    ls_vs_size.append(ls)
    ns_vs_size.append(ns)

l_max = l.max()

# %% 4/4
fig, ax = plt.subplots(dpi=150)

for k, (l, n, ls, ns) in enumerate(zip(l_vs_size, n_vs_size, ls_vs_size, ns_vs_size)):
    # plt.plot(ls / l_max, ns, '.', color=plt.cm.plasma(255 * k // len(sizes)), markersize=2, alpha=0.5)
    plt.plot(l / l_max, n, '.-', color=plt.cm.plasma(255 * k // len(sizes)), label=f'{sizes[k]}', markersize=2, linewidth=1)
    # plt.plot(l, n, '.-', color=plt.cm.plasma(255 * k // len(sizes)), label=f'{sizes[k]}', markersize=2, linewidth=1)

# p = np.polyfit(np.log(l / l_max)[:-10], np.log(n)[:-10], 1)
# ax.text(0.2, 0.2, f'{p[0].round(4)}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
ax.set_xlabel('box size')
ax.set_ylabel('Number of boxes')
ax.set_xlim([1e-2, 1.01])
ax.set_ylim([1e-3, 1.01])
ax.set_title(f'{adjacency},add={add}')
plt.yscale('log')
plt.xscale('log')
ax.legend(loc='lower left')
# plt.savefig(f'{adjacency},n={sizes}.pdf')
fig.show()

# %% Fit
def objective_function(x, a, b):
    return a + b * x


error = 1000
for r0 in np.linspace(0, -2, 100):
    a0, b0 = curve_fit(objective_function, sizes ** r0, ds)[0]
    if np.abs(objective_function(sizes ** r0, a0, b0) - ds).sum() < error:
        error = np.abs(objective_function(sizes ** r0, a0, b0) - ds).sum()
        a = a0.copy()
        b = b0.copy()
        r = r0.copy()
print(a, b, r, error)

plt.errorbar(sizes ** r, ds, yerr=errs)
# plt.plot(sizes_fit ** r, objective_function(sizes_fit ** r, a, b), '.-')
plt.title(f'{adjacency}, n={size}')
plt.show()


#%% Scaling of n_extra_bonds in 2D_small_world  with system size
adjacency = '2D_small_world'
Ls = np.arange(5, 30, 2)
sizes = Ls ** 2
bonds = 'diagonal'
periodic = False
distribution = 'ising'

n_extra_bonds_vs_size = []

for size in sizes:
    n_extra_bonds_vs_size.append(
        sg.connectivity_matrix(size, adjacency, distribution, sparse=False, add=0, periodic=periodic, bonds=bonds)[1]
    )

# for L, n in zip(Ls, n_extra_bonds_vs_size):
#     print(L, n)

plt.plot(sizes, sizes)
plt.plot(sizes,n_extra_bonds_vs_size)
plt.show()

#%%

# # %% l vs n_boxes
# ls, n_boxes_vs_l = box_counting_numba(D)
# # ls, n_boxes_vs_l = compact_box_counting_numba(D)
#
# # %%
# ls, nodes_at_l = dist_from_node(D)
# n_boxes_vs_l = 1 / nodes_at_l
#
# # %% enhance test
# ls, n_boxes_vs_l = ls.astype('float'), n_boxes_vs_l.astype('float')
# for i in range(len(ls)):
#     if i != 0 and n <= n_boxes_vs_l[i]:
#         ls[i] = np.nan
#         n_boxes_vs_l[i] = np.nan
#     else:
#         n = n_boxes_vs_l[i]
#
# ls = ls[~np.isnan(ls)].astype('int64')
# n_boxes_vs_l = n_boxes_vs_l[~np.isnan(n_boxes_vs_l)].astype('int64')
#
# # %%
# ls, n_boxes_vs_l = compact_box_counting_numba(D)
#
# print(np.polyfit(-np.log(ls), np.log(n_boxes_vs_l), 1)[0])
# print(np.polyfit(-np.log(ls[1:]), np.log(n_boxes_vs_l[1:]), 1)[0])
# # print(np.polyfit(-np.log(ls[5:]), np.log(n_boxes_vs_l[5:]), 1)[0])
# print(np.polyfit(-np.log(ls[1:-1]), np.log(n_boxes_vs_l[1:-1]), 1)[0])
# # print(np.polyfit(-np.log(ls[5:-5]), np.log(n_boxes_vs_l[5:-5]), 1)[0])
# # print(np.polyfit(-np.log(ls[:-5]), np.log(n_boxes_vs_l[:-5]), 1)[0])
# print(np.polyfit(-np.log(ls[:-1]), np.log(n_boxes_vs_l[:-1]), 1)[0])
#
# # %%
# plt.plot(ls, n_boxes_vs_l / n_boxes_vs_l.max())
# plt.yscale('log')
# plt.xscale('log')
# plt.show()
#
# # %%
# size = 100  # 30 ** 2 * 2
# adjacency = '1D+'
# add = 0
# distribution = 'ising'
# periodic = True
# l = []
# n = []
# for i in range(10):
#     A = sg.connectivity_matrix(size, adjacency, distribution, sparse=False, add=add, periodic=periodic).astype('int8')
#     D = shortest_path(csr_matrix(A)).astype('int64')
#     li, ni = compact_box_counting_numba(D)
#     l.append(li)
#     n.append(ni)
#
# print(-np.polyfit(np.log10(np.array(l).flatten())[1:], np.log10((np.array(n).flatten()))[1:], 1)[0])
# print(-np.polyfit(np.log10(li), np.log10(np.min(np.array(n), 0)), 1)[0])
# m = np.polyfit(np.log10(np.array(l).flatten()), np.log10((np.array(n).flatten())), 1)[0]
# k = np.polyfit(np.log10(np.array(l).flatten()), np.log10((np.array(n).flatten())), 1)[1]
# x = np.log10(np.array(l).flatten())
# plt.plot(np.log10(np.array(l).flatten()), np.log10((np.array(n).flatten())), '.')
# plt.plot(np.log10(np.min(np.array(l), 0)), np.log10(np.min(np.array(n), 0)), '.r')
# plt.plot(x, m * x + k)
# # plt.ylim([0,2])
# plt.show()
#
# # %%
# N = 1000
# lf, nf = dimension_numba_fast(D)
# ls, ns = dimension_numba(D)
# n_fast = np.zeros([len(ls), N])
# n_slow = np.zeros([len(ls), N])
#
# for i in range(N):
#     n_fast[:, i] = dimension_numba_fast(D)[1]
#     n_slow[:, i] = dimension_numba(D)[1]
#
# print(np.std(n_fast, 1))
# print(np.std(n_slow, 1))
#
# print(np.min(n_fast, 1))
# print(np.min(n_slow, 1))
