# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as sps
import Modules.read_data_from_cluster as rfc
import Modules.pade_fits as pf
import Modules.figures as figs
import sys
import importlib
import itertools
from joblib import Parallel, delayed, cpu_count
from scipy.optimize import curve_fit

plt.rcParams['font.size'] = '16'
plt.rcParams['figure.dpi'] = '200'
plt.rcParams['backend'] = 'QtAgg'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
   # "font.sans-serif": "Helvetica",
})


importlib.reload(rfc)
importlib.reload(pf)
importlib.reload(figs)


# %% Parameters
distribution = 'gaussian_EA'
adjacencies = ['random_regular_3', 'random_regular_5', 'random_regular_7', 'random_regular_9', '1D+', '1D+',
               '1D+', '1D+', '2D_small_world', 'chimera', 'pegasus', 'zephyr']
sizes = [100, 200, 400, 800, 1200, 1600, 3200]
# sizes = [400, 800, 1600, 3200]
# sizes = [400, 800, 1200, 1600]
add_vs_adj = [0, 0, 0, 0, 3.0, 5.0, 7.0, 9.0, 0, 0, 0, 0]
T0_Tf_vs_adj = [[0.2, 1.5], [0.5, 3.0], [0.5, 3.0], [1.0, 4.0], [0.2, 1.5], [1.0, 2.5],
                [1.3, 3.0], [1.5, 3.5], [0.5, 2.5], [0.2, 3.0], [0.2, 4.0], [0.5, 5.0]]
# [0.5, 2.5]
MCS_avg_0 = 10000

ic_jc_vs_adj = [[[3, 6, 6, 3, 6], [6, 4, 4, 6, 6]],
                [[5, 5, 4, 6, 4], [4, 5, 5, 5, 6]],
                [[5, 4, 5, 6, 6], [3, 6, 4, 3, 5]],
                [[5, 5, 6, 3, 5], [3, 4, 3, 5, 5]],
                [[6, 5, 5, 5, 6], [3, 3, 3, 4, 4]],
                [[4, 4, 4, 5, 5], [5, 5, 4, 4, 4]],
                [[4, 6, 6, 4, 4], [5, 5, 5, 4, 4]],
                [[4, 4, 4, 4, 4], [5, 4, 4, 4, 4]],
                [[5], [6]],
                [[3, 4, 3, 6, 6, 6], [4, 5, 5, 5, 5, 6]],
                [[4, 4, 4, 4, 4, 3], [5, 3, 4, 4, 4, 6]],
                [[3, 3, 5, 5, 5, 4], [4, 4, 3, 3, 3, 5]]]

max_MCSs_vs_adj_binned = np.array([[3, 3, 4, 6, 0, 6, 7],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [3, 3, 4, 6, 0, 6, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [5, 6, 7, 8, 0, 8, 0],
                                   [5, 6, 7, 8, 0, 8, 0],
                                   [5, 6, 7, 8, 0, 8, 8]])
max_MCSs_vs_adj_binned = MCS_avg_0 * 2 **  max_MCSs_vs_adj_binned

max_MCSs_vs_adj_fast = np.array([[0, 0, 6, 7, 0, 9, 0],
                                 [0, 0, 6, 7, 0, 9, 0],
                                 [4, 5, 6, 7, 0, 8, 0],
                                 [4, 5, 6, 7, 0, 8, 0],
                                 [4, 5, 6, 7, 0, 8, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 6, 7, 8, 9, 0],
                                 [3, 3, 6, 7, 8, 9, 7],
                                 [3, 3, 6, 7, 8, 9, 7]])

max_MCSs_vs_adj_fast = MCS_avg_0 * 2 **  max_MCSs_vs_adj_fast

max_MCSs_vs_adj_old = [MCS_avg_0 * 2 ** np.array([1, 2, 5, 6, 0, 7, 7]),
                       MCS_avg_0 * 2 ** np.array([5, 5, 5, 5, 0, 5, 5]) + 4,
                       # MCS_avg_0 * 2 ** np.array([1, 2, 4, 4, 0, 5, 6]) + 1,  # test for rrg7, this should be off for analysis
                       # MCS_avg_0 * 2 ** np.array([3, 3, 4, 5, 0, 6, 0]) + 2,  # test for rrg7, this should be off for analysis
                       np.array([80002, 80002, 160002, 640001, 0, 640002, 0 ]),
                       MCS_avg_0 * 2 ** np.array([1, 2, 4, 5, 0, 5, 6]),
                       MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 0, 5, 6]),
                       # MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 0, 5, 6]),
                       MCS_avg_0 * 2 ** np.array([3, 3, 4, 5, 0, 6, 0]) + 1,  # test for 1D+5, this should be off for analysis
                       MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 0, 5, 6]),
                       MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 0, 5, 6]),
                       MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 0, 5, 6]),
                       MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 0, 5, 6]),
                       MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 0, 5, 6]),
                       MCS_avg_0 * 2 ** np.array([0, 0, 0, 0, 0, 0, 0])]

adj_index = 0

adjacency = adjacencies[adj_index]
T0 = T0_Tf_vs_adj[adj_index][0]
Tf = T0_Tf_vs_adj[adj_index][1]
add = add_vs_adj[adj_index]
max_MCS_vs_size_binned = max_MCSs_vs_adj_binned[adj_index]
max_MCS_vs_size_old = max_MCSs_vs_adj_old[adj_index]
max_MCS_vs_size_fast = max_MCSs_vs_adj_fast[adj_index]

max_MCS_vs_size_binned = max_MCS_vs_size_binned[:-1]
max_MCS_vs_size_old =       max_MCS_vs_size_old[:-1]
max_MCS_vs_size_fast =     max_MCS_vs_size_fast[:-1]
max_MCS_vs_size = [max_MCS_vs_size_binned, max_MCS_vs_size_old, max_MCS_vs_size_fast]

sizes_vs_adj = [_ for _ in range(len(adjacencies))]
for i, adj in enumerate(adjacencies):
    if adj == 'chimera':
        sizes_vs_adj[i] = [72, 200, 392, 800, 1152, 1568, 3200]
    elif adj == 'pegasus':
        sizes_vs_adj[i] = [128, 256, 448, 960, 1288, 1664, 3648]
    elif adj == 'zephyr':
        sizes_vs_adj[i] = [48, 160, 336, 576, 880, 1248, 2736]
    else:
        sizes_vs_adj[i] = sizes

sizes_vs_adj =                     [ siz_vs_adj[:-1] for siz_vs_adj in sizes_vs_adj]
sizes =                                   sizes[:-1]

n_q_dist = 50
only_max_MCS = True
n_bootstrap = 36*10
# Define temperatures for the plots
n_temps = 10
T_index_0 = 0

data_type = 'all'
MCS_N_config_condition = 'max_MCS_with_minimum_N_configs'
min_N_config = 200
colors_sizes = ['turquoise',  'tab:olive', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray','tab:blue', 'goldenrod', 'tab:orange', 'tab:red']
marker_adjacencies = ['^', '>', 'v', '<', '1', '2', '3', '.', '4', 'P', 'd', '*']

# %% Read data
MCS_avg_vs_size, N_configs_vs_size, copies_vs_size, labels_vs_size, T_vs_size, q2_vs_size, q4_vs_size, \
    ql_vs_size, U_vs_size, U2_vs_size, σ2_q2_bin_vs_size, σ2_q4_bin_vs_size, q_dist_vs_size, g_vs_size, \
    g_bootstrap_vs_size, error_vs_size,dg_dT_vs_size, dg_dT_bootstrap_vs_size, error_dg_dT_vs_size = \
    rfc.read_data(adjacency, distribution, sizes, add, T0, Tf, MCS_avg_0, max_MCS_vs_size, data_type, only_max_MCS=only_max_MCS)

# Define temperatures for the plots
copies = copies_vs_size[0][0]
T_indices = np.linspace(T_index_0, copies-1, n_temps, dtype='int')
colors = plt.get_cmap('plasma')(np.linspace(0, 255, n_temps).astype('int'))

# %% Calculate Binder cumulant with errors
n_cases = sum([len(MCS_avg) for MCS_avg in MCS_avg_vs_size])
results_g_parallel = Parallel(n_jobs=min(cpu_count(), n_cases))(delayed(rfc.binder_cumulant_and_error_bootstrap)
                                                                (T_vs_size[size_ind][MCS_ind],
                                                                 q2_vs_size[size_ind][MCS_ind],
                                                                 q4_vs_size[size_ind][MCS_ind],
                                                                 n_bootstrap=n_bootstrap, error_type='1')
                                                                for size_ind in range(len(sizes))
                                                                for MCS_ind in range(len(MCS_avg_vs_size[size_ind])))

k = 0
for size_ind in range(len(sizes)):
    for MCS_ind in range(len(MCS_avg_vs_size[size_ind])):
        g_vs_size[size_ind][MCS_ind], g_bootstrap_vs_size[size_ind][MCS_ind], error_vs_size[size_ind][MCS_ind],\
        dg_dT_vs_size[size_ind][MCS_ind], dg_dT_bootstrap_vs_size[size_ind][MCS_ind], error_dg_dT_vs_size[size_ind][MCS_ind] = results_g_parallel[k]
        k += 1
del results_g_parallel

# %% Choose a MCS for each size
T_vs_size_best, g_vs_size_best, error_vs_size_best, dg_dT_vs_size_best, dg_dT_bootstrap_vs_size_best, \
error_dg_dT_vs_size_best = [ [[] for _ in range(len(sizes))] for _ in range(6)]
dead_index = None
for size_index in range(len(sizes)):
    assigned = False
    k = -1
    try:
        while not assigned:
            match MCS_N_config_condition:
                case 'max_MCS_with_minimum_N_configs':
                    if N_configs_vs_size[size_index][k] > min_N_config:
                        assigned = True
                    else:
                        k -= 1
                case 'max_N_configs':
                    if N_configs_vs_size[size_index][k] == max(N_configs_vs_size[size_index]) and N_configs_vs_size[size_index][k] >0:
                        assigned = True
                    else:
                        k -= 1
        print(sizes[size_index], k)
        T_vs_size_best[size_index] = T_vs_size[size_index][k]
        g_vs_size_best[size_index] = g_vs_size[size_index][k]
        error_vs_size_best[size_index] = error_vs_size[size_index][k]
        dg_dT_vs_size_best[size_index] = dg_dT_vs_size[size_index][k]
        dg_dT_bootstrap_vs_size_best[size_index] = dg_dT_bootstrap_vs_size[size_index][k]
        error_dg_dT_vs_size_best[size_index] = error_dg_dT_vs_size[size_index][k]
    except:
        dead_index = size_index
        continue

if dead_index is not None:
    del T_vs_size_best[dead_index], g_vs_size_best[dead_index],  error_vs_size_best[dead_index], dg_dT_vs_size_best[dead_index], \
        dg_dT_bootstrap_vs_size_best[dead_index], error_dg_dT_vs_size_best[dead_index], sizes[dead_index], sizes_vs_adj[adj_index][dead_index]
# %% Plot Binder cumulant with errors
alpha =3
Tc = 0.2
Tc_vs_size = [1.13, 1.08, 1.05, 0.98, 0.95, 0.89]
Tc_vs_size = [2.277, 2.316, 2.331, 2.186, 2.157, 2.143]
Tc_vs_size = [0.75]*len(sizes)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=[16, 9])
for size_index, (T, g, err) in enumerate(zip(T_vs_size_best, g_vs_size_best, error_vs_size_best)):
    ax1.errorbar(T, g, yerr=err,
                 label=f'{sizes_vs_adj[adj_index][size_index]}', color=colors_sizes[size_index], markersize=2, capsize=2, capthick=0.5, elinewidth=1, linewidth=0.5)
    if size_index > 2:
        ax2.errorbar((T-Tc_vs_size[size_index])*np.sqrt(np.array(sizes_vs_adj[adj_index][size_index]))**(1/alpha), g,
                     yerr=err, label=f'{sizes_vs_adj[adj_index][size_index]}', color=colors_sizes[size_index], markersize=2, capsize=2, capthick=0.5, elinewidth=1, linewidth=0.5)
ax2.legend()
ax2.set_yscale('log')
ax2.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax2.set_yticks([0.25, 0.5, 1])
ax2.set_ylim([0.25, 1])
ax2.set_xlim([-2, 1])
ax2.set_ylabel('$g$')
ax2.set_xlabel(r'$TN^{1/\alpha}$')
ax2.set_title(rf'$\alpha=${alpha}')


ax1.set_ylabel('$g$')
ax1.set_xlabel('$T$')
# ax1.set_ylim([0.125,1])
ax1.set_xlim([0, Tf])

# fig.tight_layout()
fig.show()

# %% Plot numerical derivative of Binder cumulant
fig, ax = plt.subplots()
for size_index, (T, dg_dT, error_dg_dT) in enumerate(zip(T_vs_size_best, dg_dT_vs_size_best, error_dg_dT_vs_size_best)):
    ax.errorbar(T, -dg_dT, yerr=error_dg_dT, label=f'{sizes[size_index]}', color=colors_sizes[size_index],
                markersize=2, capsize=2, capthick=1, elinewidth=1, linewidth=0.5)
fig.show()
# %% BINNED DATA ONLY Plot Autocorrelation time and thermal average error of q2
error_q2_T_vs_size = np.zeros([len(sizes), copies])
tau_q2_T_vs_size = np.zeros([len(sizes), copies])

for size_index, size in enumerate(sizes):
    if size_index == 4:
        continue
    if size_index !=5:
        continue
    σ2_q2_bin_t = σ2_q2_bin_vs_size[size_index][-1]
    MCS_avg = MCS_avg_vs_size[size_index][-1]

    T_indices = np.linspace(T_index_0, copies - 1, σ2_q2_bin_t.shape[-1], dtype='int')
    colors = plt.get_cmap('plasma')(np.linspace(0, 255, σ2_q2_bin_t.shape[-1]).astype('int'))

    fig, (ax2, ax1) = plt.subplots(ncols=2, figsize=[10, 4], dpi=150)

    for T_index in range(σ2_q2_bin_t.shape[-1]):
        σ2_q2_bin_c = σ2_q2_bin_t.mean(0)[:, T_index]
        # σ2_q2_bin_c = σ2_q2_bin_t[10,:,T_index]
        M0 = MCS_avg
        n_bins = σ2_q2_bin_c.shape[0]
        bins = np.arange(n_bins)

        # Remove the last three points
        bins = bins
        σ2_q2_bin_c = σ2_q2_bin_c
        M_bin = [M0 / 2 ** bin for bin in bins]
        error = np.array([np.sqrt((1 / M) * sigma) for M, sigma in zip(M_bin, σ2_q2_bin_c)])

        M = 2 ** bins
        tau = M * σ2_q2_bin_c[bins] / σ2_q2_bin_c[0]
        tau_max_index = np.where(tau == tau.max())[0][0]
        if np.abs(tau[tau_max_index]-tau[tau_max_index-1])/tau[tau_max_index] < 0.1:
            tau_q2_T_vs_size[size_index, T_index] = tau.max()
        else:
            tau_q2_T_vs_size[size_index, T_index] = np.nan

        # n_bin_bias = np.arange(n_bins - 1)
        # M_bias = 2 ** n_bin_bias
        # tau_bias = (4 * M_bias * σ2_q2_bin_c[n_bin_bias + 1] - M_bias * σ2_q2_bin_c[n_bin_bias]) / σ2_q2_bin_c[0]

        if np.any(T_index == T_indices):
            ax1.plot(2 ** bins, error, '.-', color=colors[np.where(T_index == T_indices)[0][0]])
            ax2.plot(M, tau, '.-', color=colors[np.where(T_index == T_indices)[0][0]])
            # ax3.plot(M_bias, tau_bias,'.-', color=colors[np.where(T_index == T_indices)[0][0]])

    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_title(r'$\overline{\Delta_{q^2}}$')
    ax1.set_xlabel(r'$M_n$')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title(r'$\overline{\tau_{q^2}}$')
    ax2.set_xlabel(r'$M_n$')

    figs.colorbar_for_lines(fig, T_vs_size[size_index][0][T_indices].round(2), label='$T$', location='top')

    fig.suptitle(f'$n=$ {size}')
    fig.tight_layout()
    fig.show()

# %% BINNED DATA ONLY - NEW VERSION - Plot Autocorrelation time and thermal average error of q2 -
size_index = 5
σ2_q2_bin_t = σ2_q2_bin_vs_size[size_index][-1]
MCS_avg = MCS_avg_vs_size[size_index][-1]
tau_q2_T_vs_configs = np.zeros_like(σ2_q2_bin_t)
tau_q2_T_vs_configs_mode = np.zeros([σ2_q2_bin_t.shape[1],σ2_q2_bin_t.shape[2]])

fig, ax2 = plt.subplots()

for config_index in range(N_configs_vs_size[size_index][-1]):
    for T_index in range(σ2_q2_bin_t.shape[-1]):
        σ2_q2_bin_c = σ2_q2_bin_t[config_index, :, T_index]
        n_bins = σ2_q2_bin_c.shape[0]
        bins = np.arange(n_bins)
        M = 2 ** bins
        tau_q2_T_vs_configs[config_index,:,T_index] = M * σ2_q2_bin_c[bins] / σ2_q2_bin_c[0]

for bin_index in range(σ2_q2_bin_t.shape[1]):
    for T_index in range(σ2_q2_bin_t.shape[-1]):
        # h, b = np.histogram(np.log10(tau_q2_T_vs_configs[:, bin_index, T_index]), 1000)
        # b_x_index = np.where(h==h.max())[0][0]
        # tau_q2_T_vs_configs_mode[bin_index,T_index] = 0.5*( b[b_x_index] + b[b_x_index+1])
        mean = tau_q2_T_vs_configs[:, bin_index, T_index].mean()
        std = tau_q2_T_vs_configs[:, bin_index, T_index].std()
        # tau_q2_T_vs_configs[:, bin_index, T_index][tau_q2_T_vs_configs[:, bin_index, T_index]>mean+4*std] = np.nan

for T_index in range(σ2_q2_bin_t.shape[-1]):
    if np.any(T_index == T_indices):
        ax2.plot(M, np.nanmean(tau_q2_T_vs_configs[:,:,T_index],0) , '.-', color=colors[np.where(T_index == T_indices)[0][0]])


ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title(r'$\overline{\tau_{q^2}}$')
ax2.set_xlabel(r'$M_n$')

figs.colorbar_for_lines(fig, T_vs_size[size_index][0][T_indices].round(2), label='$T$', location='top')

fig.suptitle(f'$n=$ {sizes[size_index]}')
fig.tight_layout()
fig.show()

#%%
T_index = -30
fig, ax = plt.subplots()

ax.hist(tau_q2_T_vs_configs[:,-4,T_index],100)
fig.suptitle(f'T={T_vs_size[size_index][-1][T_index]}')
# ax.set_xscale('log')
# ax.set_yscale('log')
fig.show()


# %% BINNED DATA ONLY Read autocorrelation time data vs adjacency
T_vs_adj = [[] for _ in range(len(adjacencies))]
tau_q2_T_vs_adj = [[] for _ in range(len(adjacencies))]
σ2_q2_bin_vs_adj = [[] for _ in range(len(adjacencies))]
MCS_avg_vs_size = [[] for _ in range(len(adjacencies))]
N_configs_vs_size = [[] for _ in range(len(adjacencies))]
adj_indices = [0, 9, 10, 11]

for adj_index in adj_indices:
    MCS_avg_vs_size[adj_index], N_configs_vs_size[adj_index], _, _, T_vs_adj[adj_index], _, _, _, _, _, σ2_q2_bin_vs_adj[adj_index] = \
        rfc.read_data(adjacencies[adj_index], distribution, sizes, add_vs_adj[adj_index], T0_Tf_vs_adj[adj_index][0],
        T0_Tf_vs_adj[adj_index][1], MCS_avg_0, [max_MCSs_vs_adj_binned[adj_index][:-1], _, _], data_type='binned')[:11]

for adj_index in adj_indices:
    del T_vs_adj[adj_index][4], σ2_q2_bin_vs_adj[adj_index][4], sizes_vs_adj[adj_index][4]
del sizes[4]

# %% BINNED DATA ONLY Calculate autocorrelation time

for adj_index in adj_indices:
    tau_q2_T_vs_adj[adj_index] = rfc.autocorrelation_time_q2(σ2_q2_bin_vs_adj[adj_index])

# %% BINNED DATA ONLY Calculate autocorrelation time fits
T_fit_tau_vs_adj, log_tau_fit_vs_adj, log_tau_vs_adj, T_tau_vs_adj = [ [[[] for _ in range(len(sizes))] for _ in range(len(adjacencies))] for _ in range(4)]
divide_by_size = True

for adj_index in adj_indices:
    for size_index in range(len(sizes)):
        fit_start_index = np.where(np.isfinite(tau_q2_T_vs_adj[adj_index][size_index]))[0][0]

        T = T_vs_adj[adj_index][size_index][-1][fit_start_index:]
        T_tau_vs_adj[adj_index][size_index] = T

        if divide_by_size:
            tau = tau_q2_T_vs_adj[adj_index][size_index][fit_start_index:]/sizes_vs_adj[adj_index][size_index]
        else:
            tau = tau_q2_T_vs_adj[adj_index][size_index][fit_start_index:]

        log_tau_vs_adj[adj_index][size_index] = np.log(tau)

        T_fit_tau_vs_adj[adj_index][size_index] = np.linspace(T[0], T[-1], 1000)
        log_tau_fit_vs_adj[adj_index][size_index] = np.poly1d(np.polyfit(T, log_tau_vs_adj[adj_index][size_index], 10))(T_fit_tau_vs_adj[adj_index][size_index])

# %% BINNED DATA ONLY Plot autocorrelation time
adjacency_names = ['$(a)$', '$(b)$', '$(c)$', '$(d)$']
fig, ax = plt.subplots(ncols=4, figsize=[4*4*0.7, 4*1.2])
# ax = ax.ravel()
if len(adj_indices) == 1:
    fig, ax = plt.subplots()
    ax = [ax]
for i, adj_index in enumerate(adj_indices):
    for size_index in range(len(sizes)):
        if i==3 and size_index == 0:
            continue
        ax[i].plot(T_tau_vs_adj[adj_index][size_index][:-3], log_tau_vs_adj[adj_index][size_index][:-3],
                   color=colors_sizes[size_index], marker=marker_adjacencies[adj_index], linewidth=0, markersize=3, label=f'$N={sizes_vs_adj[adj_index][size_index]}$')
        ax[i].plot(T_fit_tau_vs_adj[adj_index][size_index][:-100], log_tau_fit_vs_adj[adj_index][size_index][:-100], color=colors_sizes[size_index], linewidth=0.5)

    ax[i].set_title(f'{adjacency_names[i]}', y=-.4)
    # ax[i].xaxis.set_label_coords(0.5, -0.05)
    # ax[i].xaxis.set_label_coords(1,-0.05)
    ax[i].set_xlabel('$T$')
    ax[i].set_ylim(-7, 2)
    ax[i].set_xlim(0,4.5)
    ax[i].set_xticks([0,1,2,3,4])
    if i > 0:
        ax[i].set_yticks([])
    ax[i].legend(fontsize=12)
    # ax[i].set_box_aspect(.8)

    # if divide_by_size:
    #     # ax[i].set_ylim([-7, 1])
    #     # ax[i].set_xlim([0.65, 1.3])
    #     ax[i].set_xlim([0, 5])
    #
    # else:
    #     # ax[i].set_ylim([-0.1, 3.5])
    #     # ax[i].set_xlim([0.65, 1.3])
    #     ax[i].set_xlim([0, 5])

    if divide_by_size:
        # ax[0].set_ylabel(r'$\log(\overline{\tau_{q^2}}/N)$')
        fig.suptitle(r'$\log(\overline{\tau_{q^2}}/N)$', y=0.92)
    else:
        ax[0].set_ylabel(r'$\log(\overline{\tau_{q^2}})$')
fig.tight_layout()
figs.export('Autocorrelation times by N for rrg3 chimera pegasus zephyr.pdf')
fig.show()

# %% BINNED DATA ONLY Calculate autocorrelation time vs size for different temperatures
Ts = np.linspace(0.5, 5, 500)
# Ts = np.linspace(0.65, 1.3, 100)
log_tau_vs_size_for_specific_T_vs_adj = [[[] for _ in range(len(Ts))] for _ in range(len(adjacencies))]

for adj_index in adj_indices:
    for size_index in range(len(sizes)):
        for T_fit_index, T in enumerate(Ts):
            if np.any(np.abs(T_fit_tau_vs_adj[adj_index][size_index]-T) < 0.1):
                T_index = np.where(np.abs(T_fit_tau_vs_adj[adj_index][size_index]-T) == np.abs(T_fit_tau_vs_adj[adj_index][size_index]-T).min())[0][0]
                log_tau_vs_size_for_specific_T_vs_adj[adj_index][T_fit_index].append(log_tau_fit_vs_adj[adj_index][size_index][T_index])
            else:
                log_tau_vs_size_for_specific_T_vs_adj[adj_index][T_fit_index].append(np.nan)

# %% Define scaling law for tau(T) vs N
def scaling_law_manuel(x, a, b, c):
     # scaling law for log(tau) = f(log(N))
     # This corresponds to tau = c * N^a * e^(bN)
    # return a*x*0 + b*np.exp(x) + c*0
    return a*x + b*np.exp(x)*0 + c

def scaling_law(x, a, b, c):
    return a*x*0 + b*np.exp(x) + c*0

# %% BINNED DATA ONLY Plot autocorrelation time vs size for different temperatures
fit_params_log_tau_vs_size_for_specific_T_vs_adj = [[[] for _ in range(len(Ts))] for _ in range(len(adjacencies))]

colors_autocorrelation = plt.get_cmap('plasma')(np.linspace(0, 255, len(Ts)).astype('int'))
fig, ax = plt.subplots(ncols=len(adj_indices), figsize=[26, 6])
if len(adj_indices) == 1:
    fig, ax = plt.subplots()
    ax = [ax]

for i, adj_index in enumerate(adj_indices):
    for T_fit_index, T in enumerate(Ts):

        log_tau_vs_size = log_tau_vs_size_for_specific_T_vs_adj[adj_index][T_fit_index]
        # Print in log log
        ax[i].plot(np.log(sizes_vs_adj[adj_index]), log_tau_vs_size, color=colors_autocorrelation[T_fit_index], linewidth=0, marker=marker_adjacencies[adj_index], markersize=3)
        # Print in linear
        # ax[i].plot(sizes_vs_adj[adj_index], np.exp(log_tau_vs_size), color=colors_autocorrelation[T_fit_index], linewidth=0, marker=marker_adjacencies[adj_index], markersize=3)
        try:
            non_thermalized_index = np.where(~np.isnan(log_tau_vs_size))[0][-1]

            # log(tau) = A + B*log(N)
            # params = np.polyfit(np.log(sizes_vs_adj[adj_index][:non_thermalized_index+1]), log_tau_vs_size[:non_thermalized_index+1], 1)
            # N0 = np.log(sizes_vs_adj[adj_index][:non_thermalized_index+1])[0]
            # Nf = np.log(sizes_vs_adj[adj_index][:non_thermalized_index+1])[-1]
            # sizes_fit = np.linspace(N0, Nf, 100)
            # ax[i].plot(sizes_fit, np.poly1d(params)(sizes_fit), color=colors_autocorrelation[T_fit_index], linewidth=1)

            # scaling law manuel
            # params = curve_fit(scaling_law_manuel, np.log(sizes_vs_adj[adj_index][:non_thermalized_index+1]), log_tau_vs_size[:non_thermalized_index+1] ,bounds=([0,-np.inf,0],[np.inf,np.inf,np.inf]))[0]
            params = curve_fit(scaling_law_manuel, np.log(sizes_vs_adj[adj_index][:non_thermalized_index+1]), log_tau_vs_size[:non_thermalized_index+1])[0]
            N0 = np.log(sizes_vs_adj[adj_index][:non_thermalized_index+1])[0]
            Nf = np.log(sizes_vs_adj[adj_index][:non_thermalized_index+1])[-1]
            sizes_fit = np.linspace(N0, Nf, 100)
            ax[i].plot(sizes_fit, scaling_law_manuel(sizes_fit, *params), color=colors_autocorrelation[T_fit_index], linewidth=1)

            #
            # params = curve_fit(scaling_law, sizes_vs_adj[adj_index][:non_thermalized_index+1], np.exp(log_tau_vs_size[:non_thermalized_index+1]))[0]
            # N0 = sizes_vs_adj[adj_index][:non_thermalized_index+1][0]
            # Nf = sizes_vs_adj[adj_index][:non_thermalized_index+1][-1]
            # sizes_fit = np.linspace(N0, Nf, 100)
            # ax[i].plot(np.log(sizes_fit), np.log(scaling_law(sizes_fit, *params)), color=colors_autocorrelation[T_fit_index], linewidth=1)

            fit_params_log_tau_vs_size_for_specific_T_vs_adj[adj_index][T_fit_index] = params.copy()
        except:
            fit_params_log_tau_vs_size_for_specific_T_vs_adj[adj_index][T_fit_index] = np.zeros([len(fit_params_log_tau_vs_size_for_specific_T_vs_adj[adj_index][T_fit_index-1])])

    ax[i].set_title(adjacencies[adj_index])
    ax[i].set_xlabel(r'$\log(N)$')
    # ax[i].set_yscale('log', base=10)
    # ax[i].set_ylim([-0.1, 3.5])
    # ax[i]. set_xlim([4e1,2e3])
    # ax[i].set_xscale('log', base=10)

if divide_by_size:
    ax[0].set_ylabel(r'$\log(\tau/N)$')
else:
    ax[0].set_ylabel(r'$\log(\tau)$')
# fig.suptitle(f'$\\alpha={alpha}$')
figs.colorbar_for_lines(fig, Ts[::2].round(1))
fig.show()

# %% BINNED DATA ONLY Plot fit params of autocorrelation time vs size
start_range_vs_adj=[16,21,59,65]
end_range_vs_adj=[90,250,350,490]

fig, ax = plt.subplots()

for i, adj_index in enumerate(adj_indices):
    params = np.array(fit_params_log_tau_vs_size_for_specific_T_vs_adj[adj_index])
    params[params == 0] = np.nan
    ax.plot(Ts[start_range_vs_adj[i]:end_range_vs_adj[i]], np.exp(params[:, 0])[start_range_vs_adj[i]:end_range_vs_adj[i]], color=colors_sizes[adj_index],  label=f'{adjacencies[adj_index]}')
    # ax.plot(Ts, params[:, 1], marker=marker_adjacencies[adj_index], markersize=4, linewidth=1, color='r', label=f'B(T), {adjacencies[adj_index]}')
    # ax.plot(Ts[start_range_vs_adj[i]:end_range_vs_adj[i]], 1/np.exp(params[:, 2])[start_range_vs_adj[i]:end_range_vs_adj[i]], color=colors_sizes[adj_index],  label=f'{adjacencies[adj_index]}')
    # ax.plot(Ts, 10**(params[:, 1]), marker = marker_adjacencies[adj_index])

ax.legend(fontsize=8, ncols=2)
# fig.suptitle(r'$ \tau(N) \propto C(T)N^{A(T)}e^{B(T)N}$')
fig.suptitle(r'$ \overline{\tau_{q^2}}(N,T) \propto N^{A(T)}$')
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_ylabel(r'$A(T)$')
ax.set_xlabel(r'$T$')
figs.export('power scaling autocorrelation time.pdf')
fig.show()

# %% You need Tc(N) vs adj to run the cell below, (run the cell called Tc vs adjacency) also, u must delete data for 1200 because there we dont have binned data.

for i, T_max in enumerate(T_max_vs_adj):
    try:
        T_max_vs_adj[i] = np.delete(T_max,4)
    except:
        continue
# %% BINNED DATA ONLY Plot collapse of autocorrelation time
nu = -0.3
sizes = [100, 200, 400, 800, 1600]
fig, ax = plt.subplots(ncols=3, figsize=[20, 6])
for i, adj_index in enumerate(adj_indices):
    for size_index in range(len(sizes)):
        ax[i].plot((T_tau_vs_adj[adj_index][size_index]-T_max_vs_adj[adj_index][size_index])*np.array(sizes_vs_adj[adj_index][size_index])**nu,
                   10**np.array(log_tau_vs_adj[adj_index][size_index]), color=colors_sizes[size_index], marker=marker_adjacencies[adj_index], linewidth=0, markersize=3)

    ax[i].set_title(adjacencies[adj_index])
    ax[i].set_xlabel('$T$')
    ax[i].set_yscale('log')

    # if divide_by_size:
    #     ax[i].set_ylim([-3.5, 1])
    #     ax[i].set_xlim([0, 5])
    # else:
    #     ax[i].set_ylim([-0.1, 3.5])
    #     ax[i].set_xlim([0, 5])

if divide_by_size:
    ax[0].set_ylabel(r'$\log(\tau/N)$')
else:
    ax[0].set_ylabel(r'$\log(\tau)$')
fig.show()

# %% BINNED DATA ONLY Plot q_dist
MCS_index = -1
dist = np.linspace(-1, 1, n_q_dist)
replica_index = 18
size_index = -1

fig1, ax1 = plt.subplots(ncols=1, figsize=[6, 4], dpi=150)
q_dist = q_dist_vs_size[size_index][MCS_index]
for T_index in range(copies):
    q_dist_T = q_dist[replica_index, :, T_index]
    q_dist_T = q_dist_T / q_dist_T.max()
    if np.any(T_index == T_indices):
        ax1.plot(dist, q_dist_T, '.-', color=colors[np.where(T_index == T_indices)[0][0]])
ax1.set_title(r'$P(q)$')
ax1.set_xlabel(r'$q$')
figs.colorbar_for_lines(fig1, T_vs_size[size_index][0][T_indices].round(2))
fig1.tight_layout()
fig1.show()

# %% BINNED DATA ONLY Plot skewness of q_dist vs temperature for diferent sizes
dist = np.linspace(-1, 1, n_q_dist)

fig2, ax2 = plt.subplots(ncols=1, figsize=[6, 4], dpi=150)

for size_index in range(len(sizes)):
    print(sizes[size_index])
    MCS_index = len(MCS_avg_vs_size[size_index]) - 1
    q_dist = q_dist_vs_size[size_index][MCS_index]
    n_replicas = q_dist.shape[0]
    skewness = np.zeros([copies, n_replicas])

    for T_index in range(copies):
        for replica_index in range(n_replicas):
            q_dist_T = q_dist[replica_index, :, T_index]
            # skewness[T_index, replica_index] = sps.skew(np.repeat(dist, q_dist_T.astype('int')))
            skewness[T_index, replica_index] = rfc.skewness_of_histogram(q_dist_T, dist)

    ax2.plot(T_vs_size[size_index][0], np.abs(skewness).mean(1), '.-', label=f'$N=${sizes[size_index]}')
    # ax2.plot(T_vs_size[size_index][0], skewness, '.-', label=f'$N=${sizes[size_index]}')

ax2.set_yscale('linear')
# ax2.set_yscale('log')
ax2.legend()
ax2.set_xticks(T_vs_size[size_index][0][T_indices].round(1))
ax2.set_title(r'$\overline{\widetilde{\mu_3}(q)}$')
ax2.set_xlabel(r'$T$')
ax2.set_xlim([T_vs_size[size_index][0][T_index_0], Tf])
fig2.tight_layout()
fig2.show()

# %% BINNED DATA ONLY Plot skewness of q_dist vs MCS_avg for one size and diferent temperatures
dist = np.linspace(-1, 1, n_q_dist)
size_index = -2

n_replicas = N_configs_vs_size[size_index][-1]
skewness = np.zeros([len(MCS_avg_vs_size[size_index]), copies, n_replicas])

fig, ax = plt.subplots()
for T_index in range(copies):
    for MCS_index in range(len(MCS_avg_vs_size[size_index]) - 1):
        q_dist = q_dist_vs_size[size_index][MCS_index]
        for replica_index in range(n_replicas):
            q_dist_T = q_dist[replica_index, :, T_index]
            skewness[T_index, replica_index] = sps.skew(np.repeat(dist, q_dist_T.astype('int')))
            # skewness[MCS_index, T_index, replica_index] = rfc.skewness_of_histogram(q_dist_T, dist)

    if np.any(T_index == T_indices):
        ax.plot(MCS_avg_vs_size[size_index], np.abs(skewness).mean(2)[:, T_index], '.-',
                color=colors[np.where(T_index == T_indices)[0][0]])

ax.set_yscale('linear')
ax.set_xscale('log')
ax.legend()
# ax.set_xticks(T_vs_size[size_index][0][T_indices].round(1))
# ax.set_title(r'$\overline{\widetilde{\mu_3}(q)}$')
# ax.set_xlabel(r'$T$')
# ax.set_xlim([T_vs_size[size_index][0][T_index_0], Tf])
fig.tight_layout()
fig.show()

# %% BINNED DATA ONLY Convergence of q2 vs MCS_avg
fig, ax = plt.subplots(dpi=200)
size_idx = 4
n_conf = len(labels_vs_size[size_idx][0])
n_MCS = len(MCS_avg_vs_size[size_idx])

error_N = np.zeros([n_conf, n_MCS - 1, copies])
for k, label in enumerate(labels_vs_size[size_idx][0]):
    for i in reversed(range(n_MCS)):
        index = np.where(labels_vs_size[size_idx][i] == label)[0][0]
        if i == n_MCS - 1:
            q2_converged = q2_vs_size[size_idx][i][index]
        else:
            error_N[k, i] = q2_converged - q2_vs_size[size_idx][i][index]

fig, ax = plt.subplots(dpi=200)
x = MCS_avg_0 * 2 ** np.arange(n_MCS - 1)
for T_index in T_indices:
    ax.errorbar(np.arange(n_MCS - 1), error_N.mean(0)[::-1, T_index],
                yerr=error_N.std(0)[::-1, T_index] / np.sqrt(n_conf),
                color=colors[np.where(T_index == T_indices)[0][0]], linewidth=1)
    # ax.plot(np.arange(n_MCS-1), error_N.mean(0)[::-1, T_index],
    #             color=colors[np.where(T_index == T_indices)[0][0]], linewidth=1)

# ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_ylabel(r'$\delta \: q^2_n$')
figs.colorbar_for_lines(fig, T_vs_size[-1][0][T_indices].round(2), label='$T$', location='top')
fig.suptitle(f'$n=$ {sizes[size_idx]}')
fig.tight_layout()
fig.show()

# %% BINNED DATA ONLY Convergence criteria using cluster link overlap
for size_index in range(len(sizes)):
    fig, ax = plt.subplots(dpi=200)
    T = T_vs_size[size_index][0]
    n_MCS = len(MCS_avg_vs_size[size_index])

    cluster_link_convergence = np.zeros([n_MCS, copies])
    for MCS_index in range(n_MCS):
        U = U_vs_size[size_index][MCS_index].mean(0)
        ql = ql_vs_size[size_index][MCS_index].mean(0)
        cluster_link_convergence[MCS_index] = 1 - T * np.abs(U) / 1.5 - ql

    fig, ax = plt.subplots(dpi=200)
    for T_index in T_indices:
        ax.plot(np.arange(n_MCS), cluster_link_convergence[::-1, T_index], '.-',
                color=colors[np.where(T_index == T_indices)[0][0]], linewidth=1)

    figs.colorbar_for_lines(fig, T[T_indices].round(2), label='$T$', location='top')

    ax.set_title(fr'n={sizes[size_index]} \\ \\ $1-T|U|/c -q_l$')
    fig.tight_layout()
    fig.show()

# %% Calculate Cv
CvT_vs_size = []
for MCS_index, size in enumerate(sizes):
    U = U_vs_size[i][-1]
    CvT = np.gradient(U, T) / T
    CvT[0] = CvT[1] - CvT[1] * (CvT[2] - CvT[1]) / (T[2] - T[1]) * 0.5
    if size == 800:  # or size == 1600:
        CvT[1] = CvT[2] - CvT[2] * (CvT[3] - CvT[2]) / (T[3] - T[2]) * 0.5
        CvT[0] = CvT[1] - CvT[1] * (CvT[2] - CvT[1]) / (T[2] - T[1]) * 0.5
    if size == 1600:
        CvT[2] = CvT[3] - CvT[3] * (CvT[4] - CvT[3]) / (T[4] - T[3]) * 0.5
        CvT[1] = CvT[2] - CvT[2] * (CvT[3] - CvT[2]) / (T[3] - T[2]) * 0.5
        CvT[0] = CvT[1] - CvT[1] * (CvT[2] - CvT[1]) / (T[2] - T[1]) * 0.5
    CvT_vs_size.append(CvT)

    plt.plot(T, CvT, '.-', label=f'{size}')

plt.legend()
plt.show()

# %% Create optimal temperature distribution for parallel tempering
T0_n = 0.5
Tf_n = 2.5
T_opt_vs_size = []
for size, CvT in zip(sizes, CvT_vs_size):
    T0 = T[0]
    Tf = T[-1]
    copies = 40
    # copies = 2000
    dS = np.trapz(CvT, T) / copies
    error = dS / 100
    dT = (Tf - T0) / (copies - 1)
    T_opt = np.zeros(copies)
    T_opt[0] = T0

    for c in range(copies - 1):
        T_0 = T_opt[c]
        T_1 = T_0 + dT
        CvT_0 = np.interp(T_0, T, CvT)
        CvT_1 = np.interp(T_1, T, CvT)
        dS_01 = np.trapz([CvT_0, CvT_1], [T_0, T_1])
        while np.abs(dS_01 - dS) > error:
            dT /= 2
            if dS_01 < dS:
                T_1 += dT
            else:
                T_1 -= dT
            CvT_1 = np.interp(T_1, T, CvT)
            dS_01 = np.trapz([CvT_0, CvT_1], [T_0, T_1])
        T_opt[c + 1] = T_1
        dT = T_1 - T_0

    print(T_opt)
    T_opt = T_opt[(T_opt > T0_n) * (T_opt < Tf_n)]
    # T_opt = np.concatenate(([T0_n], T_opt, [Tf_n]))
    T_opt_vs_size.append(T_opt)
    print(T_opt)

    # # %%
    fig, ax = plt.subplots(dpi=200)
    ax.plot(np.linspace(0, 1, len(T_opt)), np.linspace(T0, Tf, len(T_opt)), marker='.', label='Linear')
    ax.plot(np.linspace(0, 1, len(T_opt)), T_opt, marker='.', label='Constant entropy')
    ax.set_title(f'n={size}')
    ax.legend()
    fig.show()

    current_dir = sys.path[-3]
    if add == 0:
        dir_T_dist = current_dir + f'/Cluster/temperature_distributions/{adjacency}_{distribution},n={size},T={T0_n}_{Tf_n}.dat'
    else:
        dir_T_dist = current_dir + f'/Cluster/temperature_distributions/{adjacency}_{distribution},n={size},T={T0_n}_{Tf_n},add={add}.dat'
    file_T_dist = open(dir_T_dist, 'w')
    np.savetxt(file_T_dist, T_opt)
    file_T_dist.close()

T_opt_vs_size = np.array(T_opt_vs_size)
## %% fit T_opt scaling
from scipy.optimize import curve_fit


def parabola(x, b, c):
    return b * x + c


params = np.zeros([2, copies])
for c in range(copies):
    params[:, c] = curve_fit(parabola, sizes, T_opt_vs_size[:, c])[0]

## %%
sizes_fit = np.linspace(sizes[0], sizes[-1], 100)

for k in [0, 5, 10, 15, 20, 25]:
    for c in range(0 + k, 5 + k):
        plt.plot(sizes, T_opt_vs_size[:, c], '.-')
        plt.plot(sizes_fit, parabola(sizes_fit, params[:, c][0], params[:, c][1]))
    plt.show()

## %%
T_opt_predict = np.zeros(copies)

for size in sizes:
    for c in range(copies):
        T_opt_predict[c] = parabola(size, params[:, c][0], params[:, c][1])
    plt.plot(np.linspace(0, 1, len(T_opt)), T_opt_predict, 'r.-', linewidth=0.5)

    # plt.plot(np.linspace(0, 1, len(T_opt)), T_opt_vs_size[size_idx])
    plt.xlim([0, 1])
    plt.ylim([T0, Tf])
    plt.suptitle(f'n={size}')
    plt.show()

    current_dir = sys.path[1]
    dir_T_dist = current_dir + f'/Cluster/temperature_distributions/{adjacency}_{distribution},n={size},T={T0}_{Tf}.dat'
    file_T_dist = open(dir_T_dist, 'w')
    np.savetxt(file_T_dist, T_opt_predict)
    file_T_dist.close()

# %% Estimate error of non thermalization by convergence
g_extrapolated_vs_size, error_extrapolated_vs_size = rfc.extrapolate_convergence(g_vs_size, error_vs_size, MCS_avg_0,
                                                                                 max_MCS_vs_size[2], skip_initial_MCS_0=2)

size_index = -1
fig, ax = plt.subplots(figsize=(8 / 1.5, 8 / 1.5), dpi=125)
ax.plot(T_vs_size[size_index][0], g_extrapolated_vs_size[size_index])
ax.plot(T_vs_size[size_index][0], g_vs_size[size_index][-1], 'r')
ax.set_ylim([0, 1])
fig.show()

# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16 / 1.5, 8 / 1.5), dpi=125)
# for size_index in range(len(sizes)):
#     ax2.plot(T_vs_size[size_index][0], g_extrapolated_vs_size[size_index])
#     ax1.plot(T_vs_size[size_index][0], g_vs_size[size_index][-1])
# ax1.set_ylim([0, 1])
# ax2.set_ylim([0, 1])
# fig.show()

fig, ax = plt.subplots(figsize=(8 / 1.5, 16 / 1.5), dpi=125)
ax.plot(T_vs_size[size_index][0], np.array(error_extrapolated_vs_size).T)
fig.show()

fig, ax = plt.subplots()
for size_index, (g, err) in enumerate(zip(g_vs_size, error_vs_size)):
    ax.errorbar(T_vs_size[size_index][0], g[-1], yerr=err[-1])
fig.tight_layout()
fig.show()

# %% Pade finite size scaling analysis and estimate Tc
ic = ic_jc_vs_adj[adj_index][0]
jc = ic_jc_vs_adj[adj_index][1]
# ic = list(range(2, 7))
# jc = list(range(2, 7))

# error_dg_dT_vs_size_best = [(T_vs_size_best[size_index][1]-T_vs_size_best[size_index][0])/(np.sqrt(12)) * np.ones(copies_vs_size[size_index][-1]) for size_index in range(len(sizes))]
inopt = pf.pade_fss_analysis(sizes_vs_adj[adj_index], T_vs_size_best, dg_dT_vs_size_best, error_dg_dT_vs_size_best, ntr=10, ic=ic, jc=jc,
                             figsize_in=[16 / 1.5, 6 / 1.5], dpi_in=125, adjacency=adjacency, method_ic_jc='specific')


# %% Tc and error Tc with bootstrap using multicore
Tc_bootstrap, T_max_bootstrap = pf.estimate_Tc_with_pade_bootstrap_parallel(sizes_vs_adj[adj_index], T_vs_size_best,
                                                                            error_dg_dT_vs_size_best,
                                                                            dg_dT_bootstrap_vs_size_best,
                                                                            ic=ic_jc_vs_adj[adj_index][0],
                                                                            jc=ic_jc_vs_adj[adj_index][1],
                                                                            ntr=10, maxfev=10000,
                                                                            threads=cpu_count())

Tc_max_err = 2*np.nanstd(T_max_bootstrap, 0)
Tc_max = np.nanmean(T_max_bootstrap, 0)

for T_max_print, err_print in zip(Tc_max, Tc_max_err):
    print('{:.3f} $\pm$ {:.3f}'.format(T_max_print, err_print))
print('{:.3f} $\pm$ {:.3f}'.format(np.nanmean(Tc_bootstrap), np.nanstd(Tc_bootstrap)))
# %% T_max vs size histogram
fig, ax = plt.subplots()
ax.hist(T_max_bootstrap, 500, histtype='stepfilled', alpha=0.7, density=True)
ax.bar(Tc_max, [100]*len(sizes), 0.0005, color='k')
fig.show()

# %% Tc histogram

fig, ax = plt.subplots()
ax.hist(Tc_bootstrap, 100, histtype='stepfilled', density=True)
fig.show()

# %% Scaling figure, single adjacency
sizes = sizes_vs_adj[adj_index]
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=[12,5])
ax1.errorbar(1 / np.array(sizes), Tc_max, yerr=Tc_max_err,
             label=rf'{adjacencies[adj_index]}', markersize=3, capsize=4, capthick=1, elinewidth=1)

alpha = 6.9
x_fit_T_max = np.linspace(0, 1 / sizes[1] ** (1 / alpha), 100)
z = np.polyfit(1 / np.array(sizes[4:]) ** (1 / alpha), Tc_max[4:], 1)
fit_T_max = np.poly1d(z)

ax2.errorbar(1 / np.array(sizes) ** (1 / alpha), Tc_max,
             yerr=Tc_max_err, label=rf'{adjacencies[adj_index]}, $\alpha=${alpha}', markersize=3, capsize=4, capthick=1, elinewidth=1, linewidth=0)
ax2.plot(x_fit_T_max, fit_T_max(x_fit_T_max), color='k', linestyle='--', linewidth=0.5)

ax1.set_ylabel('$T_c$')
ax1.set_xlabel(r'$N^{-1}$')
ax1.set_ylim(bottom=0)
ax1.set_xlim(left=0)
ax2.legend()
ax2.set_ylabel('$T_c$')
ax2.set_xlabel(r'$N^{-1/\alpha}$')
ax2.set_ylim(bottom=0)
ax2.set_xlim(left=0)
# ax.set_yscale('log')
# ax2.set_xscale('log')
# ax.set_title(rf'$\alpha=${alpha}')
fig.tight_layout()
fig.show()
# %% Tc vs adjacency
write_file = True
data_type = 'all'
MCS_N_config_condition = 'max_MCS_with_minimum_N_configs'
min_N_config = 1000
n_bootstrap = 20*5
T_term = 0
ntr = 10
maxfev = 10000

Tc_bootstrap_vs_adj = [[] for _ in range(len(adjacencies))]
Tc_vs_adj = [[] for _ in range(len(adjacencies))]
error_Tc_vs_adj = [[] for _ in range(len(adjacencies))]
T_max_vs_adj = [[] for _ in range(len(adjacencies))]
error_T_max_vs_adj = [[] for _ in range(len(adjacencies))]

if write_file:
    file_simulation_info = open('simulation_info.txt','a')
    file_simulation_info.write('Adjacency & Size & Min. T & Max. T & Tempering copies & MCS & Replicas & Tc \\\\ \n')

for adj_index in range(len(adjacencies)):
# for adj_index in range(len(adjacencies)-3, len(adjacencies)):
# for adj_index in range(8):
    sizes = [100, 200, 400, 800, 1200, 1600, 3200]
    sizes_vs_adj = [_ for _ in range(len(adjacencies))]
    for i, adj in enumerate(adjacencies):
        if adj == 'chimera':
            sizes_vs_adj[i] = [72, 200, 392, 800, 1152, 1568, 3200]
        elif adj == 'pegasus':
            sizes_vs_adj[i] = [128, 256, 448, 960, 1288, 1664, 3648]
        elif adj == 'zephyr':
            sizes_vs_adj[i] = [48, 160, 336, 576, 880, 1248, 2736]
        else:
            sizes_vs_adj[i] = sizes

    sizes_vs_adj = [siz_vs_adj[:-1] for siz_vs_adj in sizes_vs_adj]
    sizes = sizes[:-1]

    MCS_avg_vs_size, N_configs_vs_size, copies_vs_size, labels_vs_size, T_vs_size, q2_vs_size, q4_vs_size, \
    ql_vs_size, U_vs_size, U2_vs_size, σ2_q2_bin_vs_size, σ2_q4_bin_vs_size, q_dist_vs_size, g_vs_size, \
    g_bootstrap_vs_size, error_vs_size, dg_dT_vs_size, dg_dT_bootstrap_vs_size, error_dg_dT_vs_size = \
    rfc.read_data(adjacencies[adj_index], distribution, sizes, add_vs_adj[adj_index],
                  T0_Tf_vs_adj[adj_index][0], T0_Tf_vs_adj[adj_index][1], MCS_avg_0,
                  [max_MCSs_vs_adj_binned[adj_index][:-1], max_MCSs_vs_adj_old[adj_index][:-1], max_MCSs_vs_adj_fast[adj_index][:-1]],
                  data_type, only_max_MCS=True)

    n_cases = sum([len(MCS_avg) for MCS_avg in MCS_avg_vs_size])
    results_g_parallel = Parallel(n_jobs=min(cpu_count(), n_cases))(delayed(rfc.binder_cumulant_and_error_bootstrap)
                                                                    (T_vs_size[size_ind][MCS_ind],
                                                                     q2_vs_size[size_ind][MCS_ind],
                                                                     q4_vs_size[size_ind][MCS_ind],
                                                                     n_bootstrap=n_bootstrap, error_type='1')
                                                                    for size_ind in range(len(sizes))
                                                                    for MCS_ind in
                                                                    range(len(MCS_avg_vs_size[size_ind])))

    k = 0
    for size_ind in range(len(sizes)):
        for MCS_ind in range(len(MCS_avg_vs_size[size_ind])):
            g_vs_size[size_ind][MCS_ind], g_bootstrap_vs_size[size_ind][MCS_ind], error_vs_size[size_ind][MCS_ind], \
            dg_dT_vs_size[size_ind][MCS_ind], dg_dT_bootstrap_vs_size[size_ind][MCS_ind], error_dg_dT_vs_size[size_ind][
                MCS_ind] = results_g_parallel[k]
            k += 1
    del results_g_parallel

    T_vs_size_best, g_vs_size_best, error_vs_size_best, dg_dT_vs_size_best, dg_dT_bootstrap_vs_size_best, \
        error_dg_dT_vs_size_best, MCS_avg_vs_size_best, N_configs_vs_size_best = [[[] for _ in range(len(sizes))] for _ in range(8)]
    dead_index = None

    for size_index in range(len(sizes)):
        assigned = False
        k = -1
        try:
            while not assigned:
                match MCS_N_config_condition:
                    case 'max_MCS_with_minimum_N_configs':
                        if N_configs_vs_size[size_index][k] > min_N_config:
                            assigned = True
                        else:
                            k -= 1
                    case 'max_N_configs':
                        if N_configs_vs_size[size_index][k] == max(N_configs_vs_size[size_index]) and \
                                N_configs_vs_size[size_index][k] > 0:
                            assigned = True
                        else:
                            k -= 1
            print(sizes[size_index], k)
            T_vs_size_best[size_index] = T_vs_size[size_index][k]
            g_vs_size_best[size_index] = g_vs_size[size_index][k]
            error_vs_size_best[size_index] = error_vs_size[size_index][k]
            dg_dT_vs_size_best[size_index] = dg_dT_vs_size[size_index][k]
            dg_dT_bootstrap_vs_size_best[size_index] = dg_dT_bootstrap_vs_size[size_index][k]
            error_dg_dT_vs_size_best[size_index] = error_dg_dT_vs_size[size_index][k]
            MCS_avg_vs_size_best[size_index] = MCS_avg_vs_size[size_index][k]
            N_configs_vs_size_best[size_index] = N_configs_vs_size[size_index][k]
        except:
            dead_index = size_index
            continue

    if dead_index is not None:
        del T_vs_size_best[dead_index], g_vs_size_best[dead_index], error_vs_size_best[dead_index], dg_dT_vs_size_best[
            dead_index], \
            dg_dT_bootstrap_vs_size_best[dead_index], error_dg_dT_vs_size_best[dead_index], sizes[dead_index], \
        sizes_vs_adj[adj_index][dead_index], MCS_avg_vs_size_best[dead_index], N_configs_vs_size_best[dead_index]

    Tc_bootstrap, T_max_bootstrap = \
        pf.estimate_Tc_with_pade_bootstrap_parallel(sizes_vs_adj[adj_index], T_vs_size_best,
                                                    error_dg_dT_vs_size_best, dg_dT_bootstrap_vs_size_best,
                                                    T_term, ic_jc_vs_adj[adj_index][0], ic_jc_vs_adj[adj_index][1],
                                                    ntr, maxfev, min(n_bootstrap, cpu_count()-1))

    T_max_bootstrap = np.array(T_max_bootstrap).reshape((-1,len(sizes)))
    T_max_bootstrap[T_max_bootstrap==0] = np.nan

    T_max_vs_adj[adj_index] =  np.nanmean(T_max_bootstrap,0)
    error_T_max_vs_adj[adj_index] =  2*np.nanstd(T_max_bootstrap,0)

    Tc_bootstrap_vs_adj[adj_index] = Tc_bootstrap
    Tc_vs_adj[adj_index] = np.nanmean(Tc_bootstrap)
    error_Tc_vs_adj[adj_index] = 2*np.nanstd(Tc_bootstrap)
    # error_Tc_vs_adj.append(np.nanstd(Tc_bootstrap[np.all((Tc - 0.1 < Tc_bootstrap, Tc_bootstrap < Tc + 0.1), 0)]))

    if write_file:
        for size_index in range(len(sizes)):
            add_str = f'{add}'
            file_simulation_info.write(f'{adjacency}{ bool(add>0)*add_str } & {sizes_vs_adj[adj_index][size_index]} & {T0} & {Tf} & '
                                       f'{copies_vs_size[size_index][0]} & {MCS_avg_vs_size_best[size_index]} & {N_configs_vs_size_best[size_index]}  \\\\ \n')

if write_file:
    file_simulation_info.close()

sizes = sizes_vs_adj[adj_index]

# %% RRG and 1D vs theory
graphs = 'rrg_and_small_world'
fname = f'Processed_Data/Tc_vs_adj_{graphs}_read_mode={MCS_N_config_condition}_min_N_config={min_N_config}.npz'
data = np.load(fname, allow_pickle=True)
sizes_load = data['sizes']
sizes_vs_adj_load = data['sizes_vs_adj']

Tc_vs_adj = data['Tc_vs_adj']
error_Tc_vs_adj= data['error_Tc_vs_adj']
T_max_vs_adj = data['T_max_vs_adj']
error_T_max_vs_adj = data['error_T_max_vs_adj']

#%% Incise: calculate std ofsets
import Modules.graph_generator as sg

mu_offset_vs_adj = []
std_offset_vs_adj = []
for adj_index in range(4):
    mus = [sg.connectivity_matrix(1600, adjacencies[adj_index], distribution, rng=np.random.default_rng(34172431+1*j))[1].mean()
           for j in range(100)]
    stds = [sg.connectivity_matrix(1600, adjacencies[adj_index], distribution, rng=np.random.default_rng(34172431+1*j))[1].std()
           for j in range(100)]

    mu_offset_vs_adj.append(np.array(mus).std()*2)
    std_offset_vs_adj.append(np.array(stds).std()*2)

# %% Fit offset
k = np.array([3, 5, 7, 9])
k_fit = np.linspace(3, 9, 10000)
mu_offset_fit  = np.polyval(np.polyfit(k,  mu_offset_vs_adj, 2), k_fit)
std_offset_fit = np.polyval(np.polyfit(k, std_offset_vs_adj, 2), k_fit)

fig, ax = plt.subplots()
ax.plot(k, mu_offset_vs_adj, 'k')
ax.plot(k, std_offset_vs_adj, 'r')
ax.plot(k_fit, mu_offset_fit, 'k:')
ax.plot(k_fit, std_offset_fit, 'r:')
fig.show()

#%% Back to business
Tc_x = np.linspace(0, 3, 10000)
x = np.linspace(-100, 100, 10000)
# k_up = np.zeros([len(Tc_x)])
# k_dw = np.zeros([len(Tc_x)])
k_ideal = np.zeros([len(Tc_x)])
mu = 0.01
std_offset = 0.01
for i, Tc in enumerate(Tc_x):
    # y_up = np.tanh(x / Tc) ** 2 * np.exp(-0.5 * ((x - mu_offset_fit[i]) / (1+std_offset_fit[i])) ** 2) / (np.sqrt(2 * np.pi) * (1+std_offset_fit[i]))
    # y_dw = np.tanh(x / Tc) ** 2 * np.exp(-0.5 * ((x - mu_offset_fit[i]) / (1-std_offset_fit[i])) ** 2) / (np.sqrt(2 * np.pi) * (1-std_offset_fit[i]))
    y_ideal = np.tanh(x / Tc) ** 2 * np.exp(-0.5 * (x ** 2)) / np.sqrt(2 * np.pi)
    # k_up[i] = 1 / np.trapz(y_up, x)
    # k_dw[i] = 1 / np.trapz(y_dw, x)
    k_ideal[i] = 1 / np.trapz(y_ideal, x)

fig, ax = plt.subplots(dpi=200)
# ax.plot(k_dw + 1, Tc_x,'k')
# ax.plot(k_up + 1, Tc_x,'k')
# ax.fill_betweenx(Tc_x, k_dw+ 1, k_up+ 1, alpha=0.5)
ax.plot(k_ideal + 1, Tc_x, label=r'$k-$random regular, theory')
ax.errorbar(np.array([3, 5, 7, 9]), Tc_vs_adj[:4], error_Tc_vs_adj[:4], label=r'$k-$random regular, simulation',capsize=4, capthick=1,
                 elinewidth=1, linewidth=1.25)
ax.errorbar(np.array([3, 5, 7, 9]), Tc_vs_adj[4:-4], error_Tc_vs_adj[4:-4], label='Small world', capsize=4, capthick=1,
                 elinewidth=1, linewidth=1.25)

ax.set_xticks([3, 5, 7, 9])
ks = [np.abs(k_ideal+1-ki).argmin() for ki in [3, 5, 7, 9]]
print(Tc_x[ks])

ax.set_xlim([2,10])
ax.set_ylim([0,2.8])
ax.legend()
ax.set_xlabel('$k$')
ax.set_ylabel('$T_C$')
fig.tight_layout()
fig.show()
figs.export(f'Random-regular and small-world vs theory_with_confidence_theory.pdf')


#%%
print(Tc_x[ks])

# %% Tc histogram
fig, axs = plt.subplots(ncols=4, nrows=2, dpi=100, figsize=[16, 9])
axs = axs.flatten()
for ax, Tc_bootstrap, adjacency in zip(axs, Tc_bootstrap_vs_adj, adjacencies):
    try:
        Tc = np.nanmean(Tc_bootstrap)
        ax.hist(Tc_bootstrap,500, histtype='stepfilled')
    except:
        pass
    ax.set_title(f'{adjacency}, n_bootstrap = {len(np.isnan(Tc_bootstrap)) - np.isnan(Tc_bootstrap).sum()}')
fig.tight_layout()
plt.show()

# %% D-Wave analysis

alpha_vs_adj = [_ for _ in range(len(adjacencies))]
alpha_vs_adj[-3] = 6.5
alpha_vs_adj[-2] = 25
alpha_vs_adj[-1] = 10.5

color_vs_adj = [ 'turquoise',  'tab:olive', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray','tab:blue', 'goldenrod', 'tab:orange', 'tab:red']

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=[12,5])

for adj_index in range(len(adjacencies)-3, len(adjacencies)):
    ax1.errorbar(1/np.array(sizes_vs_adj[adj_index]), T_max_vs_adj[adj_index], yerr=error_T_max_vs_adj[adj_index], label=rf'{adjacencies[adj_index]}',
                color=color_vs_adj[adj_index], markersize=3, capsize=4, capthick=1,elinewidth=1)

    alpha = alpha_vs_adj[adj_index]
    x_fit_T_max = np.linspace(0, 1/sizes_vs_adj[adj_index][0]**(1/alpha), 100)
    z = np.polyfit(1/np.array(sizes_vs_adj[adj_index][1:])**(1/alpha), T_max_vs_adj[adj_index][1:],1)
    fit_T_max = np.poly1d(z)

    ax2.errorbar(1/np.array(sizes_vs_adj[adj_index])**(1/alpha), T_max_vs_adj[adj_index], yerr=error_T_max_vs_adj[adj_index], label=rf'{adjacencies[adj_index]}, $\alpha=${alpha}',
                color=color_vs_adj[adj_index], markersize=3, capsize=4, capthick=1,elinewidth=1, linewidth=0)
    ax2.plot(x_fit_T_max,fit_T_max(x_fit_T_max), color='k',linestyle='--')
    # ax.errorbar(np.array(sizes_vs_adj[adj_index]), T_max_vs_adj[adj_index], yerr=error_T_max_vs_adj[adj_index], label=f'{adjacencies[adj_index]}')

# ax1.legend()
ax1.set_ylabel('$T_c$')
ax1.set_xlabel(r'$N^{-1}$')
ax1.set_ylim(bottom=0)
ax1.set_xlim(left=0)
ax2.legend()
ax2.set_ylabel('$T_c$')
ax2.set_xlabel(r'$N^{-1/\alpha}$')
ax2.set_ylim(bottom=0)
ax2.set_xlim(left=0)
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_title(rf'$\alpha=${alpha}')
fig.tight_layout()
fig.show()

figs.export(f'Tc_vs_adjacency_Dwave.pdf')