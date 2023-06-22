# %% Imports
import sys
import importlib
from joblib import Parallel, delayed, cpu_count

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from scipy.optimize import curve_fit

import Modules.read_data_from_cluster as rfc
import Modules.statistical_mechanics as sm
import Modules.pade_fits as pf
import Modules.figures as figs

plt.rcParams['font.size'] = '16'
plt.rcParams['figure.dpi'] = '200'
plt.rcParams['backend'] = 'QtAgg'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
   "font.sans-serif": "Helvetica",
})

importlib.reload(rfc)
importlib.reload(pf)
importlib.reload(sm)
importlib.reload(figs)
# %% Data parameters
distribution = 'gaussian_EA'

adjacencies = ['random_regular_3', 'random_regular_5', 'random_regular_7', 'random_regular_9',
               '1D+', '1D+', '1D+', '1D+',
               'chimera', 'pegasus', 'zephyr']

add_vs_adj = [0, 0, 0, 0,
              3.0, 5.0, 7.0, 9.0,
              0, 0, 0]

T0_Tf_vs_adj = [[0.2, 1.5], [0.5, 3.0], [0.5, 3.0], [1.0, 4.0],
                [0.2, 1.5], [1.0, 2.5], [1.3, 3.0], [1.5, 3.5],
                [0.2, 3.0], [0.2, 4.0], [0.5, 5.0]]

ic_jc_vs_adj = [[[3, 6, 6, 6, 4], [6, 3, 6, 6, 5]],
                [[5, 5, 4, 6, 4], [4, 5, 5, 5, 6]],
                [[5, 4, 5, 6, 6], [3, 6, 4, 3, 5]],
                [[5, 5, 6, 3, 5], [3, 4, 3, 5, 5]],

                [[6, 5, 5, 5, 6], [3, 3, 3, 4, 4]],
                [[4, 4, 4, 5, 5], [5, 5, 4, 4, 4]],
                [[4, 6, 6, 4, 4], [5, 5, 5, 4, 4]],
                [[4, 4, 4, 4, 4], [5, 4, 4, 4, 4]],

                [[3, 4, 3, 6, 6], [4, 5, 5, 5, 6]],
                [[4, 4, 4, 4, 3], [5, 3, 4, 4, 6]],
                [[3, 3, 5, 5, 4], [4, 4, 3, 3, 5]]]

MCS_avg_0 = 10000

max_MCSs_vs_adj_binned = np.array([[3, 3, 4, 6, 6],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [3, 3, 4, 6, 6],

                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0],

                                   [5, 6, 7, 8, 8],
                                   [5, 6, 7, 8, 8],
                                   [5, 6, 7, 8, 8]])
max_MCSs_vs_adj_binned = MCS_avg_0 * 2 **  max_MCSs_vs_adj_binned

max_MCSs_vs_adj_fast = np.array([[0, 0, 6, 7, 9],
                                 [0, 0, 6, 7, 9],
                                 [4, 5, 6, 7, 9],
                                 [4, 5, 6, 7, 9],

                                 [4, 5, 6, 7, 8],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],

                                 [0, 0, 6, 7, 9],
                                 [3, 3, 6, 7, 9],
                                 [3, 3, 6, 7, 9]])
max_MCSs_vs_adj_fast = MCS_avg_0 * 2 **  max_MCSs_vs_adj_fast
max_MCSs_vs_adj_old = [MCS_avg_0 * 2 ** np.array([1, 2, 5, 6, 7]),
                       MCS_avg_0 * 2 ** np.array([5, 5, 5, 5, 5]) + 4,
                       np.array([80002, 80002, 160002, 640001, 640002]),
                       MCS_avg_0 * 2 ** np.array([1, 2, 4, 5, 5]),
                       MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 5]),
                       MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 5]),
                       MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 5]),
                       MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 5]),
                       MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 5]),
                       MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 5]),
                       MCS_avg_0 * 2 ** np.array([0, 0, 0, 0, 0])]

sizes = [100, 200, 400, 800, 1600]

sizes_vs_adj = [_ for _ in range(len(adjacencies))]

for i, adj in enumerate(adjacencies):
    if adj == 'chimera':
        sizes_vs_adj[i] = [72, 200, 392, 800, 1568]
    elif adj == 'pegasus':
        sizes_vs_adj[i] = [128, 256, 448, 960, 1664]
    elif adj == 'zephyr':
        sizes_vs_adj[i] = [48, 160, 336, 576, 1248]
    else:
        sizes_vs_adj[i] = sizes


color_vs_size = ['turquoise',  'tab:olive', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray','tab:blue', 'goldenrod', 'tab:orange', 'tab:red']
marker_vs_adjacency = ['^', '>', 'v', '<', '1', '2', '3', '.', '4', 'P', 'd', '*']

# %% Choose an adjacency
adj_index = 9
only_max_MCS = False  # Must be 'False' for thermalization tests, 'True' to read faster for the rest
n_bootstrap = 36*10
data_type = 'binned'  # Must be 'binned' for thermalization tests
MCS_N_config_condition = 'max_MCS_with_a_minimum_of_N_configs'
min_N_config = 1000

adjacency = adjacencies[adj_index]
T0 = T0_Tf_vs_adj[adj_index][0]
Tf = T0_Tf_vs_adj[adj_index][1]
add = add_vs_adj[adj_index]
max_MCS_vs_size = [max_MCSs_vs_adj_binned[adj_index], max_MCSs_vs_adj_old[adj_index], max_MCSs_vs_adj_fast[adj_index]]

# %% Read data
MCS_avg_vs_size, N_configs_vs_size, copies_vs_size, labels_vs_size, T_vs_size, q2_vs_size, q4_vs_size, \
    ql_vs_size, U_vs_size, U2_vs_size, σ2_q2_bin_vs_size, σ2_q4_bin_vs_size, q_dist_vs_size = \
    rfc.read_data(adjacency, distribution, sizes, add, T0, Tf, MCS_avg_0, max_MCS_vs_size, data_type, only_max_MCS=only_max_MCS)

# %% Calculate g and dgdT with errors using bootstrap
g_vs_size, g_bootstrap_vs_size, error_vs_size, dg_dT_vs_size, dg_dT_bootstrap_vs_size, error_dg_dT_vs_size =\
    sm.binder_cumulant_parallel(sizes, T_vs_size, MCS_avg_vs_size, q2_vs_size, q4_vs_size, n_bootstrap)

# %% Choose the optimal simulation (in terms of MCS and N_configs) for each size
N_configs_vs_size_best, MCS_avg_vs_size_best, T_vs_size_best, g_vs_size_best, g_bootstrap_vs_size_best, \
    error_vs_size_best, dg_dT_vs_size_best, dg_dT_bootstrap_vs_size_best, error_dg_dT_vs_size_best = \
    sm.choose_optimal_MCS_N_config(sizes, N_configs_vs_size, MCS_avg_vs_size, T_vs_size, g_vs_size, g_bootstrap_vs_size, error_vs_size,
                                   dg_dT_vs_size, dg_dT_bootstrap_vs_size, error_dg_dT_vs_size,
                                   MCS_N_config_condition='max_MCS_with_a_minimum_of_N_configs', min_N_config=1000)

# %% Plot g
fig, ax1 = plt.subplots()
for size_index, (T, g, err) in enumerate(zip(T_vs_size_best, g_vs_size_best, error_vs_size_best)):
    ax1.errorbar(T, g, yerr=err,
                 label=f'{sizes_vs_adj[adj_index][size_index]}', color=color_vs_size[size_index], markersize=2, capsize=2, capthick=0.5, elinewidth=1, linewidth=0.5)
ax1.set_ylabel('$g$')
ax1.set_xlabel('$T$')
ax1.set_xlim([0, Tf])
fig.tight_layout()
fig.show()

# %% Plot dgdT
fig, ax = plt.subplots()
for size_index, (T, dg_dT, error_dg_dT) in enumerate(zip(T_vs_size_best, dg_dT_vs_size_best, error_dg_dT_vs_size_best)):
    ax.errorbar(T, -dg_dT, yerr=error_dg_dT, label=f'{sizes[size_index]}', color=color_vs_size[size_index],
                markersize=2, capsize=2, capthick=1, elinewidth=1, linewidth=0.5)
fig.show()

# %% Plot error of dgdT
fig, ax = plt.subplots()
for size_index, (T, error_dg_dT) in enumerate(zip(T_vs_size_best, error_dg_dT_vs_size_best)):
    ax.plot(T, error_dg_dT, label=f'{sizes[size_index]}', color=color_vs_size[size_index], linewidth=0.5)
fig.show()

#%% PROCESS DATA FOR FIGURES 9, 10, 11, 12 - Thermalization tests. TO PROCESS THIS DATA YOU MUST CHOOOSE data_type='binned' and only_max_MCS = False
size_index = 3
# %%  Thermalization test 1 - Logarithmic binning method
copies = len(T_vs_size_best[size_index])
n_conf = N_configs_vs_size_best[size_index]
n_MCS = len(MCS_avg_vs_size[size_index])
error_N = np.zeros([n_conf, n_MCS - 1, copies])

for k, label in enumerate(labels_vs_size[size_index][-1]):
    for i in reversed(range(n_MCS)):
        index = np.where(labels_vs_size[size_index][i] == label)[0][0]
        if i == n_MCS - 1:
            q2_converged = q2_vs_size[size_index][i][index]
        else:
            error_N[k, i] = q2_converged - q2_vs_size[size_index][i][index]

error_q2_logarithmic_binning = error_N.mean(0)
error_std_q2_logarithmic_binning = error_N.std(0)

# %%  Thermalization test 2 - Cluster link overlap
T = T_vs_size[size_index][-1]
cluster_link_convergence = np.zeros([n_MCS, copies])
for MCS_index in range(n_MCS):
    U = U_vs_size[size_index][MCS_index].mean(0)
    ql = ql_vs_size[size_index][MCS_index].mean(0)
    cluster_link_convergence[MCS_index] = 1 - T * np.abs(U) / 1.5 - ql

# %%  Thermalization test 3 - Autocorrelation time and thermal average error of q2
σ2_q2_bin_t = σ2_q2_bin_vs_size[size_index][-1]
MCS_avg = MCS_avg_vs_size[size_index][-1]

error_q2_vs_T = [_ for _ in range(copies)]
tau_q2_vs_T = [_ for _ in range(copies)]

σ2_q2_bin_c = σ2_q2_bin_t.mean(0)[:, 0]
M0 = MCS_avg
n_bins = σ2_q2_bin_c.shape[0]
bins = np.arange(n_bins)

for T_index in range(copies):
    σ2_q2_bin_c = σ2_q2_bin_t.mean(0)[:, T_index]

    M_bin = [M0 / 2 ** bin for bin in bins]
    error_q2_vs_T[T_index] = np.array([np.sqrt((1 / M) * sigma) for M, sigma in zip(M_bin, σ2_q2_bin_c)])

    tau_q2_vs_T[T_index]  = 2 ** bins * σ2_q2_bin_c[bins] / σ2_q2_bin_c[0]

# %%  Thermalization test 4 - P(q)
replica_index = 0

dist = np.linspace(-1, 1, 50)
q_dist = q_dist_vs_size[size_index][-1]
q_dist_vs_T = np.zeros([50,copies])

for T_index in range(copies):
    q_dist_vs_T[:,T_index] = q_dist[replica_index, :, T_index]
    q_dist_vs_T[:,T_index] /=  q_dist_vs_T[:,T_index].max()

# %% Store processed data
fname = f'Processed_Data/thermalization_tests_adjacency={adjacencies[adj_index]},size={sizes_vs_adj[adj_index][size_index]}'
np.savez(fname,MCS_avg_vs_size=MCS_avg_vs_size, N_configs_vs_size=N_configs_vs_size, T_vs_size=T_vs_size,
         error_q2_logarithmic_binning=error_q2_logarithmic_binning, error_std_q2_logarithmic_binning=error_std_q2_logarithmic_binning,
         cluster_link_convergence=cluster_link_convergence,error_q2_vs_T=error_q2_vs_T, tau_q2_vs_T=tau_q2_vs_T, bins=bins,
         q_dist_vs_T=q_dist_vs_T,  allow_pickle=True)

#%% PROCESS DATA FOR FIGURES 2 AND 5 - g and dgdT
pass
# %%  Calculate pade fits of dgdT
dg_dT_pade, T_c, peak_height = pf.pade_fss(sizes_vs_adj[adj_index], T_vs_size_best, dg_dT_vs_size_best,
                                                  error_dg_dT_vs_size_best, T_term_vs_size=False, ntr=10,
                                                  ic=ic_jc_vs_adj[adj_index][0], jc=ic_jc_vs_adj[adj_index][1],
                                                  method_ic_jc='specific')

# %% Store processed data
T0 = T0_Tf_vs_adj[adj_index][0]
Tf = T0_Tf_vs_adj[adj_index][1]
T_pade = np.linspace(T0 + (Tf - T0) * 0.1, Tf - (Tf - T0) * 0.1, 1000)
dg_dT_pade_array = [ dg_dT_pade[size_index](T_pade) for size_index in range(len(sizes))]
dg_dT_pade_T_max_array = [ dg_dT_pade[size_index](T_c[size_index]) for size_index in range(len(sizes))]

fname = f'Processed_Data/g_and_g_pade_and_dg_dT_inset_adj={adjacencies[adj_index]}_add={add_vs_adj[adj_index]}_' \
        f'read_mode={MCS_N_config_condition}_min_N_config={min_N_config}'
np.savez(fname, T_vs_size_best=T_vs_size_best, g_vs_size_best=g_vs_size_best, error_vs_size_best=error_vs_size_best,
         dg_dT_vs_size_best=dg_dT_vs_size_best, error_dg_dT_vs_size_best=error_dg_dT_vs_size_best,
         T_c=T_c, T_pade=T_pade, dg_dT_pade_array=dg_dT_pade_array, peak_height=peak_height,
         allow_pickle=True)

# %% PROCESS DATA FOR FIGURE 3 - Finite size scaling of dgdT for mean field graphs
pass
# %% Calculate Tc (peak position), peak height and peak widht of the pade approximants of dgdT
Tc_bootstrap, inv_peak_height_bootstrap, peak_width_bootstrap, \
    Tc, Tc_err, inv_peak_height, inv_peak_height_err, peak_width, peak_width_err = \
    pf.estimate_Tc_with_pade_bootstrap_parallel(sizes_vs_adj[adj_index], T_vs_size_best, error_dg_dT_vs_size_best,
                                                dg_dT_bootstrap_vs_size_best, ic=ic_jc_vs_adj[adj_index][0],
                                                jc=ic_jc_vs_adj[adj_index][1], ntr=10, maxfev=10000, threads=cpu_count())

# %% Extrapolate previous values to thermodynamic limit for the mean field case
Tc_inf, Tc_inf_err, inv_peak_height_inf, inv_peak_height_inf_err, peak_width_inf, peak_width_inf_err =\
    pf.extrapolate_thermodynamic_limit_mean_field_graphs(adjacency, sizes, Tc_bootstrap, inv_peak_height_bootstrap, peak_width_bootstrap)

# %% Histogram of bootstrap variables to check that they are gaussian like
fig, ax = plt.subplots()
ax.hist(Tc_bootstrap, 500, histtype='stepfilled', alpha=0.7, density=True)
ax.bar(Tc, [100]*len(sizes), 0.0005, color='k')
fig.show()

fig, ax = plt.subplots()
ax.hist(inv_peak_height_bootstrap, 500, histtype='stepfilled', alpha=0.7, density=True)
ax.bar(inv_peak_height, [100]*len(sizes), 0.0005, color='k')
fig.show()

fig, ax = plt.subplots()
ax.hist(peak_width_bootstrap, 500, histtype='stepfilled', alpha=0.7, density=True)
ax.bar(peak_width, [100]*len(sizes), 0.0005, color='k')
fig.show()

# %% Store processed data
fname = f'Processed_Data/fss_dg_dT_adj={adjacencies[adj_index]}_add={add_vs_adj[adj_index]}_read_mode={MCS_N_config_condition}_min_N_config={min_N_config}'
np.savez(fname, Tc=Tc, Tc_err=Tc_err, inv_peak_height=inv_peak_height, inv_peak_height_err=inv_peak_height_err, peak_width=peak_width,
         peak_width_err=peak_width_err, Tc_inf=Tc_inf, Tc_inf_err=Tc_inf_err, inv_peak_height_inf=inv_peak_height_inf,
         inv_peak_height_inf_err=inv_peak_height_inf_err, peak_width_inf=peak_width_inf,
         peak_width_inf_err=peak_width_inf_err, allow_pickle=True)

#%% PROCESS DATA FOR FIGURES 4 AND 6 - Tc vs adjacency
pass
# %% Tc vs adjacency
# graphs = 'mean_field'  # Fig. 4
graphs = 'Dwave' # Fig. 6
write_file = True

if graphs == 'Dwave':
    adj_iterable = range(len(adjacencies)-3, len(adjacencies))
elif graphs == 'mean_field':
    adj_iterable = range(8)

Tc_vs_adj = [[] for _ in range(len(adjacencies))]
Tc_err_vs_adj = [[] for _ in range(len(adjacencies))]
Tc_inf_vs_adj = [[] for _ in range(len(adjacencies))]
Tc_inf_err_vs_adj = [[] for _ in range(len(adjacencies))]

if write_file:
    file_simulation_info = open('simulation_info.txt', 'a')
    file_simulation_info.write('Adjacency & Size & Min. T & Max. T & Tempering copies & MCS & Replicas & Tc \\\\ \n')

for adj_index in adj_iterable:

    MCS_avg_vs_size, N_configs_vs_size, copies_vs_size, labels_vs_size, T_vs_size, q2_vs_size, q4_vs_size, \
        ql_vs_size, U_vs_size, U2_vs_size, σ2_q2_bin_vs_size, σ2_q4_bin_vs_size, q_dist_vs_size = \
        rfc.read_data(adjacencies[adj_index], distribution, sizes, add_vs_adj[adj_index],
                      T0_Tf_vs_adj[adj_index][0], T0_Tf_vs_adj[adj_index][1], MCS_avg_0,
                      [max_MCSs_vs_adj_binned[adj_index], max_MCSs_vs_adj_old[adj_index], max_MCSs_vs_adj_fast[adj_index]],
                      data_type='all', only_max_MCS=only_max_MCS)

    g_vs_size, g_bootstrap_vs_size, error_vs_size, dg_dT_vs_size, dg_dT_bootstrap_vs_size, error_dg_dT_vs_size = \
        sm.binder_cumulant_parallel(sizes_vs_adj[adj_index], T_vs_size, MCS_avg_vs_size, q2_vs_size, q4_vs_size, n_bootstrap)

    N_configs_vs_size_best, MCS_avg_vs_size_best, T_vs_size_best, g_vs_size_best, g_bootstrap_vs_size_best, \
        error_vs_size_best, dg_dT_vs_size_best, dg_dT_bootstrap_vs_size_best, error_dg_dT_vs_size_best = \
        sm.choose_optimal_MCS_N_config(sizes_vs_adj[adj_index], N_configs_vs_size, MCS_avg_vs_size, T_vs_size, g_vs_size,
                                       g_bootstrap_vs_size, error_vs_size, dg_dT_vs_size, dg_dT_bootstrap_vs_size,
                                       error_dg_dT_vs_size,
                                       MCS_N_config_condition='max_MCS_with_a_minimum_of_N_configs', min_N_config=1000)

    estimate_Tc_out = pf.estimate_Tc_with_pade_bootstrap_parallel(
        sizes_vs_adj[adj_index], T_vs_size_best, error_dg_dT_vs_size_best, dg_dT_bootstrap_vs_size_best,
        ic=ic_jc_vs_adj[adj_index][0], jc=ic_jc_vs_adj[adj_index][1],
        ntr=10, maxfev=10000, threads=min(n_bootstrap, cpu_count()-1))

    Tc_bootstrap, Tc_vs_adj[adj_index], Tc_err_vs_adj[adj_index] = \
        estimate_Tc_out[0], estimate_Tc_out[3], estimate_Tc_out[4]

    if graphs == 'mean_field':
        Tc_inf_vs_adj[adj_index], Tc_inf_err_vs_adj[adj_index]= \
            pf.extrapolate_thermodynamic_limit_mean_field_graphs(sizes_vs_adj[adj_index], Tc_bootstrap,
                                                                 inv_peak_height_bootstrap, peak_width_bootstrap)[:2]


    if write_file:
        for size_index in range(len(sizes)):
            add = add_vs_adj[adj_index]
            add_str = f'{add}'
            file_simulation_info.write(f'{adjacencies[adj_index]}{ bool(add>0)*add_str } & {sizes_vs_adj[adj_index][size_index]} & {T0} & {Tf} & '
                                       f'{copies_vs_size[size_index][0]} & {MCS_avg_vs_size_best[size_index]} & {N_configs_vs_size_best[size_index]}  \\\\ \n')

if write_file:
    file_simulation_info.close()

# %% Store processed data
fname = f'Processed_Data/Tc_vs_adj_{graphs}_read_mode={MCS_N_config_condition}_min_N_config={min_N_config}'
np.savez(fname,Tc_vs_adj=Tc_vs_adj, Tc_err_vs_adj=Tc_err_vs_adj, Tc_inf_vs_adj=Tc_inf_vs_adj,
         Tc_inf_err_vs_adj=Tc_inf_err_vs_adj,  allow_pickle=True)

#%% PROCESS DATA FOR FIGURES 7, 8 AND 13 - Autocorrelation times
pass
# %% Read autocorrelation time data vs adjacency
adj_indices = [0, 8, 9, 10]
T_vs_adj = [[] for _ in range(len(adjacencies))]
σ2_q2_bin_vs_adj = [[] for _ in range(len(adjacencies))]
tau_q2_T_vs_adj = [[] for _ in range(len(adjacencies))]
for adj_index in adj_indices:
    _, _, _, _, T_vs_adj[adj_index], _, _, _, _, _, σ2_q2_bin_vs_adj[adj_index] = \
        rfc.read_data(adjacencies[adj_index], distribution, sizes, add_vs_adj[adj_index], T0_Tf_vs_adj[adj_index][0],
        T0_Tf_vs_adj[adj_index][1], MCS_avg_0, [max_MCSs_vs_adj_binned[adj_index], _, _], data_type='binned')[:11]

# %% Calculate autocorrelation times
T_for_tau_vs_size = np.linspace(0.5, 5, 500)
start_range_vs_adj = [16, 21, 120, 160]
end_range_vs_adj = [90, 250, 350, 490]

for adj_index in adj_indices:
    tau_q2_T_vs_adj[adj_index] = rfc.autocorrelation_time_q2(σ2_q2_bin_vs_adj[adj_index])

# Calculate autocorrelation time fits
T_fit_tau_vs_adj, log_tau_fit_vs_adj, log_tau_vs_adj, log_tau_specific_fit_vs_adj, log_tau_specific_vs_adj, \
    T_tau_vs_adj = [ [[[] for _ in range(len(sizes))] for _ in range(len(adj_indices))] for _ in range(6)]

for adj_iterable, adj_index in enumerate(adj_indices):
    for size_index in range(len(sizes)):
        fit_start_index = np.where(np.isfinite(tau_q2_T_vs_adj[adj_index][size_index]))[0][0]

        T = T_vs_adj[adj_index][size_index][-1][fit_start_index:]
        T_tau_vs_adj[adj_iterable][size_index] = T
        T_fit_tau_vs_adj[adj_iterable][size_index] = np.linspace(T[0], T[-1], 1000)

        tau_specific = tau_q2_T_vs_adj[adj_index][size_index][fit_start_index:]/sizes_vs_adj[adj_index][size_index]
        tau = tau_q2_T_vs_adj[adj_index][size_index][fit_start_index:]

        log_tau_specific_vs_adj[adj_iterable][size_index] = np.log(tau_specific)
        log_tau_vs_adj[adj_iterable][size_index] = np.log(tau)

        log_tau_specific_fit_vs_adj[adj_iterable][size_index] = np.poly1d(np.polyfit(T, log_tau_specific_vs_adj[adj_iterable][size_index], 10))(T_fit_tau_vs_adj[adj_iterable][size_index])
        log_tau_fit_vs_adj[adj_iterable][size_index] = np.poly1d(np.polyfit(T, log_tau_vs_adj[adj_iterable][size_index], 10))(T_fit_tau_vs_adj[adj_iterable][size_index])


# Define scaling laws for tau(T) vs N
def power_scaling(x, a, c):
    return a*x + c
def exp_scaling(x, a, b, c):
    return a*x + b*np.exp(x) + c

# Calculate autocorrelation time vs size for different temperatures, extract fit parameters
log_tau_vs_size_for_specific_T_vs_adj = [[[] for _ in range(len(T_for_tau_vs_size))] for _ in range(len(adj_indices))]
power_scaling_params_log_tau_vs_size_for_specific_T_vs_adj = [[[] for _ in range(len(T_for_tau_vs_size))] for _ in
                                                              range(len(adj_indices))]
exp_scaling_params_log_tau_vs_size_for_specific_T_vs_adj = [[[] for _ in range(len(T_for_tau_vs_size))] for _ in
                                                            range(len(adj_indices))]
sizes_fit_specific_T_vs_adj = [[[] for _ in range(len(T_for_tau_vs_size))] for _ in range(len(adj_indices))]

for adj_iterable, adj_index in enumerate(adj_indices):
    for T_fit_index, T in enumerate(T_for_tau_vs_size):
        for size_index in range(len(sizes)):
            if np.any(np.abs(T_fit_tau_vs_adj[adj_iterable][size_index]-T) < 0.1):
                T_index = np.where(np.abs(T_fit_tau_vs_adj[adj_iterable][size_index]-T) == np.abs(T_fit_tau_vs_adj[adj_iterable][size_index]-T).min())[0][0]
                log_tau_vs_size_for_specific_T_vs_adj[adj_iterable][T_fit_index].append(log_tau_fit_vs_adj[adj_iterable][size_index][T_index])
            else:
                log_tau_vs_size_for_specific_T_vs_adj[adj_iterable][T_fit_index].append(np.nan)
        log_tau_vs_size = log_tau_vs_size_for_specific_T_vs_adj[adj_iterable][T_fit_index]

        try:
            non_thermalized_index = np.where(~np.isnan(log_tau_vs_size))[0][-1]
            power_scaling_params_log_tau_vs_size_for_specific_T_vs_adj[adj_iterable][T_fit_index] = \
                curve_fit(power_scaling, np.log(sizes_vs_adj[adj_index][:non_thermalized_index+1]), log_tau_vs_size[:non_thermalized_index+1])[0]
            exp_scaling_params_log_tau_vs_size_for_specific_T_vs_adj[adj_iterable][T_fit_index] = \
                curve_fit(exp_scaling, np.log(sizes_vs_adj[adj_index][:non_thermalized_index+1]), log_tau_vs_size[:non_thermalized_index+1])[0]
        except:
            power_scaling_params_log_tau_vs_size_for_specific_T_vs_adj[adj_iterable][T_fit_index] = \
                np.zeros([len(power_scaling_params_log_tau_vs_size_for_specific_T_vs_adj[adj_iterable][T_fit_index-1])])
            exp_scaling_params_log_tau_vs_size_for_specific_T_vs_adj[adj_iterable][T_fit_index] = \
                np.zeros([len(exp_scaling_params_log_tau_vs_size_for_specific_T_vs_adj[adj_iterable][T_fit_index-1])])

power_scaling_params_log_tau_vs_size_for_specific_T_vs_adj = np.array(power_scaling_params_log_tau_vs_size_for_specific_T_vs_adj)
exp_scaling_params_log_tau_vs_size_for_specific_T_vs_adj  = np.array(exp_scaling_params_log_tau_vs_size_for_specific_T_vs_adj)
# %% Store processed data
fname = f'Processed_Data/Autocorrelation_times_adjacencies={adj_indices}'
np.savez(fname, T_tau_vs_adj=T_tau_vs_adj, T_fit_tau_vs_adj=T_fit_tau_vs_adj,
         log_tau_vs_adj=log_tau_vs_adj, log_tau_fit_vs_adj=log_tau_fit_vs_adj,
         log_tau_specific_vs_adj=log_tau_specific_vs_adj, log_tau_specific_fit_vs_adj=log_tau_specific_fit_vs_adj,
         T_for_tau_vs_size=T_for_tau_vs_size, log_tau_vs_size_for_specific_T_vs_adj=log_tau_vs_size_for_specific_T_vs_adj,
         power_scaling_params_log_tau_vs_size_for_specific_T_vs_adj = power_scaling_params_log_tau_vs_size_for_specific_T_vs_adj,
         exp_scaling_params_log_tau_vs_size_for_specific_T_vs_adj=exp_scaling_params_log_tau_vs_size_for_specific_T_vs_adj,
         start_range_vs_adj=start_range_vs_adj, end_range_vs_adj=end_range_vs_adj, allow_pickle=True)

#%% Create optimal temperature distribution for parallel tempering
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
# ic = ic_jc_vs_adj[adj_index][0]
# jc = ic_jc_vs_adj[adj_index][1]
ic = list(range(2, 7))
jc = list(range(2, 7))

# error_dg_dT_vs_size_best = [(T_vs_size_best[size_index][1]-T_vs_size_best[size_index][0])/(np.sqrt(12)) * np.ones(copies_vs_size[size_index][-1]) for size_index in range(len(sizes))]
inopt = pf.pade_fss_analysis(sizes_vs_adj[adj_index], T_vs_size_best, dg_dT_vs_size_best, error_dg_dT_vs_size_best, ntr=10, ic=ic, jc=jc,
                             figsize_in=[16 / 1.5, 6 / 1.5], dpi_in=125, adjacency=adjacency, method_ic_jc='best')


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
# ax.errorbar(np.array([3, 5, 7, 9]), Tc_vs_adj[:4], error_Tc_vs_adj[:4], label=r'$k-$random regular, simulation',capsize=4, capthick=1,
#                  elinewidth=1, linewidth=1.25)
# ax.errorbar(np.array([3, 5, 7, 9]), Tc_vs_adj[4:-4], error_Tc_vs_adj[4:-4], label='Small world', capsize=4, capthick=1,
#                  elinewidth=1, linewidth=1.25)

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