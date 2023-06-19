import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
import matplotlib.pyplot as plt
import Modules.error_estimation as ee
import Modules.read_data_from_cluster as rfc
import Modules.pade_fits as pf
import glob
import sys
import importlib
from tqdm import tqdm
from time import perf_counter

importlib.reload(ee)
importlib.reload(rfc)
importlib.reload(pf)

# %% Parameters
distribution = 'gaussian_EA'
adjacencies = ['random_regular_3', 'random_regular_5', 'random_regular_7', 'random_regular_9', '1D+', '1D+', '1D+',
               '1D+',
               '2D_small_world', 'chimera', 'pegasus']
# sizes = [100, 200, 400, 800, 1600, 3200]
sizes = [100, 200, 400, 800, 1600]
add_vs_adj = [0, 0, 0, 0, 3.0, 5.0, 7.0, 9.0, 0, 0]
T0_vs_adj = [0.2, 0.5, 0.5, 1.0, 0.5, 1.0, 1.3, 1.5, 0.5, 0.2, 0.2]
Tf_vs_adj = [1.5, 3.0, 3.0, 4.0, 2.5, 2.5, 3.0, 3.5, 2.5, 3.0, 4.0]
MCS_avg_0 = 10000
max_MCSs_vs_adj = [MCS_avg_0 * 2 ** np.array([1, 2, 4, 6, 7, 7]),
                   MCS_avg_0 * 2 ** np.array([5, 5, 5, 5, 5, 5]) + 4,
                   # MCS_avg_0 * 2 ** np.array([1, 2, 4, 4, 5, 6]) + 1,
                   MCS_avg_0 * 2 ** np.array([3, 3, 4, 5, 6]) + 2,  # test, this should be off for analysis
                   MCS_avg_0 * 2 ** np.array([1, 2, 4, 5, 5, 6]),
                   MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 5, 6]),
                   # MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 5, 6]),
                   MCS_avg_0 * 2 ** np.array([3, 3, 4, 5, 6]) + 1,  # test, this should be off for analysis
                   MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 5, 6]),
                   MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 5, 6]),
                   MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 5, 6]),
                   MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 5, 6]),
                   MCS_avg_0 * 2 ** np.array([2, 2, 3, 4, 5, 6])]

adj_index = 0
adjacency = adjacencies[adj_index]
T0 = T0_vs_adj[adj_index]
Tf = Tf_vs_adj[adj_index]
add = add_vs_adj[adj_index]
max_MCSs = max_MCSs_vs_adj[adj_index]

plot_B = True
plot_σ2 = True
plot_e = False
plot_q2 = False

max_configs = -1

# %% Read data and create plots
plt.close('all')
if plot_B:
    fig_B, ax_B = plt.subplots(figsize=(8, 6), dpi=200)
if plot_B:
    fig_q2, ax_q2 = plt.subplots(figsize=(8, 6), dpi=200)
if plot_σ2:
    fig_σ2, ax_σ2 = plt.subplots(nrows=3, ncols=2, figsize=(8 * 2, 6 * 3), dpi=200)
    if len(sizes) > 1:
        ax_σ2 = ax_σ2.flatten()
    else:
        ax_σ2 = [ax_σ2]
if plot_e:
    fig_e, ax_e = plt.subplots(nrows=3, ncols=2, figsize=(8 * 2, 6 * 3), dpi=200)
    if len(sizes) > 1:
        ax_e = ax_e.flatten()
    else:
        ax_e = [ax_e]

T_vs_size, q2_vs_size, q4_vs_size, B_vs_size, ql_vs_size, U_vs_size, U2_vs_size, MCS_avg_vs_size, labels_vs_size, error_vs_size = \
    [[[] for _ in range(len(sizes))] for _ in range(10)]

for i, (size, max_MCS) in enumerate(zip(sizes, max_MCSs)):

    MCS_avg = MCS_avg_0
    if add == 0:
        fdir = f'Data/{adjacency}_{distribution},n={size},T={T0}_{Tf},MCS_avg={MCS_avg},max_MCS={max_MCS}'
    else:
        fdir = f'Data/{adjacency}_{distribution},n={size},T={T0}_{Tf},MCS_avg={MCS_avg},max_MCS={max_MCS},add={add}'
    T = np.loadtxt(f'{fdir}/T.dat')
    copies = len(T)
    T_vs_size[i].append(T)

    while True:
        file_type = f'{fdir}/MCS_avg={MCS_avg},seed=*.dat'
        files = glob.glob(file_type)
        if not bool(files):
            break
        else:
            MCS_avg_vs_size[i].append(MCS_avg)
            MCS_avg = 2 * MCS_avg

        labels = np.array([], dtype='int')
        µ_q2_t, µ_q4_t, σ2_q2_t, σ2_q4_t, µ_ql_t, µ_U_t, µ_U2_t = [np.empty([0, len(T)]) for _ in range(7)]

        for file in files:
            # data = np.genfromtxt(file, usecols=[i for i in range(len(T))], invalid_raise=False)
            f = open(file)
            data = f.read().splitlines()
            f.close()

            file_labels = np.array(data[0::8], dtype='int')
            µ_q2, µ_q4, σ2_q2, σ2_q4, µ_ql, µ_U, µ_U2 = [np.zeros([len(file_labels), copies]) for _ in range(7)]
            for r in range(len(file_labels)):
                µ_q2[r, :], µ_q4[r, :], σ2_q2[r, :], σ2_q4[r, :], µ_ql[r, :], µ_U[r, :], µ_U2[r, :] = \
                    [np.array(data[k + 1::8][r].split(), dtype='float') for k in range(7)]

            # Append the data from µ_q2 to µ_q2_c
            labels = np.append(labels, file_labels, axis=0)
            µ_q2_t = np.append(µ_q2_t, µ_q2, axis=0)  # Thermal averages of q2 for different configurations
            µ_q4_t = np.append(µ_q4_t, µ_q4, axis=0)  # Thermal averages of q4 for different configurations
            σ2_q2_t = np.append(σ2_q2_t, σ2_q2, axis=0)  # Thermal variances of q2 for different configurations
            σ2_q4_t = np.append(σ2_q4_t, σ2_q4, axis=0)  # Thermal variances of q4 for different configurations
            µ_ql_t = np.append(µ_ql_t, µ_ql, axis=0)  # Thermal averages of q4 for different configurations
            µ_U_t = np.append(µ_U_t, µ_U, axis=0)  # Thermal averages of q4 for different configurations
            µ_U2_t = np.append(µ_U2_t, µ_U2, axis=0)  # Thermal averages of q4 for different configurations

            current_config = µ_q2_t.shape[0]
            if current_config > max_configs != -1:
                break

        # del µ_q2, µ_q4, σ2_q2, σ2_q4, µ_ql, µ_U, µ_U2

        N_configs = µ_q2_t.shape[0]
        print(size, MCS_avg // 2, N_configs)

        µ_q2_c = np.mean(µ_q2_t, 0)  # Configuration average of the thermal averages of q2
        µ_q2_2_c = np.mean(µ_q2_t ** 2, 0)
        µ_q4_c = np.mean(µ_q4_t, 0)  # Configuration average of the thermal averages of q4
        µ_q4q2_c = np.mean(µ_q4_t / µ_q2_t ** 2, 0)
        µ_σ2_q2_t = np.mean(σ2_q2_t, 0)  # Configuration average of the thermal variances of q2
        µ_σ2_q4_t = np.mean(σ2_q4_t, 0)  # Configuration average of the thermal variances of q4
        µ_ql_c = np.mean(µ_ql_t, 0)  # Configuration average of the thermal averages of ql
        µ_U_c = np.mean(µ_U_t, 0)  # Configuration average of the thermal averages of U
        µ_U2_c = np.mean(µ_U2_t, 0)  # Configuration average of the thermal averages of U

        σ2_q2_c = np.var(µ_q2_t, 0)  # Configuration variance of the thermal averages of q2
        σ2_q4_c = np.var(µ_q4_t, 0)  # Configuration variance of the thermal averages of q4

        B = 0.5 * (3 - µ_q4_c / µ_q2_c ** 2)
        # B = 0.5 * (3 - µ_q4q2_c)
        # B = 0.5 * (3 - µ_q4_c / µ_q2_2_c)
        # B = size*µ_q2_c

        # Errors
        dBdq2 = 0.5 * 2 * µ_q4_c / µ_q2_c ** 3
        dBdq4 = -0.5 * 1 / µ_q2_c ** 2
        e1 = np.sqrt(σ2_q2_c * dBdq2 ** 2 + σ2_q4_c * dBdq4 ** 2) / np.sqrt(N_configs)

        d_q2 = np.std(µ_q2_t, 0)
        d_q4 = np.std(µ_q4_t, 0)
        e2 = np.sqrt((2 * d_q2 / µ_q2_c) ** 2 + (d_q4 / µ_q4_c) ** 2) / np.sqrt(N_configs)

        σ_B = np.zeros(copies)
        for k in range(copies):
            a = µ_q2_t[:, k] ** 2
            b = 0.5 * µ_q4_t[:, k]
            µ_a = np.mean(a)
            µ_b = np.mean(b)
            σ_ab = np.cov(a, b, bias=True)
            σ_B[k] = np.sqrt(σ_ab[0, 0] * (1 / µ_a) ** 2 + σ_ab[1, 1] * (1 / µ_b) ** 2 - 2 * σ_ab[0, 1] / (µ_a * µ_b))
        e3 = σ_B / np.sqrt(N_configs)

        # e4 = ee.bootstrap_error_B(µ_q2_t, µ_q4_t)
        B_error = e1

        B_vs_size[i].append(B)
        q2_vs_size[i].append(µ_q2_t)
        q4_vs_size[i].append(µ_q4_t)
        ql_vs_size[i].append(µ_ql_c)
        U_vs_size[i].append(µ_U_c)
        U2_vs_size[i].append(µ_U2_c)
        labels_vs_size[i].append(labels)
        error_vs_size[i].append(B_error)

        if plot_B and max_MCS - MCS_avg / 2 < 100:
            ax_B.errorbar(T, B, yerr=B_error, label=f'n = {size}', linewidth=0.5, capsize=2, capthick=0.5,
                          elinewidth=0.2)

        if plot_q2 and max_MCS - MCS_avg / 2 < 100:
            ax_q2.plot(T, µ_q2_c / size, label=f'n = {size}', linewidth=0.5)

        if plot_e and max_MCS - MCS_avg / 2 < 100:
            ax_e[i].plot(T, e1, 'r')
            ax_e[i].plot(T, e2, 'b')
            ax_e[i].plot(T, e3, 'g')
            # ax_e[i].plot(T, e4, 'k')
            ax_e[i].set_title(f'n={size}, N_configs = {N_configs}')

        if plot_σ2 and max_MCS - MCS_avg / 2 < 100:
            ax_σ2[i].plot(T, µ_σ2_q2_t, 'r', label='µ_σ2_q2_t')
            ax_σ2[i].plot(T, µ_σ2_q4_t, 'r:', label='µ_σ2_q4_t')
            ax_σ2[i].plot(T, σ2_q2_c, 'b', label='σ2_q2_c')
            ax_σ2[i].plot(T, σ2_q4_c, 'b:', label='σ2_q4_c')
            ax_σ2[i].set_xlabel('T')
            ax_σ2[i].set_title(f'n = {size}')
            ax_σ2[i].legend()

# %% Plot B
ax_B.set_xlabel('T')
ax_B.set_ylabel('g')
ax_B.set_ylim([0, 1])
ax_B.set_xlim([T0, Tf])
ax_B.legend()
# fig_B.suptitle(f'{adjacency},{distribution},add={add}')
fig_B.suptitle(f'{adjacency}, {distribution}')
fig_B.tight_layout()
fig_B.show()

# %% Plot q2
fig_q2, ax_q2 = plt.subplots(figsize=(8, 6), dpi=200)

for size, q2 in zip(sizes, q2_vs_size):
    ax_q2.plot(T, np.mean(q2[-1], 0), label=f'n = {size}', linewidth=0.5)
ax_q2.set_xlabel('T')
ax_q2.set_ylabel('q_2')
ax_q2.set_xlim([T0, Tf])
# ax_q2.set_ylim([0, 0.01])
ax_q2.legend()
fig_q2.suptitle(f'{adjacency},{distribution},add={add}')
fig_q2.tight_layout()
fig_q2.show()

# %% Plot σ2
fig_σ2.tight_layout()
fig_σ2.show()

# %% Plot errors
fig_e.tight_layout()
fig_e.show()

# %% B vs MCS_avg and vs N_configs
max_configs_0 = 100

T_vs_size, B_vs_size, _, _, _, N_configs_vs_size, MCS_avg_vs_size = \
    rfc.read_MC(adjacency, distribution, sizes, add, T0, Tf, MCS_avg_0, max_MCSs)

error_max_of_Nconf_MCS_vs_size = [np.zeros_like(B) for B in B_vs_size]

for size_idx in range(len(sizes)):
    max_configs = max_configs_0
    # B vs MCS_avg
    n_MCS = len(MCS_avg_vs_size[size_idx])
    fig, (ax1, ax2) = plt.subplots(ncols=2, dpi=150, figsize=[16 / 2, 9 / 2])

    [ax1.plot(T_vs_size[size_idx][0], np.abs(B_vs_size[size_idx][i] - B_vs_size[size_idx][-1]), '.-', linewidth=0.5,
              label=f'{MCS_avg_vs_size[size_idx][i]}') for i in range(len(MCS_avg_vs_size[size_idx]) - 1)]

    ax1.set_yscale('log')
    # [ax.plot(T, B_vs_size[size_idx][i], '.-', linewidth=0.5,
    #          label=f'{MCS_avg_vs_size[size_idx][i]}') for i in range(len(MCS_avg_vs_size[size_idx]))]
    # ax.set_ylim([0, 1])
    ax1.set_xlabel('$T$')
    ax1.set_title('Convergence vs $MCS_{avg}$')
    ax1.set_xlim([T0, Tf])
    ax1.legend()

    # B vs N_configs
    sizes_N = [sizes[size_idx]]
    max_MCSs_N = [max_MCSs[size_idx]]

    B_max = B_vs_size[size_idx][-1]
    N_configs_max = N_configs_vs_size[size_idx][-1]
    B = np.zeros(len(B_max))

    while True:
        T, B, _, _, _, N_configs, _ = rfc.read_MC(adjacency, distribution, sizes_N, add, T0, Tf, MCS_avg_0, max_MCSs_N,
                                                  max_configs)
        T = T[0][0]
        B = B[0][-1]
        max_configs *= 2

        if N_configs[0][-1] < N_configs_max:
            ax2.plot(T, np.abs(B - B_max), '.-', label=f'N_configs = {N_configs[0][-1]}')
        else:
            break

    ax2.legend()
    ax2.set_xlabel('$T$')
    ax2.set_title('Convergence vs $N_{configs}$')
    ax2.set_yscale('log')

    fig.tight_layout()
    fig.suptitle(f'size = {sizes[size_idx]}')
    fig.show()

# %% Single instance of B
fig, ax = plt.subplots(dpi=200)
size_idx = 1
label = labels_vs_size[size_idx][0][1]
n_MCS = len(MCS_avg_vs_size[size_idx])

for i in range(n_MCS):
    index = np.where(labels_vs_size[size_idx][i] == label)[0][0]
    # ax.plot(T, q2_vs_size[0][i][index], label=f'{MCS_avg_vs_size[0][i]}')
    ax.plot(T, 0.5 * (3 - q4_vs_size[size_idx][i][index] / q2_vs_size[size_idx][i][index] ** 2),
            label=f'{MCS_avg_vs_size[size_idx][i]}', linewidth=1)
ax.set_ylim([0, 1])
ax.set_xlim([T0, Tf])
ax.legend()
fig.show()

# %% Single instance of B
fig, ax = plt.subplots(dpi=200)
size_idx = -2
n_conf = len(labels_vs_size[size_idx][0])
n_MCS = len(MCS_avg_vs_size[size_idx])

error_N = np.zeros([n_conf, n_MCS - 1, copies])
for k, label in enumerate(labels_vs_size[size_idx][0]):
    for i in reversed(range(n_MCS)):
        index = np.where(labels_vs_size[size_idx][i] == label)[0][0]
        if i == n_MCS - 1:
            B_converged = 0.5 * (3 - q4_vs_size[size_idx][i][index] / q2_vs_size[size_idx][i][index] ** 2)
        else:
            error_N[k, i] = B_converged - 0.5 * (
                        3 - q4_vs_size[size_idx][i][index] / q2_vs_size[size_idx][i][index] ** 2)

# %%
fig, ax = plt.subplots(dpi=200)
x = np.arange(n_MCS - 1)[::-1]
ax.plot(np.abs(error_N.mean(0)[:,0]), 'b')
ax.plot(np.abs(error_N.mean(0)[:,14]), 'k')
ax.plot(np.abs(error_N.mean(0)[:,29]), 'r')
# ax.errorbar(x, error_N.mean(0)[:, 0], yerr=error_N.std(0)[:, 0], color='b')
# ax.errorbar(x, error_N.mean(0)[:, 14], yerr=error_N.std(0)[:, 14], color='k')
# ax.errorbar(x, error_N.mean(0)[:, 29], yerr=error_N.std(0)[:, 29], color='r')
# ax.set_ylim([0, 1])
# ax.set_yscale('log')
# ax.set_xlim([T0, Tf])
ax.legend()
fig.show()

# %% Convergence criteria
for size_idx in range(len(sizes)):
    fig, ax = plt.subplots(dpi=200)
    n_MCS = len(MCS_avg_vs_size[size_idx])
    for i in reversed(range(n_MCS)):
        U = U_vs_size[size_idx][i]
        ql = ql_vs_size[size_idx][i]
        T = T_vs_size[size_idx][0]
        conv = 1 - T * np.abs(U) / 3.55 - ql
        ax.plot(T, conv, label=f'{MCS_avg_vs_size[size_idx][i]}', linewidth=1)

    ax.set_title('Thermalization criteria vs MCS_avg, must be 0 if thermalized')
    # ax.set_ylim([0, 1])
    ax.legend()
    # ax.set_yscale('log')
    # ax.set_ylim(bottom=np.eps)
    ax.set_xlim([T0, Tf])
    fig.suptitle(f'size = {sizes[size_idx]}')

    fig.show()

# %% Cv
CvT_vs_size = []

T_vs_size, _, _, U_vs_size, _, _, _ = rfc.read_MC(adjacency, distribution, sizes, add, T0, Tf, MCS_avg_0, max_MCSs,
                                                  max_configs)

# %%
for i, size in enumerate(sizes):
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
# %% fit T_opt scaling
from scipy.optimize import curve_fit


def parabola(x, b, c):
    return b * x + c


params = np.zeros([2, copies])
for c in range(copies):
    params[:, c] = curve_fit(parabola, sizes, T_opt_vs_size[:, c])[0]

# %%
sizes_fit = np.linspace(sizes[0], sizes[-1], 100)

for k in [0, 5, 10, 15, 20, 25]:
    for c in range(0 + k, 5 + k):
        plt.plot(sizes, T_opt_vs_size[:, c], '.-')
        plt.plot(sizes_fit, parabola(sizes_fit, params[:, c][0], params[:, c][1]))
    plt.show()

# %%
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

# %% Convergence of error vs N_configs
Ncs = np.linspace(500, N_configs, 20).astype('int64')
copys = [0, 14, 29]
fig, ax = plt.subplots(nrows=1, ncols=len(copys), dpi=200, figsize=[30, 9])

for c, copy in enumerate(copys):
    Bc_vs_Nc = np.zeros(len(Ncs))
    Bc_error_vs_NC = np.zeros(len(Ncs))

    for n, Nc in enumerate(Ncs):
        µ_q2_c = np.mean(µ_q2_t[:Nc, copy], 0)
        µ_q4_c = np.mean(µ_q4_t[:Nc, copy], 0)
        σ2_q2_c = np.var(µ_q2_t[:Nc, copy], 0)
        σ2_q4_c = np.var(µ_q4_t[:Nc, copy], 0)
        d_q2 = np.std(µ_q2_t[:Nc, copy], 0)
        d_q4 = np.std(µ_q4_t[:Nc, copy], 0)

        Bc_vs_Nc[n] = 0.5 * (3 - µ_q4_c / µ_q2_c ** 2)

        # Error 1
        # dBdq2 = 0.5 * 2 * µ_q4_c / µ_q2_c ** 3
        # dBdq4 = -0.5 * 1 / µ_q2_c ** 2
        # Bc_error_vs_NC[n] =  np.sqrt(σ2_q2_c * dBdq2 ** 2 + σ2_q4_c * dBdq4 ** 2) / np.sqrt(Nc)

        # Error 2
        # Bc_error_vs_NC[n] = np.sqrt((2 * d_q2 / µ_q2_c) ** 2 + (d_q4 / µ_q4_c) ** 2) / np.sqrt(Nc)

        # Error 3
        # a = µ_q2_t[:Nc, copy] ** 2
        # b = 0.5 * µ_q4_t[:Nc, copy]
        # µ_a = np.mean(a)
        # µ_b = np.mean(b)
        # σ_ab = np.cov(a, b, bias=True)
        # σ_B = np.sqrt(σ_ab[0, 0] * (1 / µ_a) ** 2 + σ_ab[1, 1] * (1 / µ_b) ** 2 - 2 * σ_ab[0, 1] / (µ_a * µ_b))
        # Bc_error_vs_NC[n] = σ_B / np.sqrt(Nc)

        # Error 4
        Bc_error_vs_NC[n] = ee.bootstrap_error_B(µ_q2_t[:Nc, :], µ_q4_t[:Nc, :])[copy]

    try:
        # ax[c].errorbar(Ncs, Bc_vs_Nc, yerr=Bc_error_vs_NC)
        ax[c].plot(Ncs, Bc_vs_Nc)
        ax[c].set_ylabel(f'B(T={T[copy]})')
        ax[c].set_xlabel('N_configs')
    except:
        # ax.errorbar(Ncs, Bc_vs_Nc, yerr=Bc_error_vs_NC)
        ax.plot(Ncs, Bc_vs_Nc)
        ax.set_ylabel(f'B(T={T[copy]})')
        ax.set_xlabel('N_configs')

fig.tight_layout()
# fig.savefig(f'Figures/{adjacency}_{distribution},n={sizes},T={T0}_{Tf},MCS_avg={MCS_avg},max_MCS={max_MCS},'
#             f'add={add},B_vs_N_config.pdf')
fig.show()

# %% Estimate error of non thermalization by convergence     - Development 1

size_index = -1
r = np.log2(max_MCS / MCS_avg_0).astype('int')
MCS_eq = [max_MCS / 2 ** k for k in reversed(range(r + 1))]

fig, ax = plt.subplots(figsize=(16 / 1.5, 16 / 1.5), dpi=125)
for T_index in range(len(T_vs_size[size_index][0])):
    ax.plot(MCS_eq, np.array([B_Ti[T_index] for B_Ti in B_vs_size[size_index]]), '.-')
    ax.set_xscale('log')
fig.show()


# %%        - Development 2
def f_conv_vs_MCS(x, a, b, c):
    return a - b * np.exp(-x ** c)


size_index = -1
skip_initial = 0
T_index = 0

B_vs_MCS = np.array(B_vs_size[size_index])
max_MCS = max_MCSs[size_index]

r = np.log2(max_MCS / MCS_avg_0).astype('int')
MCS_eq = [max_MCS / 2 ** k for k in reversed(range(r + 1))]

# params = curve_fit(f_conv_vs_MCS, MCS_eq[2:], np.array([B_Ti[T_index] for B_Ti in B_vs_size[size_index][2:]]),
#                  p0=[1,1,0.000001], bounds=([0, -np.inf,-np.inf],[1, np.inf,np.inf]))[0]

params = curve_fit(f_conv_vs_MCS, MCS_eq[skip_initial:], B_vs_MCS[skip_initial:, T_index],
                   p0=[1, 1, 0.000001], bounds=([0, -np.inf, -np.inf], [1, np.inf, np.inf]))[0]
print(params)
print(np.abs(params[0] - B_vs_size[size_index][-1][T_index]))

x = np.geomspace(MCS_eq[0], MCS_eq[-1] * 10, 1000)
fig, ax = plt.subplots(figsize=(16 / 1.5, 16 / 1.5), dpi=125)
ax.plot(x, f_conv_vs_MCS(x, *params))
ax.plot(MCS_eq, np.array([B_Ti[T_index] for B_Ti in B_vs_size[size_index]]), '.')
ax.set_xscale('log')
fig.show()

# %%        - Tests 1
B_extrapolated_vs_size, error_extrapolated_vs_size = rfc.extrapolate_convergence(B_vs_size, error_vs_size, MCS_avg_0,
                                                                                 max_MCSs, skip_initial_MCS_0=0)

size_index = -1
fig, ax = plt.subplots(figsize=(8 / 1.5, 8 / 1.5), dpi=125)
ax.plot(T_vs_size[size_index][0], B_extrapolated_vs_size[size_index])
ax.plot(T_vs_size[size_index][0], B_vs_size[size_index][-1], 'r')
ax.set_ylim([0, 1])
fig.show()

fig, ax = plt.subplots(figsize=(8 / 1.5, 8 / 1.5), dpi=125)
ax.plot(T_vs_size[size_index][0], error_extrapolated_vs_size[size_index])
ax.plot(T_vs_size[size_index][0], error_vs_size[size_index][-1], 'r')
fig.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16 / 1.5, 8 / 1.5), dpi=125)
for size_index in range(len(sizes)):
    ax2.plot(T_vs_size[size_index][0], B_extrapolated_vs_size[size_index])
    ax1.plot(T_vs_size[size_index][0], B_vs_size[size_index][-1])
ax1.set_ylim([0, 1])
ax2.set_ylim([0, 1])
fig.show()

fig, ax = plt.subplots(figsize=(8 / 1.5, 16 / 1.5), dpi=125)
ax.plot(T_vs_size[size_index][0], np.array(error_extrapolated_vs_size).T)
fig.show()

# %% Pade fits and pade errors  - Development 1
Lfit = sizes
sizes = np.array(sizes)
T_term = 0
ntr = 5
ic = [5]
jc = [6]

T_term_ind = [np.where(T[0] > T_term)[0][0] for T in T_vs_size]

try:
    Tfit = [T[0][k:] for T, k in zip(T_vs_size, T_term_ind)]
except:
    Tfit = [T[k:] for T, k in zip(T_vs_size, T_term_ind)]

try:
    Ofit = [B[-1][k:] for B, k in zip(B_vs_size, T_term_ind)]
except:
    Ofit = [B[k:] for B, k in zip(B_vs_size, T_term_ind)]

try:
    Ofit_er = [err[-1][k:] for err, k in zip(error_vs_size, T_term_ind)]
except:
    Ofit_er = [err[k:] for err, k in zip(error_vs_size, T_term_ind)]

nf = len(sizes)

fT, fT_d, l_rchi = pf.pade_best(Tfit[0:nf], Ofit[0:nf], np.array(Ofit_er[0:nf]), ntr=ntr, ic=ic, jc=jc)

# %% Obtain T_c with crossing of functions
x = np.linspace(T0 + (Tf - T0) * 0.1, Tf - (Tf - T0) * 0.1, 100000)
Tc_cross = []

sizes_for_Tc_crossing = [0, 1, 2, 3]
for i in sizes_for_Tc_crossing:
    diff = np.abs(fT[i](x) - fT[i + 1](x))
    Tc_cross.append(x[np.where(diff == diff.min())[0][0]])
xx = np.linspace(0, 0.25, 100)
z = np.polyfit(1 / sizes[sizes_for_Tc_crossing] ** (1 / 3), Tc_cross, 1)
# z = np.polyfit(1/np.log(sizes[sizes_for_Tc_crossing]), Tc_cross, 1)
p = np.poly1d(z)
plt.plot(1 / sizes[sizes_for_Tc_crossing] ** (1 / 3), Tc_cross, '.')
# plt.plot(1/np.log(sizes[sizes_for_Tc_crossing]),Tc_cross, '.')
plt.plot(xx, p(xx), "-")
plt.show()
# %% plot pade fits
color = ['turquoise', 'tab:olive', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
         'tab:blue', 'goldenrod', 'tab:orange', 'tab:red']
fig, ax1 = plt.subplots(dpi=500)
# x = np.linspace(T0 + (Tf - T0) * 0.1, Tf - (Tf - T0) * 0.1, 1000)
x = np.linspace(T0, Tf, 1000)
for i in range(nf):
    ax1.errorbar(Tfit[i], Ofit[i], yerr=Ofit_er[i], markerfacecolor="None", capsize=1, capthick=0.5,
                 elinewidth=0.5, linewidth=1, ls='', marker='+', markersize=2,
                 label=" size" + str(Lfit[i]), color=color[i], alpha=0.7)

    # ax1.errorbar((Tfit[i]-2)*sizes[i]**(1/3), Ofit[i], yerr=Ofit_er[i],  markerfacecolor="None", capsize=1, capthick=0.5,
    #              elinewidth=0.5, linewidth = 0.5,
    #              label=" size" + str(Lfit[i]))
    ax1.plot(x, fT[i](x), linewidth=0.5, color=color[i])
    # ax1.plot((x-1.9957)*sizes[i]**(1/3), fT[i](x), linewidth = 0.5)
    # ax1.plot(sizes[i]**(1/3)*(x-Tc*0.94), fT[i](x), linewidth = 0.5)
# ax1.set_xlim(-0.5,0.5)
# ax1.set_ylim(0.1,0.6)
ax1.set_xlim(T0, Tf)
ax1.set_ylim(0, 1)
ax1.set_xlabel('$T$')
ax1.set_ylabel("$g$")
fig.suptitle('B')
fig.tight_layout()
# ax1.legend()
fig.show()
# %%        - Development 2

T_max = np.zeros(len(sizes))
sizes_fit = np.array(sizes)
for i in range(len(sizes)):
    T_max[i] = minimize_scalar(fT_d[i], method='brent', bracket=(T0 + 0.1, T0)).x
    if T_max[i] < T0 + 0.1 or T_max[i] > Tf - 0.5:
        T_max[i] = minimize_scalar(fT_d[i], method='bounded', bounds=(T0 + 0.1, Tf - 0.5)).x
    if i > 0 and np.abs(T_max[i] - T_max[i - 1]) / T_max[i - 1] > 0.2:
        T_max[i] = -1
        sizes_fit[i] = -1
T_max = T_max[T_max > 0]
sizes_fit = sizes_fit[sizes_fit > 0]
z = np.polyfit(1 / sizes_fit ** (1 / 3), T_max, 1)

print(pf.estimate_Tc_with_pade(sizes, T0, Tf, fT_d))

# %%        - Test 1: obtain B_bootstrap_vs_size
T_vs_size, error_vs_size, B_bootstrap_vs_size = rfc.B_bootstrap_max_MCS(adjacency, distribution, sizes, add, T0, Tf,
                                                                        MCS_avg_0, max_MCSs, n_bootstrap=800)

# %%        - Development 3
Lfit = sizes
sizes = np.array(sizes)
T_term = 0
ntr = 2
ic = [5]
jc = [6]

T_term_ind = [np.where(T[0] > T_term)[0][0] for T in T_vs_size]
n_bootstrap = len(B_bootstrap_vs_size[0])
Tc_bootstrap = np.zeros(n_bootstrap)

try:
    Tfit = [T[0][k:] for T, k in zip(T_vs_size, T_term_ind)]
except:
    Tfit = [T[k:] for T, k in zip(T_vs_size, T_term_ind)]
T0 = Tfit[0][0]
Tf = Tfit[0][-1]
nf = len(sizes)

t = perf_counter()
try:
    Ofit_er = [err[-1][k:] for err, k in zip(error_vs_size, T_term_ind)]
except:
    Ofit_er = [err[k:] for err, k in zip(error_vs_size, T_term_ind)]

for i_b in tqdm(range(n_bootstrap)):
    Ofit_bootstrap = [B[i_b][k:] for B, k in zip(B_bootstrap_vs_size, T_term_ind)]

    try:
        fT_d = pf.pade_best_fast(Tfit, Ofit_bootstrap, np.array(Ofit_er), ic=ic, jc=jc, ntr=1, maxfev=500000)
        # fT_d = pf.pade_best_fast(Tfit, Ofit_bootstrap, np.array(Ofit_er), ic=ic, jc=jc, ntr=1, maxfev=100000)
        Tc_bootstrap[i_b] = pf.estimate_Tc_with_pade(sizes, T0, Tf, fT_d)
    except:
        Tc_bootstrap[i_b] = np.nan
print(f'Percentage of good results: {100 * (len(Tc_bootstrap) - np.isnan(Tc_bootstrap).sum()) / len(Tc_bootstrap)}')
print('Elapsed time', perf_counter() - t)

# %%        - Test 2: Tc and error Tc with bootstrap using single core
Tc_bootstrap = pf.estimate_Tc_with_pade_bootstrap(sizes, T_vs_size, error_vs_size, B_bootstrap_vs_size,
                                                  T_term=0, ic=[5], jc=[6], ntr=1, maxfev=10000)

print(np.nanmean(Tc_bootstrap), np.nanstd(Tc_bootstrap))

# %%        - Test 3: Tc and error Tc with bootstrap using multicore
Tc_bootstrap = pf.estimate_Tc_with_pade_bootstrap_parallel(sizes, T_vs_size, error_vs_size, B_bootstrap_vs_size,
                                                           T_term=0, ic=[5],
                                                           jc=[6], ntr=1, maxfev=10000, threads=8)
print(np.nanmean(Tc_bootstrap), np.nanstd(Tc_bootstrap))
# %%        - Tests 4: pade_fss_analysis

# pf.pade_fss_analysis(sizes, T_vs_size, B_extrapolated_vs_size, error_extrapolated_vs_size, T_term=0,  \
# ntr=4, ic=[5], jc=[6], figsize_in=(16/1.5,6/1.5), dpi_in=125)

pf.pade_fss_analysis(sizes, T_vs_size, B_vs_size, error_vs_size, T_term=0, ntr=4, ic=[5], jc=[6],
                     figsize_in=[16 / 1.5, 6 / 1.5], dpi_in=125)

# %% Tc vs adjacency
Tc_bootstrap_vs_adj = [[] for _ in range(8)]

n_bootstrap = 8
T_term = 0
ic = [5]
jc = [6]
ntr = 1
maxfev = 10000
# maxfev = 500000
threads = 8

for adj_index in range(8):
    adjacency = adjacencies[adj_index]
    T0 = T0_vs_adj[adj_index]
    Tf = Tf_vs_adj[adj_index]
    add = add_vs_adj[adj_index]
    max_MCSs = max_MCSs_vs_adj[adj_index]
    print(adjacency)

    T_vs_size, error_vs_size, B_bootstrap_vs_size = rfc.B_bootstrap_max_MCS(adjacency, distribution, sizes, add, T0, Tf,
                                                                            MCS_avg_0, max_MCSs, n_bootstrap)
    Tc_bootstrap = pf.estimate_Tc_with_pade_bootstrap_parallel(sizes, T_vs_size, error_vs_size, B_bootstrap_vs_size,
                                                               T_term, ic, jc, ntr, maxfev, threads)
    Tc_bootstrap_vs_adj[adj_index] = Tc_bootstrap

# %%
Tc_vs_adj = []
error_Tc_vs_adj = []
for Tc_bootstrap in Tc_bootstrap_vs_adj:
    Tc = np.nanmean(Tc_bootstrap)
    Tc_vs_adj.append(Tc)
    # error_Tc_vs_adj.append(np.nanstd(Tc_bootstrap))
    error_Tc_vs_adj.append(np.nanstd(Tc_bootstrap[np.all((Tc - 0.1 < Tc_bootstrap, Tc_bootstrap < Tc + 0.1), 0)]))

# %%
Tc_x = np.linspace(0, 3, 10000)
x = np.linspace(-100, 100, 1000)
k = np.zeros([len(Tc_x)])
mu = 0
std = 0.93
for i, Tc in enumerate(Tc_x):
    y = np.tanh(x / Tc) ** 2 * np.exp(-0.5 * ((x - mu) / std) ** 2) / (np.sqrt(2 * np.pi) * std)
    k[i] = 1 / np.trapz(y, x)

fig, ax = plt.subplots(dpi=200)
ax.plot(k + 1, Tc_x)
ax.errorbar([5, 7], [Tc_bootstrap.mean(), 1.99], [Tc_bootstrap.std(), 0.02])
# ax.errorbar(np.array([3, 5, 7, 9]), Tc_vs_adj[:4], error_Tc_vs_adj[:4], linewidth=1, label='Random Regular Graphs')
# ax.errorbar(np.array([3, 5, 7, 9]), Tc_vs_adj[4:], error_Tc_vs_adj[4:], linewidth=1,  label='1D small world')
ax.legend()
ax.set_xlabel('Number of neighbors')
ax.set_ylabel('Tc')
fig.show()

# %%
fig, axs = plt.subplots(ncols=4, nrows=2, dpi=100, figsize=[16, 9])
axs = axs.flatten()
for ax, Tc_bootstrap, adjacency in zip(axs, Tc_bootstrap_vs_adj, adjacencies):
    Tc = np.nanmean(Tc_bootstrap)
    ax.hist(Tc_bootstrap, 100, range=[Tc - 0.1, Tc + 0.1])
    ax.set_title(f'{adjacency}, n_bootstrap = {len(np.isnan(Tc_bootstrap)) - np.isnan(Tc_bootstrap).sum()}')
fig.tight_layout()
plt.show()
