import numpy as np
from joblib import Parallel, delayed, cpu_count
import Modules.pade_fits as pf
from scipy.optimize import root_scalar, minimize_scalar

# %% Calculate binder cumulant, error and bootstrap
def binder_cumulant_and_error_bootstrap(T, µ_q2_t, µ_q4_t, n_bootstrap=1000, error_type='1'):
    try:
        N = min(µ_q4_t.shape[0], µ_q2_t.shape[0])
        copies = µ_q2_t.shape[1]
    except:
        return [], [], [], [], [], []
    g_bootstrap = np.zeros([n_bootstrap, copies])
    dg_dT_bootstrap = np.zeros([n_bootstrap, copies])
    # g_bootstrap_extrapolated = np.zeros([n_bootstrap, copies])

    g = 0.5 * (3 - np.mean(µ_q4_t, 0) / (np.mean(µ_q2_t, 0) ** 2))
    dg_dT = np.gradient(g, T)
    # g = 0.5 * (3 - np.mean(µ_q4_t / (µ_q2_t ** 2), 0))

    for i in range(n_bootstrap):
        bootstrap_indices = np.random.randint(N, size=N)
        µ_q2_t_bootstrap = µ_q2_t[bootstrap_indices, :]
        µ_q4_t_bootstrap = µ_q4_t[bootstrap_indices, :]

        g_bootstrap[i, :] = 0.5 * (3 - np.mean(µ_q4_t_bootstrap, 0) / (np.mean(µ_q2_t_bootstrap, 0) ** 2))
        # g_bootstrap[i, :] = 0.5 * (3 - np.mean(µ_q4_t_bootstrap / (µ_q2_t_bootstrap ** 2), 0))
        dg_dT_bootstrap[i, :] = np.gradient(g_bootstrap[i, :], T)

    if error_type == '1' or error_type == 'all':
        e1 = np.std(g_bootstrap, 0)
        g_error = e1
        dg_dT_error = np.std(dg_dT_bootstrap, 0)

    if error_type == '2' or error_type == 'all':
        µ_q2_c = np.mean(µ_q2_t, 0)  # Configuration average of the thermal averages of q2
        µ_q4_c = np.mean(µ_q4_t, 0)  # Configuration average of the thermal averages of q4
        σ2_q2_c = np.var(µ_q2_t, 0)  # Configuration variance of the thermal averages of q2
        σ2_q4_c = np.var(µ_q4_t, 0)  # Configuration variance of the thermal averages of q4

        dBdq2 = 0.5 * 2 * µ_q4_c / µ_q2_c ** 3
        dBdq4 = -0.5 * 1 / µ_q2_c ** 2

        e2 = np.sqrt(σ2_q2_c * dBdq2 ** 2 + σ2_q4_c * dBdq4 ** 2) / np.sqrt(N)
        g_error = e2

    if error_type == '3' or error_type == 'all':
        µ_q2_c = np.mean(µ_q2_t, 0)  # Configuration average of the thermal averages of q2
        µ_q4_c = np.mean(µ_q4_t, 0)  # Configuration average of the thermal averages of q4
        d_q2 = np.std(µ_q2_t, 0)
        d_q4 = np.std(µ_q4_t, 0)
        e3 = np.sqrt((2 * d_q2 / µ_q2_c) ** 2 + (d_q4 / µ_q4_c) ** 2) / np.sqrt(N)
        g_error = e3

    if error_type == '4' or error_type == 'all':
        e4 = np.zeros(copies)
        for k in range(copies):
            a = µ_q2_t[:, k] ** 2
            b = 0.5 * µ_q4_t[:, k]
            µ_a = np.mean(a)
            µ_b = np.mean(b)
            σ_ab = np.cov(a, b, bias=True)
            e4[k] = np.sqrt(
                σ_ab[0, 0] * (1 / µ_a) ** 2 + σ_ab[1, 1] * (1 / µ_b) ** 2 - 2 * σ_ab[0, 1] / (µ_a * µ_b)) / np.sqrt(N)
        g_error = e4

    if error_type == 'all':
        g_error = [e1, e2, e3, e4]

    return g, g_bootstrap, g_error, dg_dT, dg_dT_bootstrap, dg_dT_error

# %% Parallelize previous function
def binder_cumulant_parallel(sizes, T_vs_size, MCS_avg_vs_size, q2_vs_size, q4_vs_size, n_bootstrap):

    g_vs_size, g_bootstrap_vs_size, error_vs_size, dg_dT_vs_size, dg_dT_bootstrap_vs_size, error_dg_dT_vs_size = \
        [[[[] for _ in range(len(MCS_avg_vs_size[i]))] for i in range(len(sizes))] for _ in range(6)]

    n_cases = sum([len(MCS_avg) for MCS_avg in MCS_avg_vs_size])

    results_g_parallel = Parallel(n_jobs=min(cpu_count(), n_cases))(delayed(binder_cumulant_and_error_bootstrap)
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
                dg_dT_vs_size[size_ind][MCS_ind], dg_dT_bootstrap_vs_size[size_ind][MCS_ind], \
            error_dg_dT_vs_size[size_ind][MCS_ind] = results_g_parallel[k]
            k += 1
    del results_g_parallel

    return g_vs_size, g_bootstrap_vs_size, error_vs_size, dg_dT_vs_size, dg_dT_bootstrap_vs_size, error_dg_dT_vs_size

# %% Choose the optimal simulation in terms of MCS and N_configs
def choose_optimal_MCS_N_config(sizes, N_configs_vs_size, MCS_avg_vs_size, T_vs_size, g_vs_size, g_bootstrap_vs_size,
                                error_vs_size, dg_dT_vs_size, dg_dT_bootstrap_vs_size, error_dg_dT_vs_size,
                                MCS_N_config_condition='max_MCS_with_a_minimum_of_N_configs', min_N_config=1000):

    N_configs_vs_size_best, MCS_avg_vs_size_best, T_vs_size_best, g_vs_size_best, g_bootstrap_vs_size_best, error_vs_size_best, \
        dg_dT_vs_size_best, dg_dT_bootstrap_vs_size_best, error_dg_dT_vs_size_best = \
        [[[] for _ in range(len(sizes))] for _ in range(9)]

    for size_index in range(len(sizes)):
        assigned = False
        k = -1
        while not assigned:
            match MCS_N_config_condition:
                case 'max_MCS_with_a_minimum_of_N_configs':
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

        N_configs_vs_size_best[size_index] = N_configs_vs_size[size_index][k]
        MCS_avg_vs_size_best[size_index] = MCS_avg_vs_size[size_index][k]
        T_vs_size_best[size_index] = T_vs_size[size_index][k]
        g_vs_size_best[size_index] = g_vs_size[size_index][k]
        g_bootstrap_vs_size_best[size_index] = g_bootstrap_vs_size[size_index][k]
        error_vs_size_best[size_index] = error_vs_size[size_index][k]
        dg_dT_vs_size_best[size_index] = dg_dT_vs_size[size_index][k]
        dg_dT_bootstrap_vs_size_best[size_index] = dg_dT_bootstrap_vs_size[size_index][k]
        error_dg_dT_vs_size_best[size_index] = error_dg_dT_vs_size[size_index][k]

    return N_configs_vs_size_best, MCS_avg_vs_size_best, T_vs_size_best, g_vs_size_best, \
        g_bootstrap_vs_size_best, error_vs_size_best, \
        dg_dT_vs_size_best, dg_dT_bootstrap_vs_size_best, error_dg_dT_vs_size_best
