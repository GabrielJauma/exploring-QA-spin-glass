import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from scipy.optimize import curve_fit
import glob

# %% Read data with a specific size and value of MCS
def read_data_specific_size_and_MCS(adjacency, distribution, size, add, T0, Tf, MCS_avg, MCS_avg_0, max_MCS,
                                    data_type, max_configs=-1, n_q_dist=50):

    match data_type:
        case 'binned':
            if add == 0:
                fdir = f'Data/{adjacency}_{distribution},n={size},T={T0}_{Tf},MCS_avg={MCS_avg_0},max_MCS={max_MCS},binned'
            else:
                fdir = f'Data/{adjacency}_{distribution},n={size},T={T0}_{Tf},MCS_avg={MCS_avg_0},max_MCS={max_MCS},add={add},binned'
            try:
                T = np.loadtxt(f'{fdir}/T.dat')
            except:
                # print(f'No data for {adjacency},n={size},max_MCS={max_MCS},add={add},binned.\n')
                return size, MCS_avg, [], [], [], [], [], [], [], [], [], [], []
            copies = len(T)

            # while True:
            file_type = f'{fdir}/MCS_avg={MCS_avg},seed=*.csv'
            files = glob.glob(file_type)

            n_bins = np.log2(MCS_avg).astype('int')
            lines_per_config = 2 + n_bins * 2 + 3 + n_q_dist + 2

            for file in files:
                data = np.array(pd.read_csv(file))
                if file == files[0]:
                    n_lines = data.shape[0]
                    configs = (n_lines + 1) // lines_per_config
                    label_indices = np.arange(0, n_lines, lines_per_config)
                    µ_q2_indices = np.arange(1, n_lines, lines_per_config)
                    µ_q4_indices = np.arange(2, n_lines, lines_per_config)
                    σ2_q2_indices = np.array(
                        [np.arange(k, k + n_bins) for k in range(3, n_lines, lines_per_config)]).flatten()
                    σ2_q4_indices = np.array(
                        [np.arange(k, k + n_bins) for k in range(3 + n_bins, n_lines, lines_per_config)]).flatten()
                    µ_ql_indices = np.arange(3 + 2 * n_bins, n_lines, lines_per_config)
                    µ_U_indices = np.arange(4 + 2 * n_bins, n_lines, lines_per_config)
                    µ_U2_indices = np.arange(5 + 2 * n_bins, n_lines, lines_per_config)
                    q_dist_indices = np.array([np.arange(k, k + n_q_dist) for k in
                                               range(6 + 2 * n_bins, n_lines, lines_per_config)]).flatten()
                    n_lines_0 = n_lines

                    labels = data[label_indices, 0]
                    µ_q2_t = data[µ_q2_indices, :]
                    µ_q4_t = data[µ_q4_indices, :]
                    σ2_q2_bin_t = data[σ2_q2_indices, :].reshape([configs, n_bins, copies])
                    σ2_q4_bin_t = data[σ2_q4_indices, :].reshape([configs, n_bins, copies])
                    µ_ql_t = data[µ_ql_indices, :]
                    µ_U_t = data[µ_U_indices, :]
                    µ_U2_t = data[µ_U2_indices, :]
                    q_dist_t = data[q_dist_indices, :].reshape([configs, n_q_dist, copies]).astype('int')

                else:
                    # Append the data from µ_q2 to µ_q2_c
                    n_lines = data.shape[0]
                    if n_lines != n_lines_0:
                        n_lines = data.shape[0]
                        configs = (n_lines + 1) // lines_per_config
                        label_indices = np.arange(0, n_lines, lines_per_config)
                        µ_q2_indices = np.arange(1, n_lines, lines_per_config)
                        µ_q4_indices = np.arange(2, n_lines, lines_per_config)
                        σ2_q2_indices = np.array(
                            [np.arange(k, k + n_bins) for k in range(3, n_lines, lines_per_config)]).flatten()
                        σ2_q4_indices = np.array(
                            [np.arange(k, k + n_bins) for k in range(3 + n_bins, n_lines, lines_per_config)]).flatten()
                        µ_ql_indices = np.arange(3 + 2 * n_bins, n_lines, lines_per_config)
                        µ_U_indices = np.arange(4 + 2 * n_bins, n_lines, lines_per_config)
                        µ_U2_indices = np.arange(5 + 2 * n_bins, n_lines, lines_per_config)
                        q_dist_indices = np.array([np.arange(k, k + n_q_dist) for k in
                                                   range(6 + 2 * n_bins, n_lines, lines_per_config)]).flatten()
                        n_lines_0 = n_lines

                    labels = np.append(labels, data[label_indices, 0], axis=0)
                    µ_q2_t = np.vstack([µ_q2_t, data[µ_q2_indices, :]])
                    µ_q4_t = np.vstack([µ_q4_t, data[µ_q4_indices, :]])
                    σ2_q2_bin_t = np.vstack([σ2_q2_bin_t, data[σ2_q2_indices, :].reshape([configs, n_bins, copies])])
                    σ2_q4_bin_t = np.vstack([σ2_q4_bin_t, data[σ2_q4_indices, :].reshape([configs, n_bins, copies])])
                    µ_ql_t = np.vstack([µ_ql_t, data[µ_ql_indices, :]])
                    µ_U_t = np.vstack([µ_U_t, data[µ_U_indices, :]])
                    µ_U2_t = np.vstack([µ_U2_t, data[µ_U2_indices, :]])
                    q_dist_t = np.vstack(
                        [q_dist_t, data[q_dist_indices, :].reshape([configs, n_q_dist, copies]).astype('int')])

                current_config = len(labels)
                if current_config > max_configs != -1:
                    break
            N_configs = µ_q2_t.shape[0]
            print(size, MCS_avg, N_configs)

            return size, MCS_avg, N_configs, labels, T, µ_q2_t, µ_q4_t, µ_ql_t, µ_U_t, µ_U2_t, \
                σ2_q2_bin_t, σ2_q4_bin_t, q_dist_t[:, :, ::-1]

        case 'fast':
            if add == 0:
                fdir = f'Data/{adjacency}_{distribution},n={size},T={T0}_{Tf},max_MCS={max_MCS},fast'
            else:
                fdir = f'Data/{adjacency}_{distribution},n={size},T={T0}_{Tf},max_MCS={max_MCS},add={add},fast'
            try:
                T = np.loadtxt(f'{fdir}/T.dat')
            except:
                # print(f'No data for {adjacency},n={size},max_MCS={max_MCS},add={add},fast.\n')
                return size, MCS_avg, [], [], [], [], []

            # while True:
            file_type = f'{fdir}/MCS_avg={MCS_avg},seed=*.csv'
            files = glob.glob(file_type)

            lines_per_config = 4

            for file in files:
                data = np.array(pd.read_csv(file))
                if file == files[0]:
                    n_lines = data.shape[0]
                    label_indices = np.arange(0, n_lines, lines_per_config)
                    µ_q2_indices = np.arange(1, n_lines, lines_per_config)
                    µ_q4_indices = np.arange(2, n_lines, lines_per_config)
                    n_lines_0 = n_lines

                    labels = data[label_indices, 0]
                    µ_q2_t = data[µ_q2_indices, :]
                    µ_q4_t = data[µ_q4_indices, :]

                else:
                    # Append the data from µ_q2 to µ_q2_c
                    n_lines = data.shape[0]
                    if n_lines != n_lines_0:
                        n_lines = data.shape[0]
                        label_indices = np.arange(0, n_lines, lines_per_config)
                        µ_q2_indices = np.arange(1, n_lines, lines_per_config)
                        µ_q4_indices = np.arange(2, n_lines, lines_per_config)
                        n_lines_0 = n_lines

                    labels = np.append(labels, data[label_indices, 0], axis=0)
                    µ_q2_t = np.vstack([µ_q2_t, data[µ_q2_indices, :]])
                    µ_q4_t = np.vstack([µ_q4_t, data[µ_q4_indices, :]])

                current_config = len(labels)
                if current_config > max_configs != -1:
                    break
            N_configs = µ_q2_t.shape[0]
            print(size, MCS_avg, N_configs)

            return size, MCS_avg, N_configs, labels, T, [µ_q2_t], [µ_q4_t]

        case 'old':
            if add == 0:
                fdir = f'Data/{adjacency}_{distribution},n={size},T={T0}_{Tf},MCS_avg={MCS_avg_0},max_MCS={max_MCS}'
            else:
                fdir = f'Data/{adjacency}_{distribution},n={size},T={T0}_{Tf},MCS_avg={MCS_avg_0},max_MCS={max_MCS},add={add}'
            try:
                T = np.loadtxt(f'{fdir}/T.dat')
            except:
                # print(f'No data for {adjacency},n={size},max_MCS={max_MCS},add={add},old.\n')
                return size, MCS_avg, [], [], [], [], [], [], [], []
            copies = len(T)

            file_type = f'{fdir}/MCS_avg={MCS_avg},seed=*.dat'
            files = glob.glob(file_type)

            labels = np.array([], dtype='int')
            µ_q2_t, µ_q4_t, σ2_q2_t, σ2_q4_t, µ_ql_t, µ_U_t, µ_U2_t = [np.empty([0, len(T)]) for _ in range(7)]

            for file in files:
                f = open(file)
                data = f.read().splitlines()
                f.close()

                file_labels = np.array(data[0::8])
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
            N_configs = µ_q2_t.shape[0]
            print(size, MCS_avg, N_configs)

            return size, MCS_avg, N_configs, labels, T, µ_q2_t, µ_q4_t, µ_ql_t, µ_U_t, µ_U2_t


# %% Read data of all the sizes and MCS values
def read_data(adjacency, distribution, sizes, add, T0, Tf, MCS_avg_0, max_MCS_vs_size, data_type='binned',
              max_configs=-1, n_q_dist=50, only_max_MCS=False):
    print(adjacency)
    max_MCS_vs_size_binned = max_MCS_vs_size[0]
    max_MCS_vs_size_old = max_MCS_vs_size[1]
    max_MCS_vs_size_fast = max_MCS_vs_size[2]


    if data_type == 'old' or data_type == 'all':
        n_MCS_old = [_ for _ in range(len(sizes))]
        cases_old = []
        for i, (size, max_MCS) in enumerate(zip(sizes, max_MCS_vs_size_old)):
            MCS_avgs = MCS_avg_0 * 2 ** (np.arange(np.log2(max_MCS // MCS_avg_0).astype('int') + 1))
            n_MCS_old[i] = len(MCS_avgs)
            for MCS_avg in MCS_avgs:
                if only_max_MCS and np.abs(MCS_avg-max_MCS)>10:
                    cases_old.append([size, MCS_avg, -1])
                else:
                    cases_old.append([size, MCS_avg, max_MCS])


    if data_type == 'binned' or data_type == 'all':
        n_MCS_binned = [_ for _ in range(len(sizes))]
        cases_binned = []
        for i, (size, max_MCS) in enumerate(zip(sizes, max_MCS_vs_size_binned)):
            MCS_avgs = MCS_avg_0 * 2 ** (np.arange(np.log2(max_MCS // MCS_avg_0).astype('int') + 1))
            n_MCS_binned[i] = len(MCS_avgs)

            for MCS_avg in MCS_avgs:
                if only_max_MCS and MCS_avg != max_MCS:
                    cases_binned.append([size, MCS_avg, -1])
                else:
                    cases_binned.append([size, MCS_avg, max_MCS])

    if data_type == 'fast' or data_type == 'all':
        n_MCS_fast = [_ for _ in range(len(sizes))]
        cases_fast = []
        for i, (size, max_MCS) in enumerate(zip(sizes, max_MCS_vs_size_fast)):
            MCS_avgs = [max_MCS]
            n_MCS_fast[i] = 1
            for MCS_avg in MCS_avgs:
                cases_fast.append([size, MCS_avg, max_MCS])


    match data_type:
        case 'binned':
            max_MCS_vs_size = max_MCS_vs_size_binned
        case 'old':
            max_MCS_vs_size = max_MCS_vs_size_old
        case 'fast':
            max_MCS_vs_size = max_MCS_vs_size_fast
        case 'all':
            max_MCS_vs_size = np.array( [max(MCS_old, MCS_binned, MCS_fast) for MCS_old, MCS_binned, MCS_fast
                                         in zip(max_MCS_vs_size_binned, max_MCS_vs_size_old, max_MCS_vs_size_fast)])

    n_MCS_vs_size = np.log2(max_MCS_vs_size // MCS_avg_0).astype('int') + 1
    MCS_avg_vs_size = [MCS_avg_0 * 2 ** (np.arange(n_MCS)) for n_MCS in n_MCS_vs_size]

    labels_vs_size, T_vs_size, q2_vs_size, q4_vs_size, ql_vs_size, U_vs_size, U2_vs_size, σ2_q2_bin_vs_size, \
    σ2_q4_bin_vs_size, q_dist_vs_size, N_configs_vs_size, copies_vs_size = \
        [[[[] for _ in range(n_MCS_vs_size[i])] for i in range(len(sizes))] for _ in range(12)]


    if data_type == 'old' or data_type == 'all':
        print('Old data')
        Data = Parallel(n_jobs=min(cpu_count(), len(cases_old)))(delayed(read_data_specific_size_and_MCS)
                                                                 (adjacency, distribution, size, add, T0, Tf,
                                                                     MCS_avg, MCS_avg_0, max_MCS, data_type='old',
                                                                     max_configs=max_configs, n_q_dist=n_q_dist)
                                                                 for [size, MCS_avg, max_MCS] in cases_old)
        for i, size in enumerate(sizes):
            size_indices = np.where(size == np.array([case[0] for case in Data]))[0]
            for k in range(n_MCS_old[i]):
                labels_vs_size[i][k], T_vs_size[i][k], q2_vs_size[i][k], \
                q4_vs_size[i][k], ql_vs_size[i][k], U_vs_size[i][k], U2_vs_size[i][k] = \
                    [Data[size_indices[k]][l] for l in range(3, 10)]

    if data_type == 'binned' or data_type == 'all':
        print('Binned data')
        Data = Parallel(n_jobs=min(cpu_count(), len(cases_binned)))(delayed(read_data_specific_size_and_MCS)
                                                                    (adjacency, distribution, size, add, T0, Tf,
                                                                     MCS_avg, MCS_avg_0, max_MCS, data_type='binned',
                                                                     max_configs=max_configs, n_q_dist=n_q_dist)
                                                                    for [size, MCS_avg, max_MCS] in cases_binned)

        for i, size in enumerate(sizes):
            size_indices = np.where(size == np.array([case[0] for case in Data]))[0]
            for k in range(n_MCS_binned[i]):
                try:
                    σ2_q2_bin_vs_size[i][k], σ2_q4_bin_vs_size[i][k], q_dist_vs_size[i][k] = \
                        [Data[size_indices[k]][l] for l in range(10, 13)]

                    if data_type == 'binned' or (data_type == 'all' and k >= n_MCS_old[i]):
                        labels_vs_size[i][k], T_vs_size[i][k], q2_vs_size[i][k], \
                            q4_vs_size[i][k], ql_vs_size[i][k], U_vs_size[i][k], U2_vs_size[i][k]\
                            = [Data[size_indices[k]][l] for l in range(3, 10)]

                    elif data_type == 'all':
                        if np.all(np.isclose(T_vs_size[i][k], Data[size_indices[k]][4])):
                            labels_vs_size[i][k] = np.concatenate([labels_vs_size[i][k], Data[size_indices[k]][3]])
                            q2_vs_size[i][k] = np.concatenate([q2_vs_size[i][k], Data[size_indices[k]][5]])
                            q4_vs_size[i][k] = np.concatenate([q4_vs_size[i][k], Data[size_indices[k]][6]])
                            ql_vs_size[i][k] = np.concatenate([ql_vs_size[i][k], Data[size_indices[k]][7]])
                            U_vs_size[i][k] = np.concatenate([U_vs_size[i][k], Data[size_indices[k]][8]])
                            U2_vs_size[i][k] = np.concatenate([U2_vs_size[i][k], Data[size_indices[k]][9]])
                        else:
                            print('Temperatures are not the same')
                            break
                except:
                    pass

    if data_type == 'fast' or data_type == 'all':
        print('Fast data')
        Data = Parallel(n_jobs=min(cpu_count(), len(cases_fast)))(delayed(read_data_specific_size_and_MCS)
                                                                  (adjacency, distribution, size, add, T0, Tf,
                                                                   MCS_avg, MCS_avg_0, max_MCS, data_type='fast',
                                                                   max_configs=max_configs, n_q_dist=n_q_dist)
                                                                  for [size, MCS_avg, max_MCS] in cases_fast)
        for i, size in enumerate(sizes):
            size_indices = np.where(size == np.array([case[0] for case in Data]))[0]
            k = np.where(max_MCS_vs_size_fast[i] == MCS_avg_vs_size[i])[0][0]
            try:
                try:
                    if np.all(np.isclose(T_vs_size[i][k], Data[size_indices[0]][4])):
                        labels_vs_size[i][k] = np.concatenate([labels_vs_size[i][k], Data[size_indices[0]][3]])
                        q2_vs_size[i][k] = np.concatenate([q2_vs_size[i][k], Data[size_indices[0]][5][0]])
                        q4_vs_size[i][k] = np.concatenate([q4_vs_size[i][k], Data[size_indices[0]][6][0]])
                except:
                    labels_vs_size[i][k] = Data[size_indices[0]][3]
                    T_vs_size[i][k] = Data[size_indices[0]][4]
                    q2_vs_size[i][k] = Data[size_indices[0]][5][0]
                    q4_vs_size[i][k] = Data[size_indices[0]][6][0]
            except:
                pass

    del Data

    for size_index, q2_vs_MCS in enumerate(q2_vs_size):
        for MCS_index, q2 in enumerate(q2_vs_MCS):
            try:
                N_configs_vs_size[size_index][MCS_index] = q2.shape[0]
                copies_vs_size[size_index][MCS_index] = q2.shape[1]
            except:
                N_configs_vs_size[size_index][MCS_index] = 0
                copies_vs_size[size_index][MCS_index] = 0

    return MCS_avg_vs_size, N_configs_vs_size, copies_vs_size, labels_vs_size, T_vs_size, q2_vs_size, q4_vs_size, \
           ql_vs_size, U_vs_size, U2_vs_size, σ2_q2_bin_vs_size, σ2_q4_bin_vs_size, q_dist_vs_size


# %% Extrapolate B at MCS -> inf to estimate error from non thermalization
def f_conv_vs_MCS(x, a, b, c):
    return a - b * np.exp(-x ** c)


def extrapolate_convergence(B_vs_size, error_vs_size, MCS_avg_0, max_MCSs, skip_initial_MCS_0=2):
    B_extrapolated_vs_size = []
    error_extrapolated_vs_size = []

    for B_vs_MCS, max_MCS, error in zip(B_vs_size, max_MCSs, error_vs_size):

        if len(B_vs_MCS) < skip_initial_MCS_0 + 2:
            skip_initial_MCS = 0
        else:
            skip_initial_MCS = skip_initial_MCS_0

        B_vs_MCS = np.array(B_vs_MCS)

        r = np.log2(max_MCS / MCS_avg_0).astype('int')
        MCS_eq = [max_MCS / 2 ** k for k in reversed(range(r + 1))]

        B_extrapolated = B_vs_MCS[-1]
        error_extrapolated = error[-1]

        for T_index in range(len(B_vs_MCS[0])):
            # print(T_index)

            try:
                params = curve_fit(f_conv_vs_MCS, MCS_eq[skip_initial_MCS:], B_vs_MCS[skip_initial_MCS:, T_index],
                                   p0=[1, 1, 0.000001], bounds=([0, -np.inf, -np.inf], [1, np.inf, np.inf]))[0]
            except:
                B_extrapolated_vs_size.append(B_extrapolated)
                error_extrapolated_vs_size.append(error_extrapolated)
                break

            if params[-1] < 0.05:
                B_extrapolated_vs_size.append(B_extrapolated)
                error_extrapolated_vs_size.append(error_extrapolated)
                break
            else:
                if error[-1][T_index] < np.abs(params[0] - B_extrapolated[T_index]):
                    error_extrapolated[T_index] = np.abs(params[0] - B_extrapolated[T_index])
                else:
                    error_extrapolated[T_index] = error[-1][T_index]
                B_extrapolated[T_index] = params[0]
            # print(params)

    return B_extrapolated_vs_size, error_extrapolated_vs_size


def skewness_of_histogram(q, dist):
    n = q.sum()
    q3 = (q * dist ** 3).sum() / n
    q2 = (q * dist ** 2).sum() / n
    return q3 / (q2 ** (3 / 2))
# %% Calculate autocorrelation time
def autocorrelation_time_q2(σ2_q2_bin_vs_size):
    tau_q2_T_vs_size = np.zeros([len(σ2_q2_bin_vs_size), σ2_q2_bin_vs_size[0][0].shape[-1]])

    for size_index in range(len(σ2_q2_bin_vs_size)):
        σ2_q2_bin_t = σ2_q2_bin_vs_size[size_index][-1]

        for T_index in range(σ2_q2_bin_t.shape[-1]):
            σ2_q2_bin_c = σ2_q2_bin_t.mean(0)[:, T_index]
            bins = np.arange(σ2_q2_bin_c.shape[0])

            # Remove the last three points
            bins = bins[:-2]
            σ2_q2_bin_c = σ2_q2_bin_c[:-2]

            M = 2 ** bins
            tau = M * σ2_q2_bin_c[bins] / σ2_q2_bin_c[0]

            tau_max_index = np.where(tau == tau.max())[0][0]
            if np.abs(tau[tau_max_index] - tau[tau_max_index - 1]) / tau[tau_max_index] < 0.15:
                tau_q2_T_vs_size[size_index, T_index] = tau.max()
            else:
                tau_q2_T_vs_size[size_index, T_index] = np.nan

        try:
            nan_index = np.where(np.isnan(tau_q2_T_vs_size[size_index, :]))[0][-1]
            tau_q2_T_vs_size[size_index, :nan_index + 1] = np.nan
        except:
            pass

    return tau_q2_T_vs_size