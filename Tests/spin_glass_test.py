import sys
import time
from importlib import reload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from Modules import spin_glass as sg, chip_architecture as ca, monte_carlo as mc


sys.path.append('../')
ca = reload(ca)
sg = reload(sg)
mc = reload(mc)
plt.rcParams.update({
    "text.usetex": False})


# %% Parameters
size = 1600
adjacency = 'random_regular_9'
distribution = 'gaussian_EA'
add = 0
periodic = True

T0 = 0.2
Tf = 1.5
copies = 30

MCS_avg = 2 ** 12
max_MCS = 2 ** 12

T = np.linspace(T0, Tf, copies)
T_replicas = np.zeros(copies * 2)
T_replicas[0::2] = T.copy()
T_replicas[1::2] = T.copy()

# %%
rng = np.random.default_rng(3412431)
J = sg.connectivity_matrix(size, adjacency, distribution, rng=rng, add=add, periodic=periodic)

# %% Single disorder config, no binned data
t = time.perf_counter()
µ_q2_vs_MCS, µ_q4_vs_MCS, σ2_q2_vs_MCS, σ2_q4_vs_MCS, µ_ql_vs_MCS, µ_U_vs_MCS, µ_U2_vs_MCS, MCS_avg_s = \
    mc.equilibrate_and_average(1/T_replicas, J, MCS_avg, max_MCS, rng=rng)
print(time.perf_counter()-t)

# %% Single disorder config, binned data
t = time.perf_counter()
µ_q2_vs_MCS, µ_q4_vs_MCS, σ2_q2_bin_vs_MCS, σ2_q4_bin_vs_MCS, µ_ql_vs_MCS, µ_U_vs_MCS, µ_U2_vs_MCS, \
q_dist_vs_MCS, MCS_avg_s = mc.equilibrate_and_average_bin(1/T_replicas, J, MCS_avg, max_MCS, rng=rng)
print(time.perf_counter()-t)

σ2_q2_vs_MCS = [σ2_q2_bin[0] for σ2_q2_bin in σ2_q2_bin_vs_MCS]
σ2_q4_vs_MCS = [σ2_q4_bin[0] for σ2_q4_bin in σ2_q4_bin_vs_MCS]

# %% Single disorder config, fast
t = time.perf_counter()
µ_q2, µ_q4 = mc.equilibrate_and_average_fast(1/T_replicas, J, max_MCS, rng=rng)
µ_q2_vs_MCS = [µ_q2]
µ_q4_vs_MCS = [µ_q4]
print(time.perf_counter()-t)
# %% Test save data
for i, MCS_avg_i in enumerate(MCS_avg_s):
    df = pd.DataFrame(np.concatenate((12313*np.ones([1,len(µ_q2_vs_MCS[i])]), np.array([µ_q2_vs_MCS[i]]), np.array([µ_q2_vs_MCS[i]]),
                           σ2_q2_bin_vs_MCS[i], σ2_q4_bin_vs_MCS[i],
                           np.array([µ_ql_vs_MCS[i]]), np.array([µ_U_vs_MCS[i]]), np.array([µ_U2_vs_MCS[i]]),
                           q_dist_vs_MCS[i])))
    df.to_csv('please_work_15.csv', mode='a', index=False)

# %%
df = np.array( pd.read_csv('please_work_15.csv') )


# %% Plot pdf of q for multiple temperatures
T_index = np.linspace(0,copies-1,5, dtype='int')
MCS_index = -1

n_distributon = q_dist_vs_MCS[0].shape[0]
dist = np.linspace(-1, 1, n_distributon, endpoint=False)
dist += dist[1]-dist[0]

q_dist = q_dist_vs_MCS[MCS_index] / q_dist_vs_MCS[MCS_index].max(0)

fig, ax = plt.subplots(dpi=200)
ax.plot(dist, q_dist[:,T_index],'.-')
fig.show()

fig, ax = plt.subplots(dpi=200)
ax.plot(dist[n_distributon//2:n_distributon], np.abs( q_dist[n_distributon//2:n_distributon,T_index][::-1] - q_dist[:n_distributon//2+1,T_index] )  ,'.-')
fig.show()

# %% Plot B for a single config vs MCS
fig, ax = plt.subplots(dpi=500)
[ax.plot(T, 0.5 * (3 - µ_q4_vs_MCS[i] / µ_q2_vs_MCS[i] ** 2), '.-', label=f'{MCS_avg_s[i]}', linewidth=1) for i in range(len(µ_U_vs_MCS))]
# [ax.plot(T, µ_q2_vs_MCS[i], '.-', label=f'{MCS_avg_s[i]}', linewidth=1) for i in range(len(µ_U_vs_MCS))]
# [ax.plot(T, µ_U2_vs_MCS[i], '.-', label=f'{MCS_avg_s[i]}', linewidth=1) for i in range(len(µ_U_vs_MCS))]
# [ax.plot(T, µ_U_vs_MCS[i], '.-', label=f'{MCS_avg_s[i]}', linewidth=1) for i in range(len(µ_U_vs_MCS))]
# [ax.plot(T, 1-T*np.abs(µ_U_vs_MCS[i])/2.5 - µ_ql_vs_MCS[i], '.-', label=f'{MCS_avg_s[i]}', linewidth=1) for i in range(len(µ_U_vs_MCS))]
# [ax.plot(T, (µ_U2_vs_MCS[i]-µ_U_vs_MCS[i]**2)/T**3, '.-', label=f'{MCS_avg_s[i]}', linewidth=1) for i in range(len(µ_U_vs_MCS))]
ax.legend()
# ax.set_xlim([T0,Tf])
# ax.set_ylim([0,1])

ax.set_title(f'n={size}')
fig.show()

# %%
T_index = 20

fig1, ax1 = plt.subplots(dpi=200)
fig2, ax2 = plt.subplots(dpi=200)
for MCS_index in range(len(MCS_avg_s)):
    sigma_q2 = σ2_q2_bin_vs_MCS[MCS_index][:,T_index]
    M0 = MCS_avg_s[MCS_index]
    n_bins = sigma_q2.shape[0]
    M_l = [M0/2**bin for bin in range(n_bins)]
    error = np.array( [np.sqrt((1/M)*sigma) for M, sigma in zip(M_l, sigma_q2)] )

    tau = 0.5*((error.max()/error[0])**2-1)
    print(tau)

    ax1.plot(error[:-3], label = M0)

    n_bin = np.arange(sigma_q2.shape[0])
    ax2.plot(2 ** n_bin, 2 ** n_bin * sigma_q2[n_bin] / sigma_q2[0])

ax1.legend()
# ax1.set_yscale('log')
ax2.set_xscale('log')
fig1.show()
fig2.show()

# %%
fig, ax = plt.subplots(dpi=200)
ax.set_xscale('log')
fig.show()


# %% Calculate and plot CV, use it to estimate optimal T_replicas distribution

# µ_U_vs_MCS = U_vs_size[0]
# µ_U2_vs_MCS = U2_vs_size[0]
# MCS_avg_s = MCS_avg_vs_size[0]
# µ_U_vs_MCS, µ_U2_vs_MCS = [np.array(µ_U_vs_MCS), np.array(µ_U2_vs_MCS)]

µ_U_vs_MCS, µ_U2_vs_MCS = [np.array(µ_U_vs_MCS), np.array(µ_U2_vs_MCS)]
Cv_s = (µ_U2_vs_MCS - µ_U_vs_MCS ** 2) / T ** 2

fig, ax = plt.subplots(dpi=500)
# [plt.plot(T, µ_U_vs_MCS[i], label=f'{MCS_avg_s[i]}') for i in range(len(µ_U_vs_MCS))]
[plt.plot(T, np.gradient(µ_U_vs_MCS[i], T)/T, '--', label=f'dE/dT {MCS_avg_s[i]}') for i in range(len(µ_U_vs_MCS))]
[plt.plot(T, Cv_s[i]/np.max(Cv_s[i]), label=f'<E^2>-<E>^2 {MCS_avg_s[i]}') for i in range(len(µ_U_vs_MCS))]
# [ax.plot(T, Cv_s[i] / T, '.-', label=f'Cv, MCS_avg={MCS_avg_s[i]}') for i in range( len(µ_U_vs_MCS))]
# plt.ylim([0,1])
ax.set_xlim([0, Tf])
ax.set_title(f'n={size}')
ax.legend()
fig.show()

# %% multiple disorder configs
threads = 8
N_configs = threads * 4
config_seeds = np.random.SeedSequence(5231).spawn(N_configs)
rngs = [np.random.default_rng(config_seeds[i]) for i in range(N_configs)]

data_multiple_configs = Parallel(n_jobs=threads)(
    delayed(mc.equilibrate_and_average)(
        1/T_replicas, sg.connectivity_matrix(size, adjacency, distribution, rng=rngs[i], add=add),
        MCS_avg, max_MCS, rng=rngs[i])
    for i in tqdm(range(N_configs)))

# %%
U_c = np.zeros([N_configs, copies])
U2_c = np.zeros([N_configs, copies])
for i in range(N_configs):
    U_c[i] = data_multiple_configs[i][5][-1]
    U2_c[i] = data_multiple_configs[i][6][-1]

Cv_c_by_T = (U2_c - U_c ** 2) / T ** 3
CvT = np.mean(Cv_c_by_T, 0)

# %%
fig, ax = plt.subplots(dpi=500)
ax.plot(T, Cv_c_by_T.T)
ax.plot(T, CvT, 'k')
ax.set_title(r'$C_v/T$')
fig.show()

# %%
fig, ax = plt.subplots(dpi=500)
ax.plot(T, CvT / CvT.max(), '.-')
ax.set_xlim([T0, Tf])
ax.set_ylim([0, 1])
ax.set_title(f'n={size}')
fig.show()

# %%
copies = 30
# copies = 2000
dS = np.trapz(CvT, T) / copies
error = dS / 100
dT = (Tf - T0) / (copies - 1)
T_n = np.zeros(copies)
T_n[0] = T0

for c in range(copies - 1):
    T_0 = T_n[c]
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
    T_n[c + 1] = T_1
    dT = T_1 - T_0

# %%
fig, ax = plt.subplots(dpi=500)
T_n_hist = np.histogram(T_n, 50, density=True)
ax.hist(T_n, 50, histtype='step', density=True, stacked=False)
ax.plot(T, T_n_hist[0].max() * CvT / CvT.max(), 'k')
fig.show()

# %%
fig, ax = plt.subplots(dpi=500)
ax.plot(T_n, 2 * np.ones_like(T_n), '.', label='Constant entropy')
ax.plot(T, np.ones_like(T), '.', label='Linear')
ax.legend()
fig.show()

# %%
fig, ax = plt.subplots(dpi=500)
ax.plot(np.linspace(0, 1, len(T_n)), np.linspace(0.5, 2.0, len(T_n)), marker='.', label='Linear')
ax.plot(np.linspace(0, 1, len(T_n)), T_n, marker='.', label='Constant entropy')
ax.set_title(f'n={size}')
ax.set_ylim([0.3,2.1])
ax.legend()
fig.show()

# %%
T_n = np.linspace(0.5,3,30)
size=3200
current_dir = sys.path[1]
dir_T_dist = current_dir + f'/Cluster/temperature_distributions/{adjacency}_{distribution},n={size},T={T0}_{Tf}.dat'
file_T_dist = open(dir_T_dist, 'w')
np.savetxt(file_T_dist, T_n)
file_T_dist.close()

# %% Plot B vs T and µ_q2, µ_q4, σ2_q2, σ2_q4 vs T
µ_q2 = µ_q2_vs_MCS[-1]
µ_q4 = µ_q4_vs_MCS[-1]
σ2_q2 = σ2_q2_vs_MCS[-1]
σ2_q4 = σ2_q4_vs_MCS[-1]
MCS_avg = MCS_avg_s[-1]

B = 0.5 * (3 - µ_q4 / µ_q2 ** 2)
dBdq4 = -1 / µ_q2 ** 2
dBdq2 = 2 * µ_q4 / µ_q2 ** 3
B_error = np.sqrt(σ2_q2 * dBdq2 ** 2 + σ2_q4 * dBdq4 ** 2) / np.sqrt(MCS_avg)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
ax1.plot(T, B, label='$B$')
# ax1.errorbar(T, B, yerr=B_error, label='$B$')
ax1.set_ylim(0, 1.01)
ax1.set_xlim(T[0], T[-1])
ax1.legend()
fig.show()
#%%
ax2.plot(T, µ_q2, '.-', label='$\mu_q^2$')
ax2.plot(T, µ_q4, '.-', label='$\mu_q^4$')
ax2.plot(T, σ2_q2, '.-', label='$\sigma^{2}_{q_2}$')
ax2.plot(T, σ2_q4, '.-', label='$\sigma^{2}_{q_4}$')
ax2.set_ylim(0, 1.01)
ax2.set_xlim(T[0], T[-1])
ax2.legend()

fig.suptitle(f'MCS-AV={MCS_avg}')
fig.tight_layout()
fig.show()

