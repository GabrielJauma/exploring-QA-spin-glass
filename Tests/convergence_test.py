from sys import path
from os.path import exists

path.insert(0, '..')

import Modules.spin_glass as sg
import Modules.monte_carlo as mc

import numpy as np
from joblib import Parallel, delayed
import time

import matplotlib.pyplot as plt
from tqdm import tqdm

dir = 'Figures/'
if not exists(dir):
    dir = '/home/gabriel/Architecture_v1/Figures/'

# %%
test_N_initial = False
test_N_term = False
test_N_config = False
test_N_config_vs = True

N_initials = np.logspace(2, 5, 4).astype(int)
N_terms = np.logspace(2, 5, 4).astype(int)
N_configs = np.logspace(2, 3, 3).astype(int)

# %%
adjacency = '3D'
distribution = 'gaussian_EA'
size = 6 ** 3
T0 = 0.1
Tf = 3
seed = 12

N_initial0 = N_initials[-1]
N_term0 = N_terms[-1]
N_config0 = N_configs[-1]
steps_until_temp = 10

copies = 20
Ti = np.linspace(T0, Tf, copies)
# init_steps = int(1e6)
# avg_steps = int(1e6)
# rng = np.random.default_rng(12)
# J = sg.connectivity_matrix(size, adjacency, distribution, rng=rng)
# Ti, _ = mc.optimal_temperature_distribution(T0, Tf, J, rng, init_steps=init_steps, avg_steps=avg_steps,
#                                             accept_prob_min=0.3, accept_prob_max=0.6, plot=False)
copies = len(Ti)
T = np.zeros(copies * 2)
T[0::2], T[1::2] = Ti.copy(), Ti.copy()

# %% TEST N_initial
if test_N_initial:
    N_term = N_term0
    print('TEST N_initial')

    J = sg.connectivity_matrix(size, adjacency, distribution, rng=np.random.default_rng(seed))
    q2_q4 = Parallel(n_jobs=len(N_initials))(
        delayed(mc.equilibrate_and_average)
        (1 / T, J, N_initials[i], N_term, steps_until_temp=steps_until_temp, rng=np.random.default_rng(seed))
        for i in tqdm(range(len(N_initials))))

    q2 = np.array([q2_q4[i][0] for i in range(len(N_initials))])
    q4 = np.array([q2_q4[i][1] for i in range(len(N_initials))])
    B = 0.5 * (3 - q4 / q2 ** 2)
    E_B = np.abs(B[:-1] - B[-1])

    fig, ax = plt.subplots(dpi=500)
    ax.plot(N_initials[:-1], np.max(E_B, 1), 'r', label=r'max $\epsilon_B$')
    ax.plot(N_initials[:-1], np.trapz(E_B, T[0::2]), 'k', label=r'$\int \epsilon_B dT$')
    ax.set_xlabel(r'$N_{initials}$')
    ax.set_title(r'$\epsilon_B=|B(N_{term})-B|$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    plt.savefig(f'{dir}TEST N_initial {distribution} {adjacency} N={size} T=({T0},{Tf}) N_t=10^{np.log10(N_term).astype(int)}.pdf')

# %% TEST N_term
if test_N_term:
    N_initial = N_initial0
    print('TEST N_term')

    J = sg.connectivity_matrix(size, adjacency, distribution, rng=np.random.default_rng(seed))
    q2_q4 = Parallel(n_jobs=len(N_terms))(
        delayed(mc.equilibrate_and_average)
        (1 / T, J, N_initial, N_terms[i], steps_until_temp=steps_until_temp, rng=np.random.default_rng(seed))
        for i in tqdm(range(len(N_terms))))

    q2 = np.array([q2_q4[i][0] for i in range(len(N_terms))])
    q4 = np.array([q2_q4[i][1] for i in range(len(N_terms))])
    B = 0.5 * (3 - q4 / q2 ** 2)
    E_B = np.abs(B[:-1] - B[-1])

    fig, ax = plt.subplots(dpi=500)
    ax.plot(N_terms[:-1], np.max(E_B, 1), 'r', label=r'max $\epsilon_B$')
    ax.plot(N_terms[:-1], np.trapz(E_B, T[0::2]), 'k', label=r'$\int \epsilon_B dT$')
    ax.set_xlabel(r'$N_{term}$')
    ax.set_title(r'$\epsilon_B=|B(N_{initial})-B|$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    plt.savefig(f'{dir}TEST N_term  {distribution} {adjacency} N={size} T=({T0},{Tf}) N_i=10^{np.log10(N_initial).astype(int)}.pdf')

# %% TEST N_config

if test_N_config:
    N_initial = N_initial0
    N_term = N_term0
    print('TEST N_config')

    B_1 = np.zeros([len(N_configs), len(T)//2])
    B_2 = np.zeros([len(N_configs), len(T)//2])
    for c, N_config in enumerate(N_configs):
        seeds = np.random.SeedSequence().spawn(N_config)
        q2_q4_c = Parallel(n_jobs=-1)(
            delayed(mc.equilibrate_and_average)
            (1 / T, sg.connectivity_matrix(size, adjacency, distribution, rng=np.random.default_rng(seed=seeds[i])),
             N_initial, N_term, steps_until_temp=steps_until_temp, rng=np.random.default_rng(seed=seeds[i]))
            for i in tqdm(range(N_config)))
        q2 = np.array([q2_q4_c[i][0] for i in range(N_config)])
        q4 = np.array([q2_q4_c[i][1] for i in range(N_config)])

        B_1[c, :] = 0.5 * (3 - np.mean(q4, 0) / np.mean(q2, 0) ** 2)
        B_2[c, :] = 0.5 * (3 - np.mean(q4 / q2 ** 2, 0))

    E_B_1 = np.abs(B_1[:-1] - B_1[-1])
    E_B_2 = np.abs(B_2[:-1] - B_2[-1])

    fig, ax = plt.subplots(ncols=2, dpi=500)
    for i, E_B in enumerate([E_B_1,E_B_2]):
        ax[i].plot(N_configs[:-1], np.max(E_B, 1), 'r', label=r'max $\epsilon_B$')
        ax[i].plot(N_configs[:-1], np.trapz(E_B, T[0::2]), 'k', label=r'$\int \epsilon_B dT$')
        ax[i].set_xlabel(r'$N_{configs}$')
        ax[i].set_yscale('log')
        ax[i].set_xscale('log')
    ax[1].legend()
    ax[0].set_title(r'$B=[<>]/[<>]$')
    ax[1].set_title(r'$B=[<>/<>]$')
    fig.suptitle(r'$\epsilon_B=|B(N_{term})-B|$')
    plt.show()
    plt.savefig(f'{dir}TEST N_config {distribution} {adjacency} N={size} T=({T0},{Tf}) N_i=10^{np.log10(N_initial).astype(int)} N_t=10^{np.log10(N_term).astype(int)}.pdf')


# %% TEST N_config vs N_term and N_initial

if test_N_config_vs:
    print('TEST N_config vs N_term and N_initial')
    fig, ax = plt.subplots(ncols=len(N_initials), nrows=len(N_terms), dpi=500, figsize=[3*len(N_initials), 3*len(N_terms)])

    # Calculate reference binder cumulant with N_initial0, N_term0, and N_config0.
    seeds = np.random.SeedSequence().spawn(N_config0)
    q2_q4_c = Parallel(n_jobs=-1)(
        delayed(mc.equilibrate_and_average)
        (1 / T, sg.connectivity_matrix(size, adjacency, distribution, rng=np.random.default_rng(seed * i)),
         N_initial0, N_term0, steps_until_temp=steps_until_temp, rng=np.random.default_rng(seed*i))
        for i in tqdm(range(N_config0)))
    q2 = np.array([q2_q4_c[i][0] for i in range(N_config0)])
    q4 = np.array([q2_q4_c[i][1] for i in range(N_config0)])
    Br_1 = 0.5 * (3 - np.mean(q4, 0) / np.mean(q2, 0) ** 2)
    Br_2 = 0.5 * (3 - np.mean(q4 / q2 ** 2, 0))

    for n_i, N_initial in enumerate(N_initials):
        print(f'N_i={N_initial}')
        for n_t, N_term in enumerate(N_terms):
            print(f'N_t={N_term}')
            B_1 = np.zeros([len(N_configs), len(T)//2])
            B_2 = np.zeros([len(N_configs), len(T)//2])
            for c, N_config in enumerate(N_configs):

                if N_initial==N_initials[-1] and N_term==N_terms[-1] and N_config==N_configs[-1]:
                    B_1[c, :] = B_1[c-1, :]
                    B_2[c, :] = B_2[c-1, :]
                    continue

                q2_q4_c = Parallel(n_jobs=-1)(
                    delayed(mc.equilibrate_and_average)
                    (1 / T, sg.connectivity_matrix(size, adjacency, distribution, rng=np.random.default_rng(seed * i)),
                     N_initial, N_term, steps_until_temp=steps_until_temp, rng=np.random.default_rng(seed*i))
                    for i in tqdm(range(N_config)))
                q2 = np.array([q2_q4_c[i][0] for i in range(N_config)])
                q4 = np.array([q2_q4_c[i][1] for i in range(N_config)])

                B_1[c, :] = 0.5 * (3 - np.mean(q4, 0) / np.mean(q2, 0) ** 2)
                B_2[c, :] = 0.5 * (3 - np.mean(q4 / q2 ** 2, 0))

            E_B_1 = np.abs(B_1 - Br_1)
            E_B_2 = np.abs(B_2 - Br_2)

            ax[n_i, n_t].plot(N_configs, np.max(E_B_1, 1), 'r', label=r'max $\epsilon_B$')
            ax[n_i, n_t].plot(N_configs, np.max(E_B_2, 1), 'r:', label=r'max $\epsilon_B$')
            ax[n_i, n_t].plot(N_configs, np.trapz(E_B_1, T[0::2]), 'k', label=r'$\int \epsilon_B dT$')
            ax[n_i, n_t].plot(N_configs, np.trapz(E_B_2, T[0::2]), 'k:', label=r'$\int \epsilon_B dT$')
            ax[n_i, n_t].set_yscale('log')
            ax[n_i, n_t].set_xscale('log')


    ax[0,0].legend()
    ax[len(N_initials)//2, -1].set_xlabel(r'$N_{configs}$')
    fig.suptitle(r'$\epsilon_B=|B(N_{initial}, N_{term}, N_{config})-B|$    '
                 f'N_i left to right 10^({np.log10(N_initials).astype(int)}    '
                 f'N_t up to down 10^({np.log10(N_terms).astype(int)}')
    plt.show()
    plt.tight_layout()
    plt.savefig(f'{dir}TEST N_config {distribution} {adjacency} N={size} T=({T0},{Tf}) '
                f'N_i=10^({np.log10(N_initials[0]).astype(int)},{np.log10(N_initials[-1]).astype(int)}) '
                f'N_t=10^({np.log10(N_terms[0]).astype(int)},{np.log10(N_terms[-1]).astype(int)}) '
                f'N_c=10^({np.log10(N_configs[0]).astype(int)},{np.log10(N_configs[-1]).astype(int)}).pdf')
