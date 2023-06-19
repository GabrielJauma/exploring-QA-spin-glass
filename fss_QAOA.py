import Modules.spin_glass as sg
import Modules.monte_carlo as mc
import Modules.exact_results_spin_glass as exact

import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from numba import njit

from importlib import reload

sg = reload(sg)
mc = reload(mc)
# %% Parameters
adjacency = 'SK'
adjacency = 'random_regular_7'
distribution = 'gaussian_EA'
add = 0
sizes = np.array([6, 10, 14, 18, 22, 26])
sizes = np.array([20, 40, 80, 100])
# sizes = np.array([20, 40, 80, 100])
threads = 24
N_configs = threads * 1
runs_per_config = threads
copies = 100
MCS_min = 100
MCS_0 = 10
max_MCS = 500 #MCS_0 * 100 #4000 #MCS_0 * 2 ** 5
tempering = True

T0 = 1 / 100
Tf = 4
Ti = np.concatenate([np.geomspace(T0, 1, copies // 2, endpoint=False), np.linspace(1, Tf, copies // 2)])
T = np.zeros(copies * 2)
T[0::2] = Ti
T[1::2] = Ti

load_instances = False
load_data = False
use_exact_ground_state = False
check_goodness_of_gs = False
save_data = False

pre_seed = 45431155
if pre_seed is None:
    pre_seed = int((time.time() % 1) * 1e8)
seeds = np.random.SeedSequence(pre_seed).spawn(N_configs)

size_index = 0
config_index = 0

# %% Generate or load instances (matrices J and exact mins, ...)
if load_instances == False:
    J_vs_size_configs, E_min_vs_size_configs, rng_vs_size_configs = \
        [[[[] for _ in range(N_configs)] for _ in sizes] for _ in range(3)]

    for s, size in enumerate(sizes):
        print('\n Size', size, 'of', sizes, '\n')
        for i in tqdm(range(N_configs)):
            # print('Config', i + 1, 'of', N_configs)
            rng = np.random.default_rng(seed=seeds[i])
            J = sg.connectivity_matrix(size, adjacency, distribution, rng, add=add,
                                       sparse=False)

            J_norm_2 = np.linalg.norm(J, ord=2)
            J = J / J_norm_2
            J_sp = sg.custom_sparse(J)

            t = time.perf_counter()
            if use_exact_ground_state:
                E_min = exact.numba_min_energy_parallel(J)
            else:
                E_min = mc.estimate_ground_state(1 / T, J_sp, MCS_min, rng, tempering)
                if check_goodness_of_gs:
                    print(f'The approximate GS is good? {np.isclose(E_min, exact.numba_min_energy_parallel(J))}')
            print(time.perf_counter() - t)
            J_vs_size_configs[s][i] = J_sp
            E_min_vs_size_configs[s][i] = E_min
            rng_vs_size_configs[s][i] = rng

    if save_data:
        np.savez(f'Data/QAOA/Instances, sizes = {sizes}, T in [{T0},{Tf}], copies = {copies}, N_configs = {N_configs}, '
                 f'MCS = {max_MCS}',J_vs_size_configs=J_vs_size_configs,E_min_vs_size_configs=E_min_vs_size_configs,
                 rng_vs_size_configs=rng_vs_size_configs)
else:
    instances = np.load(f'Data/QAOA/Instances, sizes = {sizes}, T in [{T0},{Tf}], copies = {copies}, N_configs = {N_configs}.npz', allow_pickle=True)
    J_vs_size_configs = instances['J_vs_size_configs']
    E_min_vs_size_configs = instances['E_min_vs_size_configs']
    rng_vs_size_configs = instances['rng_vs_size_configs']

# %% SIMULATIONS SAVING E vs T
t = time.perf_counter()
E_vs_size_configs = [[[0] for _ in range(N_configs)] for _ in sizes]
for s, size in enumerate(sizes):
    print('Size', size, 'of', sizes)
    E_vs_size_configs[s] = np.array(
        Parallel(n_jobs=threads)
        (delayed(mc.mc_evolution_tests)
         (1 / T, J, steps=max_MCS, start=None, rng=rng, tempering=tempering,
          trajectories=True, tempering_probabilities=False, only_E=True)
         for J, rng in zip(J_vs_size_configs[s], rng_vs_size_configs[s])))

print('Elapsed time', time.perf_counter() - t)

# %% Calculate P_reach_GS_vs_size_configs and P_sample_GS_vs_size_configs
size_index = 0
config_index = 0
P_sample_GS_vs_size_configs = [[] for _ in sizes]
P_sample_GS_vs_size_configs_no_T = [[] for _ in sizes]

for s, (size, E_vs_config, E_min_vs_config) in tqdm(enumerate(zip(sizes, E_vs_size_configs, E_min_vs_size_configs))):
    P_sample_GS_vs_configs = np.zeros([N_configs,max_MCS*size,copies*2])
    P_sample_GS_vs_configs_no_T = np.zeros([N_configs,max_MCS*size])
    for c, (E_vs_t, E_min) in enumerate(zip(E_vs_config, E_min_vs_config)):
        is_min = np.isclose(E_vs_t, E_min)
        is_min_no_T = np.any(np.isclose(E_vs_t, E_min),1)
        steps = np.array([np.arange(1, len(is_min) + 1) for _ in range(2 * copies)]).T + 1
        steps_no_T = np.arange(1,len(is_min_no_T)+1)
        P_sample_GS_vs_configs[c] = np.cumsum(is_min, 0) / steps
        P_sample_GS_vs_configs_no_T[c] = np.cumsum(is_min_no_T) / steps_no_T
    P_sample_GS_vs_size_configs[s] = P_sample_GS_vs_configs
    P_sample_GS_vs_size_configs_no_T[s] = P_sample_GS_vs_configs_no_T

P_sample_GS = P_sample_GS_vs_size_configs[size_index][config_index]
E_vs_t = E_vs_size_configs[size_index][config_index]
E_min = E_min_vs_size_configs[size_index][config_index]
J = J_vs_size_configs[size_index][config_index]


# %% Calculate P to have reached GS vs n MC steps

P_to_have_reached_GS_vs_size_vs_MC_step = [[] for _ in sizes]
for s, (size, E_vs_config, E_min_vs_config) in tqdm(enumerate(zip(sizes, E_vs_size_configs, E_min_vs_size_configs))):
    MC_step_to_min_vs_configs = np.zeros([N_configs, copies * 2])
    for c, (E_vs_t, E_min) in enumerate(zip(E_vs_config, E_min_vs_config)):
        MC_step_to_min_vs_configs[c] = exact.steps_to_min(E_vs_t, E_min)

    P_to_have_reached_GS_vs_size_vs_MC_step[s] = exact.p_to_GS_vs_steps(MC_step_to_min_vs_configs, E_vs_t.shape[0])

# %% Plot probability of reaching the minimum energy vs beta for a specific size and config.
is_min = np.isclose(E_vs_t, E_min)
reach_min = np.logical_and(~is_min[:-1], is_min[1:])
steps = np.array([np.arange(1, len(reach_min) + 1) for _ in range(2 * copies)]).T
probability_reach = np.cumsum(reach_min, 0) / steps

fig, ax = plt.subplots(dpi=150)
ax.plot(Ti, probability_reach[-1, ::2])
ax.plot(Ti, probability_reach[-1, 1::2])
ax.plot(Ti, np.ones(copies) * 2 / 2 ** (sizes[size_index]))

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel('P reach GS')
ax.set_xlabel('T')
fig.suptitle(f'n={size}')
fig.show()

# %% Plot convergence of previous plot
y = probability_reach[:, 0::16]
colors = plt.cm.jet(np.linspace(0, 1, y.shape[1]))

fig, ax = plt.subplots(dpi=300)
for i in range(y.shape[1]):
    ax.plot(y[:, i], color=colors[i], linewidth=0.5)

ax.set_ylim([0, y[-1].max() * 1.1])
ax.set_xlim([1000, len(reach_min)])
ax.set_xscale('log')
fig.show()

# %% Compare P_sample_GS with boltzmann distribution for a single config
Z_T = exact.partition_function(T, sizes[size_index], J)
fig, ax = plt.subplots(dpi=150)
ax.plot(Ti, P_sample_GS[-1,::2], '.')
ax.plot(Ti, 2*np.exp(-E_min/Ti)/Z_T[::2])
ax.set_yscale('log')
ax.set_xlabel('T')
ax.set_ylabel('$P_{ground}$')
# ax.set_xlim([1,Tf])
# ax.set_ylim([1,1e-3])
fig.suptitle(f'P vs T and botzmann n={sizes[size_index]}')
fig.show()


# %% SIMULATIONS SAVING STEPS TO GS
if load_data == False:
    t = time.perf_counter()
    steps_to_GS_vs_size_configs_runs = [[[] for _ in range(N_configs)] for _ in sizes]
    for s, size in enumerate(sizes):
        print('Size', size, 'of', sizes)
        for c, (J, rng, E_min) in enumerate(zip(J_vs_size_configs[s], rng_vs_size_configs[s], E_min_vs_size_configs[s])):
            seeds = np.random.SeedSequence(rng.integers(0, runs_per_config*10000)).spawn(runs_per_config)

            steps_to_GS_vs_size_configs_runs[s][c] = np.array(
                Parallel(n_jobs=threads)
                (delayed(mc.steps_to_ground_state)
                 (1 / T, J, max_MCS, np.random.default_rng(seed=seeds[i]), E_min, tempering) for i in range(runs_per_config)))

    print('Elapsed time', time.perf_counter() - t)

    steps_to_GS_vs_size_configs = [ [] for _ in sizes]
    for s, size in enumerate(sizes):
        for c in range(N_configs):
            steps_to_GS_vs_size_configs[s].append(steps_to_GS_vs_size_configs_runs[s][c])
        steps_to_GS_vs_size_configs[s] = np.array(steps_to_GS_vs_size_configs[s]).reshape((runs_per_config*N_configs,copies*2))

    size_index = 0
    config_index = 0
    size = sizes[size_index]
    steps_to_GS = steps_to_GS_vs_size_configs[size_index][config_index]

    if save_data:
        np.savez(f'Data/QAOA/steps_to_GS, sizes = {sizes}, T in [{T0},{Tf}], copies = {copies}, N_configs = {N_configs}, '
                 f'MCS = {max_MCS}', steps_to_GS_vs_size_configs=steps_to_GS_vs_size_configs)
else:
    data = np.load(f'Data/QAOA/steps_to_GS, sizes = {sizes}, T in [{T0},{Tf}], copies = {copies}, N_configs = {N_configs}, '
                 f'MCS = {max_MCS}.npz')
    steps_to_GS_vs_size_configs = data['steps_to_GS_vs_size_configs']

# %% Calculate P to have reached GS
P_to_have_reached_GS_vs_size_vs_MC_step = [[] for _ in sizes]
for s, (size, steps_to_GS_vs_configs) in tqdm(enumerate(zip(sizes, steps_to_GS_vs_size_configs))):
    steps_to_GS_vs_configs[ steps_to_GS_vs_configs == 0 ] = max_MCS*size*10
    P_to_have_reached_GS_vs_size_vs_MC_step[s] = exact.p_to_GS_vs_steps(steps_to_GS_vs_configs, max_MCS*size)

# %% PT vs QAOA
N_PT_vs_QAOA_vs_T = []
fig, ax = plt.subplots(dpi=150)
for T_index in [0, copies, -1]:
    N_PT_vs_QAOA = np.zeros(len(sizes))
    for s in range(len(sizes)):
        N_PT_vs_QAOA[s] = np.where(
            (1 - P_to_have_reached_GS_vs_size_vs_MC_step[s][T_index] > (1 - 2 ** (-sizes[s] / 2)) ** np.arange(max_MCS * sizes[s]))[
            1:])[0].max()
    N_PT_vs_QAOA_vs_T.append(N_PT_vs_QAOA)
    ax.plot(sizes, N_PT_vs_QAOA, label=T[T_index])
ax.set_yscale('log')
ax.legend()
fig.show()
#%%
fig, ax = plt.subplots(dpi=150)
ax.plot(sizes, N_PT_vs_QAOA_vs_T[1]/N_PT_vs_QAOA_vs_T[0], '.-')
# ax.set_yscale('log')
fig.show()

# %% SIMULATIONS SAVING STEPS TO GS no T
if load_data == False:
    t = time.perf_counter()
    steps_to_GS_vs_size_configs_runs = [[[] for _ in range(N_configs)] for _ in sizes]
    for s, size in enumerate(sizes):
        print('Size', size, 'of', sizes)
        for c, (J, rng, E_min) in enumerate(zip(J_vs_size_configs[s], rng_vs_size_configs[s], E_min_vs_size_configs[s])):
            seeds = np.random.SeedSequence(rng.integers(0, runs_per_config*10000)).spawn(runs_per_config)

            steps_to_GS_vs_size_configs_runs[s][c] = Parallel(n_jobs=threads)(delayed(mc.steps_to_ground_state_no_T)
                 (1 / T, J, max_MCS, np.random.default_rng(seed=seeds[i]), E_min, tempering) for i in range(runs_per_config))

    print('Elapsed time', time.perf_counter() - t)

    steps_to_GS_vs_size_configs = [ [] for _ in sizes]
    for s, size in enumerate(sizes):
        for c in range(N_configs):
            steps_to_GS_vs_size_configs[s].append(steps_to_GS_vs_size_configs_runs[s][c])
        steps_to_GS_vs_size_configs[s] = np.array(steps_to_GS_vs_size_configs[s]).reshape((runs_per_config*N_configs))

    size_index = 0
    config_index = 0
    size = sizes[size_index]
    steps_to_GS = steps_to_GS_vs_size_configs[size_index][config_index]

    if save_data:
        np.savez(f'Data/QAOA/steps_to_GS, sizes = {sizes}, T in [{T0},{Tf}], copies = {copies}, N_configs = {N_configs}, '
                 f'MCS = {max_MCS}', steps_to_GS_vs_size_configs=steps_to_GS_vs_size_configs)
else:
    data = np.load(f'Data/QAOA/steps_to_GS, sizes = {sizes}, T in [{T0},{Tf}], copies = {copies}, N_configs = {N_configs}, '
                 f'MCS = {max_MCS}.npz')
    steps_to_GS_vs_size_configs = data['steps_to_GS_vs_size_configs']

# %% Calculate P to have reached GS
P_to_have_reached_GS_vs_size_vs_MC_step = [[] for _ in sizes]
for s, (size, steps_to_GS_vs_configs) in tqdm(enumerate(zip(sizes, steps_to_GS_vs_size_configs))):
    P_to_have_reached_GS_vs_size_vs_MC_step[s] = exact.p_to_GS_vs_steps_no_T(steps_to_GS_vs_configs, max_MCS*size)

# %% PT vs QAOA
fig, ax = plt.subplots(dpi=150)
for s in range(len(sizes)):
    try:
        N_PT_vs_QAOA[s] = np.where(
            (1 - P_to_have_reached_GS_vs_size_vs_MC_step[s] > (1 - 2 ** (-sizes[s] / 2)) ** np.arange(max_MCS * sizes[s]))[
            1:])[0].max()
    except:
        N_PT_vs_QAOA[s] = 1
ax.plot(sizes, N_PT_vs_QAOA)
ax.set_yscale('log')
# ax.legend()
fig.show()

#%% Plot probability to NOT have reached GS for cerain size and temperature vs MC_step
size_index = 0
P_to_GS = P_to_have_reached_GS_vs_size_vs_MC_step[size_index]
y = (1-P_to_GS[::4,::10]).T
colors = plt.cm.plasma(np.linspace(0, 1, y.shape[1]))
fig, ax = plt.subplots(dpi=150)
# fig, ax = plt.subplots(dpi=150)
for i in range(y.shape[1]):
    ax.plot(y[:, i], color=colors[i])
ax.plot((1-P_to_GS[copies,::10]).T, 'k')
# ax.plot(1-P_to_GS[100,::10].T)
ax.set_yscale('log')
ax.set_xlim([0,200])
fig.suptitle(f'probability to not reach the GS vs MC_step for size = {sizes[size_index]}')
fig.show()

#%% Calculate MC steps to reach a certain probability for a certain temperature vs size.
p  = 0.5
steps_to_p_vs_beta_size = np.zeros([copies * 2, len(sizes)])
for c in range(copies*2):
    for s in range(len(sizes)):
        try:
            steps_to_p_vs_beta_size[c,s] = np.where( 1 - P_to_have_reached_GS_vs_size_vs_MC_step[s][c,:] < p)[0].min()
        except:
            steps_to_p_vs_beta_size[c, s] = np.nan

#%% Plot MC steps to reach a certain probability vs size for diferent temperatures.
y = steps_to_p_vs_beta_size[4::4,:].T
colors = plt.cm.plasma(np.linspace(0, 1, y.shape[1]))

fig, ax = plt.subplots(dpi=150)
for i in range(y.shape[1]):
    ax.plot(sizes, y[:, i], '.-', color=colors[i])
ax.set_yscale('log')
fig.suptitle('MC steps to reach the GS 50% of the time vs size for different Ts')
fig.show()

#%% Plot MC steps to reach a certain probability vs size for T = ||J||
fig, ax = plt.subplots(dpi=150)
ax.plot(sizes, 0.5*(steps_to_p_vs_beta_size[106,:]+steps_to_p_vs_beta_size[107,:]).T, '.-')
ax.set_yscale('log')
fig.suptitle('MC steps to reach the GS 50% of the time vs size for T = 1.5*||J||')
fig.show()
# %% SIMULATIONS SAVING PROBABILITY REACH GS
t = time.perf_counter()
P_sample_GS_vs_size_configs = [[] for size in sizes]

for s, size in enumerate(sizes):
    print('Size', size, 'of', sizes)
    P_sample_GS_vs_size_configs[s] = np.array(
        Parallel(n_jobs=threads)
        (delayed(mc.probability_ground_state_vs_MCS)
         (1 / T, J, MCS_0, max_MCS, rng, E_min, tempering) for J, rng, E_min in
         zip(J_vs_size_configs[s], rng_vs_size_configs[s], E_min_vs_size_configs[s])))

print('Elapsed time', time.perf_counter() - t)

size_index = 0
config_index = 0
size = sizes[size_index]
P_vs_t = P_sample_GS_vs_size_configs[size_index][config_index]
E_min = E_min_vs_size_configs[size_index][config_index]
J = J_vs_size_configs[size_index][config_index]

# %% Plot probability of reaching the ground state a PT-MCMC step vs temperature
fig, ax = plt.subplots(dpi=150)
ax.plot(Ti, P_vs_t[-1, ::2], '.-')
ax.plot(Ti, P_vs_t[-1, 1::2], '.-')
ax.plot(Ti, np.ones(copies) * 2 / 2 ** (size))
ax.set_yscale('log')
ax.set_ylabel('P sample GS')
ax.set_xlabel('T / || J ||')
ax.set_xscale('log')
fig.suptitle(f'n={size}')
fig.show()

# %% Previous plot vs different number of MCS
fig, ax = plt.subplots(dpi=150)

colors = plt.cm.jet(np.linspace(0, 1, P_vs_t.shape[0]))
for i, y in enumerate(P_vs_t[:, ::2]):
    ax.plot(Ti, y, color=colors[i])
ax.plot(Ti, np.ones(copies) * 2 / 2 ** (size))
ax.set_yscale('log')
ax.set_ylabel('P sample GS')
ax.set_xlabel('T / || J ||')
ax.set_xscale('log')
fig.show()

# %% Probability of reaching the ground state a PT-MCMC step at a given temperature vs MCS
fig, ax = plt.subplots(dpi=150)
y = P_vs_t[:,0::4 ]
# y = P_vs_t[:, 30:32]
colors = plt.cm.jet(np.linspace(0, 1, y.shape[1]))
for i in range(y.shape[1]):
    ax.plot(y[:, i], '.-', color=colors[i], linewidth=0.5)
ax.set_ylim([1e-6, y[-1].max() * 1.1])
ax.set_yscale('log')
fig.show()


# %% Calculate configurational averages and errors
P_sample_GS_config_avg_vs_size = [[] for _ in range(len(sizes))]
P_sample_GS_config_avg_vs_size_no_T = [[] for _ in range(len(sizes))]
for s in range(len(sizes)):
    P_sample_GS_config_avg_vs_size[s] = np.mean(P_sample_GS_vs_size_configs[s], 0)
    P_sample_GS_config_avg_vs_size_no_T[s] = np.mean(P_sample_GS_vs_size_configs_no_T[s], 0)
#
# std_P_sample_GS_config_avg_vs_size = np.std(P_sample_GS_vs_size_configs,1)
# std_P_sample_GS_config_avg_vs_size = std_P_sample_GS_config_avg_vs_size[:,MCS_avg_index,:]
# std_P_sample_GS_config_avg_vs_size = (std_P_sample_GS_config_avg_vs_size[:,::2] + std_P_sample_GS_config_avg_vs_size[:,1::2])/2
#
# if save_data:
#     np.savez(f'Data/QAOA/Probabilities and temperatures, sizes = {sizes}, T in [{T0},{Tf}], copies = {copies}, '
#                 f'N_configs = {N_configs}, MCS = {max_MCS}',P_sample_GS_vs_size_configs=P_sample_GS_vs_size_configs,Ti=Ti)
#
#     np.savez(f'Data/QAOA/Probabilities avg and errors and temperatures, sizes = {sizes}, T in [{T0},{Tf}], copies = {copies}, '
#                 f'N_configs = {N_configs}, MCS = {max_MCS}',P_sample_GS_config_avg_vs_size=P_sample_GS_config_avg_vs_size,std_P_sample_GS_config_avg_vs_size=std_P_sample_GS_config_avg_vs_size,Ti=Ti)

# %% P_sample_GS_config_avg vs T for different sizes
fig, ax = plt.subplots(dpi=200)
colors = plt.cm.Set1(np.linspace(0, 1, len(sizes)))
for size_index, size in enumerate(sizes):
    ax.errorbar(Ti, P_sample_GS_config_avg_vs_size[size_index], yerr=std_P_sample_GS_config_avg_vs_size[size_index]/np.sqrt(N_configs), label=f'n={size}', color=colors[size_index])
    ax.plot(Ti, np.ones(copies) * 2 / 2 ** (size), ':', color=colors[size_index])
ax.set_yscale('log', base=2)
# ax.set_ylim([1e-8, 2e-1])
ax.set_xscale('log')
ax.set_ylabel('P sample GS')
ax.set_xlabel('T / || J ||')
ax.legend()
fig.suptitle(f'Tempering = {tempering}')
# plt.savefig( f'Figures/P_reach_GS_config_avg vs T for different sizes, sizes = {sizes}, T/||J|| in [{T0},{Tf}], copies = {copies}, '
#             f'N_configs = {N_configs}, MCS = {max_MCS}.pdf')
# fig.savefig( 'Figures/P',dpi=50)
fig.show()


# %% P_sample_GS vs T for different sizes
fig, ax = plt.subplots(dpi=150)
colors = plt.cm.Set1(np.linspace(0, 1, len(sizes)))
for size_index, size in enumerate(sizes):
    ax.plot(Ti, P_sample_GS_config_avg_vs_size[size_index], label=f'n={size}', color=colors[size_index])
    ax.plot(Ti, np.ones(copies) * 2 / 2 ** (size), ':', color=colors[size_index])
ax.set_yscale('log', base=2)
# ax.set_ylim([1e-8, 2e-1])
ax.set_xscale('log')
ax.set_ylabel('P sample GS')
ax.set_xlabel('T / || J ||')
ax.legend()
fig.suptitle(f'Tempering = {tempering}')
# plt.savefig( f'Figures/P_reach_GS_config_avg vs T for different sizes, sizes = {sizes}, T/||J|| in [{T0},{Tf}], copies = {copies}, '
#             f'N_configs = {N_configs}, MCS = {max_MCS}.pdf')
# fig.savefig( 'Figures/P',dpi=50)
fig.show()


# %% P_sample_GS_config_avg vs size for diferent Ts
y = P_sample_GS_config_avg_vs_size[:, 0::copies // 20]
colors = plt.cm.plasma(np.linspace(0, 1, y.shape[1]))

fig, ax = plt.subplots(dpi=300)
for i in range(y.shape[1]):
    ax.plot(sizes, y[:, i], '.-', color=colors[i])

ax.set_ylim([1e-6, 2e-1])
ax.set_ylabel('P sample GS')
ax.set_xlabel('n')
ax.set_yscale('log', base=2)
fig.suptitle(f'Tempering = {tempering}')
fig.show()

# %% P_sample_GS_config_avg for T = ||J|| vs size
fig, ax = plt.subplots(dpi=300)
ax.plot(sizes, P_sample_GS_config_avg_vs_size[:,copies//2 ], '.-')
ax.set_ylabel('P sample GS')
ax.set_xlabel('n')
ax.set_yscale('log', base=2)
fig.suptitle(f'Tempering = {tempering} \n T / || J || = 1')

fig.show()

# %% P_sample_GS_config_avg for T = ||J|| vs size
fig, ax = plt.subplots(dpi=300)
ax.plot(sizes, P_sample_GS_config_avg_vs_size[:,copies//2 ], '.-')
ax.set_ylabel('P sample GS')
ax.set_xlabel('n')
ax.set_yscale('log', base=2)
fig.suptitle(f'Tempering = {tempering} \n T / || J || = 1')

fig.show()

# %% Max P_sample_GS_config_avg vs size for any T
fig, ax = plt.subplots(dpi=300)
ax.plot(sizes, np.max(P_sample_GS_config_avg_vs_size,1), '.-')
ax.set_ylabel('P sample GS')
ax.set_xlabel('n')
ax.set_yscale('log', base=2)
fig.suptitle(f'max P sample GS for any T')
fig.show()

# %% Max P_sample_GS_config_avg vs size for any T
T_max_P = [ Ti[ np.where(P_sample_GS_config_avg_vs_size[i,:] == np.max(P_sample_GS_config_avg_vs_size, 1)[i]) ] for i in range(len(sizes)) ]
fig, ax = plt.subplots(dpi=150)
ax.plot(sizes, T_max_P, '.-')
ax.set_ylabel('T max P')
ax.set_xlabel('n')
ax.set_yscale('log')
ax.set_ylim([1e-2,1e0])
fig.suptitle(f' T such that P reach GS is max ')
fig.tight_layout()
fig.show()
