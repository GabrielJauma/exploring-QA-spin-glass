import Modules.spin_glass
import Modules.spin_glass as sg
# import Modules.chip_architecture as ca
import Modules.monte_carlo as mc_new
# import Modules.mc_cython as mc_c
import Deprecated.monte_carlo_v5 as mc_c
import Modules.monte_carlo as mc

import numpy as np
# from joblib import Parallel, delayed

import time
import matplotlib.pyplot as plt
import Modules.figures as f

plt.rcParams.update({
    "text.usetex": False})

# from importlib import reload
# ca = reload(ca)
# sg = reload(sg)
# mc = reload(mc)
# mc_c = reload(mc_c)
# f = reload(f)

rng = np.random.default_rng(123)
size = 6 ** 3

T0 = 0.1
Tf = 3
copies = 10
T = np.linspace(T0, Tf, copies)
# T = np.zeros(copies * 2)
# T[0::2] = Ti.copy()
# T[1::2] = Ti.copy()
beta = 1 / T
copies = len(T)


J = sg.connectivity_matrix(size, '3D', 'gaussian_EA')
Jrows, Jvals = J[1][0], J[1][1]
J = J[0]

s = rng.integers(0, 2, (copies, size), np.int8) * 2 - 1
E = mc.cost(J, s)

steps = int(1e4)
steps_until_temp = 10
random_sites = rng.integers(0, size, (steps, copies), dtype='int32')
random_chances = rng.random((steps, copies))
random_tempering = rng.random((steps // steps_until_temp, copies))

# Which tests to do
test_cost = True
test_delta_cost = True
test_mc_step = True
test_delta_cost_and_mc_step = True

# %% TEST optimal temp
T, p = mc.optimal_temperature_distribution(T0, Tf, J, rng, MCS_eq=int(1e6), MCS_avg=int(1e6), accept_prob_min=0.3,
                                           accept_prob_max=0.6, plot=True, copies0=20)

# %% TEST cost
if test_cost:
    E = mc.cost(J, s)
    E_c = mc_new.cost(J, s)
    all_good = np.all(E == E_c)
    print('Test results: Ou' + bool(all_good) * ' Yeah!' + bool(not all_good) * ' Shit!')

# %% TEST delta cost
if test_delta_cost:
    rng = np.random.default_rng(17)
    flip_sites = random_sites[0]

    s = rng.integers(0, 2, (copies, size), np.int8) * 2 - 1
    dE = np.zeros(copies)

    dE_new = np.zeros([copies, size])
    mc_new.delta_cost_total(Jrows, Jvals, s, dE_new)

    all_good = True
    for flip_sites in random_sites:

        mc.delta_cost(Jrows, Jvals, s, dE, flip_sites)

        all_good = np.all([dE[i] == dE_new[i, flip_sites[i]] for i in range(copies)])
        if not all_good:
            break

    print('Test results: Ou' + bool(all_good) * ' Yeah!' + bool(not all_good) * ' Shit!')
#%% TEST mc_step
if test_mc_step:
    rng = np.random.default_rng(121)
    flip_sites = random_sites[0]
    flip_chances = random_chances[0]
    flip_sites[0] = 0
    flip_chances[0] = 0

    copies = 1
    s = rng.integers(0, 2, (copies, size), np.int8) * 2 - 1
    E = mc.cost(J, s)
    dE = np.zeros([copies, size])

    s_new = s.copy()
    E_new = mc.cost(J, s_new)
    dE_new = np.zeros([copies, size])

    mc_new.delta_cost_total(Jrows, Jvals, s, dE)
    s[0][0] *= -1
    mc_new.delta_cost_total(Jrows, Jvals, s, dE)

    mc_new.delta_cost_total(Jrows, Jvals, s_new, dE_new)
    mc_new.mc_step(beta, Jrows, Jvals, s_new, E, dE_new, flip_chances, flip_sites)

    all_good = np.allclose(dE, dE_new)
    print('Test results: Ou' + bool(all_good) * ' Yeah!' + bool(not all_good) * ' Shit!')


# %% TEST mc_step and delta cost

if test_delta_cost_and_mc_step:
    s = rng.integers(0, 2, (copies, size), np.int8) * 2 - 1
    E = mc.cost(J, s)
    dE = np.zeros(copies)

    s_new = s.copy()
    E_new = E.copy()
    dE_new = np.zeros([copies, size])
    mc_new.delta_cost_total(Jrows, Jvals, s_new, dE_new)

    all_good = True
    for n, (flip_sites, flip_chances) in enumerate(zip(random_sites, random_chances)):
        mc.delta_cost(Jrows, Jvals, s, dE, flip_sites)
        mc.mc_step(flip_chances, beta * dE, E, s, dE, flip_sites)

        mc_new.mc_step(beta, Jrows, Jvals, s_new, E_new, dE_new, flip_chances, flip_sites)

        all_good = np.allclose(s, s_new) and np.allclose(E, E_new)
        if not all_good:
            print(n)
            break

    print('Test results: Ou' + bool(all_good) * ' Yeah!' + bool(not all_good) * ' Shit!')

# %% Test mc_loop
rng = np.random.default_rng(79)

s = rng.integers(0, 2, (copies, size), np.int8) * 2 - 1
E = mc.cost(J, s)

s_new = s.copy()
E_new = E.copy()
dE_new = np.zeros([copies, size])
mc_new.delta_cost_total(Jrows, Jvals, s_new, dE_new)


# %%
print(E)
mc.mc_loop(beta, Jrows, Jvals, s, E, steps_until_temp, random_sites, random_chances, random_tempering)
print(E, np.min(E))

# %%
print(E_new)
mc_new.mc_loop(beta, Jrows, Jvals, s_new, E_new, dE_new, steps_until_temp, random_sites, random_chances)
print(E_new, np.min(E_new))
# %% TEST tempering_step
swap = np.arange(copies, dtype=np.int32)
E = E0.copy()
for _ in range(10):
    mc.tempering_step(beta, E, swap, rng.random(beta.size))
E = E[swap]
s = s[swap, :]
print(swap)

# %% TEST numba vs cython speed in size
T0 = 0.1
Tf = 2
copies = 10
T = np.linspace(T0, Tf, copies)
beta = 1 / T

sizes = np.arange(4, 12) ** 3  # steps_list = np.arange(100, 100000, 1000)
steps = int(1e5)

times_c_avg = []
times_n_avg = []
for size in sizes:
    print((np.argwhere(size == sizes)[0][0] + 1) / len(sizes))
    rng = np.random.default_rng(17)
    J = sg.connectivity_matrix(size, '3D', 'gaussian_EA', rng=rng)

    times_c = []
    for i in range(1):
        t_c = time.perf_counter()
        _ = mc_c.mc_evolution(beta, J, steps=steps, start=None, steps_until_temp=10, rng=rng)
        t_c = time.perf_counter() - t_c
        times_c.append(t_c)
    times_c_avg.append(np.array(times_c).mean())

    times_n = []
    for i in range(1):
        t_n = time.perf_counter()
        _ = mc.mc_evolution(beta, J, steps=steps, start=None, rng=rng)
        t_n = time.perf_counter() - t_n
        times_n.append(t_n)
    times_n_avg.append(np.array(times_n).mean())

fig1, ax1 = plt.subplots(dpi=500)
ax1.plot(sizes, np.array(times_c_avg), label='cython')
ax1.plot(sizes, np.array(times_n_avg), label='numba')
ax1.title.set_text('time for ' + str(steps) + ' MC steps')
ax1.set_xlabel('number of spins')
ax1.legend()
plt.show()
# %% TEST numba vs cython speed in number of steps
T0 = 0.1
Tf = 2
copies = 10
T = np.geomspace(T0, Tf, copies)
beta = 1 / T
rng = np.random.default_rng(17)
size = 8 ** 3
J = sg.connectivity_matrix(size, '3D', 'gaussian_EA', rng=rng)

steps_list = np.logspace(2, 7, 7, dtype='int32')

times_c_avg = []
times_n_avg = []
i = 0
for steps in steps_list:
    print((np.argwhere(steps == steps_list)[0][0] + 1) / len(steps_list))

    times_c = []
    for i in range(1):
        t_c = time.perf_counter()
        _ = mc_c.mc_evolution(beta, J, steps=steps, start=None, steps_until_temp=10, rng=rng)
        t_c = time.perf_counter() - t_c
        times_c.append(t_c)
    times_c_avg.append(np.array(times_c).mean())

    times_n = []
    for i in range(1):
        t_n = time.perf_counter()
        _ = mc.mc_evolution(beta, J, steps=steps, start=None, rng=rng)
        t_n = time.perf_counter() - t_n
        times_n.append(t_n)
    times_n_avg.append(np.array(times_n).mean())

fig1, ax1 = plt.subplots(dpi=500)
ax1.plot(steps_list, np.array(times_c_avg), label='cython')
ax1.plot(steps_list, np.array(times_n_avg), label='numba')
ax1.title.set_text(f'time for x MC steps, size ={size}]')
ax1.set_xlabel('number of steps')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend()
plt.show()

# %% TEST parallelization
Parallel(n_jobs=24)(
    delayed(mc.mc_evolution)(size, beta, J, steps=10000, start=None, tempering=True, trajectories=False) for i
    in range(48))
