import Modules.spin_glass
import Modules.spin_glass as sg
# import Modules.chip_architecture as ca
import Modules.monte_carlo as mc
# import Modules.mc_cython as mc_c
import Deprecated.monte_carlo_v5 as mc_new

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

# %%
rng = np.random.default_rng()
size = 9 ** 3
T0 = 0.1
Tf = 3
h = 3
v = 1
copies = v * h
Ti = np.linspace(T0, Tf, copies)
T = np.zeros(copies * 2)
T[0::2] = Ti.copy()
T[1::2] = Ti.copy()
beta = 1 / T  # np.linspace(10,0.1, 10) #1/T
J = sg.connectivity_matrix(size, '3D', 'gaussian_EA')
Jrv = Modules.spin_glass.custom_sparse(J)
copies = len(T)

p = rng.random(copies)
swap = np.arange(copies, dtype=np.int32)
change = rng.integers(0, size, copies, dtype='int32')

s0 = rng.integers(0, 2, (copies, size), np.int8)
s0[s0 == 0] = -1
s = s0.copy()
news = s.copy()
news[swap, change] = -news[swap, change]

E0 = mc.cost(J, s0)
dEJ = np.zeros(copies)
newE = mc.cost(J, news)
# %% TEST cost
E = mc.cost(J, s)
E_c = mc_c.cost(J, s)
print(np.all(E == E_c))
# %% TEST delta_cost numba
dE = mc.cost(J, news) - mc.cost(J, s)
dE2 = np.zeros_like(dE)
mc.delta_cost(Jrv, s, dE2, change)
newE = mc.cost(J, news)
newE2 = E + dE2

print(np.allclose(dE, dE2), np.allclose(newE, newE2))

# %%
rng = np.random.default_rng()

s = rng.integers(0, 2, (copies, size), np.int8)
s[s == 0] = -1
s0 = s.copy()

change = rng.integers(0, size, copies, dtype='int32')

dE = np.zeros(copies)
dE_c = np.zeros(copies)
mc.delta_cost(Jrv, s, dE, change)
mc_c.delta_cost(J, dE_c, s0, change, copies)

# %%
for i in range(10000):
    change = rng.integers(0, size, copies, dtype='int32')

    dE = mc.delta_cost(dE, s, Jrows, Jvals, change)
    mc_c.delta_cost(J, dE0, s0, change, copies)

print(np.all(dE == dE0))

# %%
dE = mc.delta_cost(s, J, change)
dE2 = mc.delta_cost2(s, J, change)
print(np.allclose(dE, dE2))

# %% TEST mc_step
rng = np.random.default_rng()
s = rng.integers(0, 2, (copies, size), np.int8)
s[s == 0] = -1
E = mc.cost(J, s)
s0 = s.copy()
E0 = E.copy()
# %%
print(E)
for i in range(1000):
    news = s.copy()
    newE = E.copy()
    news0 = s0.copy()
    newE0 = E0.copy()

    change = rng.integers(0, size, copies, dtype='int32')
    p = rng.random(copies)

    news[swap, change] = -news[swap, change]
    # news0[swap, change] = -news0[swap, change]
    newE = mc.cost(J, news)
    # newE0 = mc_c.cost_p(J, news0)

    mc.mc_step(p, beta * (newE - E), E, s, (newE - E), change)
    # mc_c.mc_step_p(rng, beta, E0, s0, (newE0 - E0), change, copies)

# print('E0 \n', E0, '\n', np.min(E0))
print('E \n', E, '\n', np.min(E), '\n')

# %% TEST mc_step and delta cost

rng = np.random.default_rng()
s = rng.integers(0, 2, (copies, size), np.int8)
s[s == 0] = -1
E = mc.cost(J, s)
s0 = s.copy()
E0 = E.copy()
dE = mc.delta_cost(np.zeros_like(dE), s, Jrows, Jvals, change)
dE0 = E0.copy()  # np.zeros_like(dE)
mc_c.delta_cost(J, dE0, s0, change, copies)
print(np.all(dE == dE0))
# %%
rng = np.random.default_rng(17)
J = sg.connectivity_matrix(size, '3D', 'gaussian_EA')
Jrows, Jvals = Modules.spin_glass.custom_sparse(J)
s = rng.integers(0, 2, (copies, size), np.int8) * 2 - 1
E = mc.cost(J, s)

dE_new = np.empty([copies, size])
mc_new.delta_cost_total(Jrows, Jvals, s, dE)

for i in range(1000):
    change = rng.integers(0, size, copies, dtype='int32')
    p = rng.random(copies)

    dE = mc.delta_cost(dE, s, Jrows, Jvals, change)
    mc.mc_step(p, beta * dE, E, s, dE, change)

    mc_c.delta_cost(J, dE0, s0, change, copies)
    mc_c.mc_step_p(rng, beta, E0, s0, dE0, change, copies)

print('E \n', E, '\n', np.min(E), '\n')
print('E0 \n', E0, '\n', np.min(E0))
print(np.all(E == E0))
print(np.all(s == s0))

# %% Test mc_loop
rng1 = np.random.default_rng(11)
rng2 = np.random.default_rng(11)
s = rng1.integers(0, 2, (copies, size), np.int8)
E = mc.cost(J, s)
s0 = s.copy()
E0 = E.copy()

steps = 1000
random_sites = rng1.integers(0, size, (steps, copies))
random_chances = rng1.random((steps, copies))
random_tempering = rng1.random((int(steps / 10), copies))

# %%
print(E)
mc.mc_loop(beta, Jrows, Jvals, s, E, steps, random_sites, random_chances, random_tempering)
print(E, np.min(E))

# %%
print(E)
mc_c.mc_loop_p(swap, beta, J, s0, E0, steps, rng2)
print(E0, np.min(E0))
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
    print( (np.argwhere(size == sizes)[0][0]+1)/len(sizes) )
    rng = np.random.default_rng(17)
    J = sg.connectivity_matrix(size, '3D', 'gaussian_EA', rng=rng)

    times_c = []
    for i in range(1):
        t_c = time.perf_counter()
        _ = mc_c.mc_evolution(beta, J, steps=steps, start=None, rng=rng)
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
ax1.title.set_text('time for '+str(steps)+' MC steps')
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

steps_list = np.logspace(2, 8, 7, dtype='int32')

times_c_avg = []
times_n_avg = []
i = 0
for steps in steps_list:
    print( (np.argwhere(steps == steps_list)[0][0]+1)/len(steps_list) )

    times_c = []
    for i in range(1):
        t_c = time.perf_counter()
        _ = mc_c.mc_evolution(beta, J, steps=steps, start=None, rng=rng)
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
