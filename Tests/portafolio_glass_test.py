import Portafolio.get_financial_data as gfd
import Modules.monte_carlo as mc
import Modules.spin_glass as sg
import matplotlib.pyplot as plt
import numpy as np

# %% Portafolio spin glass test
T0 = 0.1
Tf = 3
copies = 20

MCS_eq = 100
MCS_avg = 100
max_MCS = int(1e4)
error = 0.1

Ti = np.zeros(copies * 2)
Ti[0::2] = np.linspace(T0, Tf, copies)
Ti[1::2] = Ti[0::2].copy()

beta = 1 / Ti
T = Ti[1::2]
rng = np.random.default_rng(123745)

J = sg.connectivity_matrix(10,'portafolio','-',rng=rng)

# size = 20
# adjacency = 'hexagonal_np_2'
# distribution = 'gaussian_EA'
# J = sg.connectivity_matrix(size, adjacency, distribution, rng=rng)
#%%
s, E = mc.mc_evolution(beta, J, 10000)
#%%
s, E = mc.mc_evolution(beta, J, 10000, [s,E])

# %% single disorder config
µ_q2, µ_q4, σ2_q2, σ2_q4, MCS_eq_f, MCS_avg_f = mc.equilibrate_and_average(beta, J, MCS_eq, MCS_avg, max_MCS, max_error=error, rng=rng)
B = 0.5 * (3 - µ_q4 / µ_q2 ** 2)
dBdq4 = -1 / µ_q2
dBdq2 = 2 * µ_q4 / µ_q2 ** 3
B_error = np.sqrt(σ2_q2 * dBdq2 ** 2 + σ2_q4 * dBdq4 ** 2) / np.sqrt(MCS_avg_f)

# %% Plot B vs T and µ_q2, µ_q4, σ2_q2, σ2_q4 vs T
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
ax1.errorbar(T, B, yerr=B_error, label='$B$')
ax1.set_ylim(0, 1.01)
ax1.set_xlim(T[0], T[-1])
ax1.legend()

ax2.plot(T, µ_q2, '.-', label='$\mu_q^2$')
ax2.plot(T, µ_q4, '.-', label='$\mu_q^4$')
ax2.plot(T, σ2_q2, '.-', label='$\sigma^{2}_{q_2}$')
ax2.plot(T, σ2_q4, '.-', label='$\sigma^{2}_{q_4}$')
ax2.set_ylim(0, 1.01)
ax2.set_xlim(T[0], T[-1])
ax2.legend()

fig.suptitle(f'MCS-EQ={MCS_eq}, MCS-AV={MCS_avg}')
fig.tight_layout()
fig.show()


# %% Initialize variables

copies = len(beta)
size = len(J[1])
# s = rng.integers(0, 2, (copies, size), np.int8) * 2 - 1

# %% Which tests to do
test_delta_cost = True

# %% TEST delta cost

if test_delta_cost:

    all_good = True
    dE_total= np.zeros([copies, size])
    mc.delta_cost_total(J[0], J[1], s, dE_total)

    for c in range(copies):
        for i in range(size):
            news = s.copy()
            news[c, i] = -news[c, i]
            dE = (mc.cost(J, news) - mc.cost(J, s))[c]

            all_good = np.allclose( dE, dE_total[c, i] )
            if not all_good:
                break

    print('Test results: Ou' + bool(all_good) * ' Yeah!' + bool(not all_good) * ' Shit!')

# %% TEST delta cost
if test_delta_cost:

    all_good = True
    dE = np.zeros([copies, size])
    mc.delta_cost_total(J[0], J[1], s, dE)

    for c in range(copies):
        for i in range(size):
            news = s.copy()
            news[c, i] = -news[c, i]
            dE = (mc.cost(J, news) - mc.cost(J, s))[c]

            all_good = np.allclose( dE, dE_total[c, i] )
            if not all_good:
                break

    print('Test results: Ou' + bool(all_good) * ' Yeah!' + bool(not all_good) * ' Shit!')