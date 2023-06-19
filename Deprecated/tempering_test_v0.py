import Modules.spin_glass as sg

import numpy as np
from numpy.polynomial import Polynomial
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import Modules.figures as f

from importlib import reload

sg = reload(sg)
f = reload(f)

plt.rcParams.update({
    "text.usetex": True})

# %% Parameters
method = 'Numba 2'
T_dist = 'geometric'
save_pdf = False
show_figures = True
T_dists_fig = False
T0 = 0.01
Tf = 2
copies = 12
a = 10

connectivity = '2D'
seed = 72355  # 152  # 1845
size = 15 ** 2

total_steps = int(400e3)
steps_for_avg = int(200e3)
# %% Initialize

rng = np.random.default_rng(seed)
J = sg.connectivity_matrix(size, connectivity, 'gaussian_EA', rng=rng)

if method == 'Numba 1' or method == 'Numba 2' or method == 'Numba 3':
    import Modules.monte_carlo as mc
elif method == 'Cython 1' or method == 'Cython 2' or method == 'Cython 3':
    import Modules.mc_cython as mc
mc = reload(mc)

if T_dist == 'geometric':
    T = np.geomspace(T0, Tf, copies)
elif T_dist == 'linear':
    T = np.linspace(T0, Tf, copies)
elif T_dist == 'rational':
    T = 1 / np.linspace((1 / T0) ** (1 / a), (1 / Tf) ** (1 / a), copies) ** a
    T_dist = f'{T_dist} {a}'
elif T_dist == 'optimal':
    T = np.geomspace(T0, Tf, copies * 2)
    *_, E_vs_t, T_vs_t = mc.mc_evolution(1 / T, J, steps=total_steps, start=None, eq_steps=1, rng=rng,
                                         trajectories=True)
    T = mc.optimal_temperature_distribution(T, E_vs_t, T_vs_t, steps_for_avg=steps_for_avg, target=target)
    copies = len(T)
else:
    T = None

# %% T dist
if T_dists_fig:
    T_dists = ['linear', 'geometric', 'rational', 'optimal']
    fig, ax = plt.subplots(dpi=300)

    for T_dist in T_dists:
        if T_dist == 'geometric':
            T = np.geomspace(T0, Tf, copies)
        elif T_dist == 'linear':
            T = np.linspace(T0, Tf, copies)
        elif T_dist == 'rational':
            T = 1 / np.linspace((1 / T0) ** (1 / a), (1 / Tf) ** (1 / a), copies) ** a
            T_dist = f'{T_dist} {a}'
        elif T_dist == 'optimal':
            T = np.geomspace(T0, Tf, copies * 2)
            *_, E_vs_t, T_vs_t = mc.mc_evolution(1 / T, J, steps=total_steps, start=None, eq_steps=1, rng=rng,
                                                 trajectories=True)
            T = mc.optimal_temperature_distribution(T, E_vs_t, T_vs_t, steps_for_avg=50000, target=target)
            copies = len(T)

        ax.plot(T, marker='.', label=T_dist)
    plt.legend()
    plt.tight_layout()

    pdf_temps_fig = f'/home/gabriel/OneDrive/2021/Avaqus/Architecture/Figures/Temperature distribution {connectivity}.pdf'
    plt.savefig(pdf_temps_fig)
# %% Trajectories in energy and tempering
beta = 1 / T
copies = len(T)
*_, E_vs_t, T_vs_t = mc.mc_evolution(beta, J, steps=total_steps, start=None, eq_steps=1, rng=rng, trajectories=True)

E_vs_t = np.delete(E_vs_t, 0, 0)
T_index_vs_t = np.argsort(np.argsort(T_vs_t))
T_E_vs_t = np.zeros([E_vs_t.shape[0], E_vs_t.shape[1]])
for copy in range(copies):
    T_E_vs_t[:, copy] = np.array([[T_vs_t[i, copy]] * 10 for i in range(1, T_vs_t.shape[0])]).flatten()

T_E_index_vs_t = np.argsort(T_E_vs_t)
E_fixed_T_vs_t = np.zeros_like(E_vs_t)
for i in range(E_vs_t.shape[0]):
    E_fixed_T_vs_t[i, :] = E_vs_t[i, T_E_index_vs_t[i, :]]

plotted_trajectories = np.array([0, int((copies - 1) / 2), copies - 1])
trajectories_start = 1000
l = int(np.sqrt(copies))
h = int(copies / l)
if l * h != copies:
    for i in range(2, copies):
        if copies % i == 0:
            l = int(i)
            break
    h = int(copies / l)

# %% E_vs_t fixed copy
fig1, ax = plt.subplots(dpi=300)
f.multiline_color_change(E_vs_t[trajectories_start:, plotted_trajectories],
                         T_E_vs_t[trajectories_start:, plotted_trajectories], fig1, ax, linewidth=0.5)
fig1.suptitle(r'$E_i(t)$ for a fixed $i$, color according to $T_i(t)$')
plt.tight_layout()
if show_figures:
    plt.show()

# %% E_vs_t fixed T
fig2, ax = plt.subplots(dpi=300)
f.multiline(T[plotted_trajectories], E_fixed_T_vs_t[trajectories_start:, plotted_trajectories], ax=ax, linewidth=0.5)
fig2.suptitle(r'$E_i(t)$ for a fixed temperature')
plt.tight_layout()
if show_figures:
    plt.show()

# %% Gaussian Fit of E(T), each temperature in one plot
p_T = []
x_T = []
mu_T = []
sigma_T = []
E_vs_t_T = []

fig3, ax = plt.subplots(ncols=l, nrows=h, dpi=300, figsize=[3 * l, 3 * h])
ax = ax.ravel()

for i in range(l * h):
    E_vs_t_T.append(E_fixed_T_vs_t[-steps_for_avg:-1, i])

    mu, sigma = norm.fit(E_vs_t_T[i])
    # x = np.linspace(E_vs_t_T[i].min(), E_vs_t_T[i].max(), 200)
    x = np.linspace(E_vs_t.min(), E_vs_t.max(), 10000)
    p = norm.pdf(x, mu, sigma)
    mu_T.append(mu)
    sigma_T.append(sigma)
    x_T.append(x)
    p_T.append(p)

    ax[i].hist(E_vs_t_T[i], 20, density=True)
    ax[i].plot(x, p, 'k', linewidth=1)
    ax[i].set_xlim([E_vs_t_T[i].min(), E_vs_t_T[i].max()])
mu = np.array(mu_T)
sigma = np.array(sigma_T)

fig3.suptitle(r'$P(E)$ for fixed temperatures')
plt.tight_layout()
if show_figures:
    plt.show()
# %% Gaussian Fit of E(T), all temperatures in one plot
fig11, ax = plt.subplots(dpi=300, figsize=[16 / 3, 9 / 3])
for i in range(copies):
    ax.plot(x_T[i], p_T[i])
    # ax.hist(E_vs_t_T[i], 20, density = True)

plt.tight_layout()
if show_figures:
    plt.show()

# %% Mu and sigma for each temperature
mu_fit = Polynomial.fit(T, mu, 2)
sigma_fit = Polynomial.fit(T, sigma, 2)

fig4, ax = plt.subplots(nrows=2, dpi=300)
ax[0].plot(T, np.array(mu_T), label=r'$\mu$', color='k', marker='.')
ax[0].plot(mu_fit.linspace(100)[0], mu_fit.linspace(100)[1], label=r'fit', linestyle=':', color='r')
ax[0].legend()
ax[1].plot(T, np.array(sigma_T), label=r'$\sigma$', color='k', marker='.')
ax[1].plot(sigma_fit.linspace(100)[0], sigma_fit.linspace(100)[1], label=r'fit', linestyle=':', color='r')
ax[1].set_xlabel('T')
ax[1].legend()
fig4.suptitle(r'$\mu(E)$ and $\sigma(E)$ for fixed temperatures')

plt.tight_layout()
if show_figures:
    plt.show()

# %% T_vs_t
fig5, ax = plt.subplots(ncols=l, nrows=h, dpi=300, figsize=[3 * l, 3 * h])
ax = ax.ravel()
for i in range(l * h):
    ax[i].hlines(T, 0, len(T_vs_t), color='gray', linewidth=0.5)
    ax[i].plot(T_vs_t[:, i], linewidth=0.5, color='k')
    # ax[i].set_ylim([0.1, copies + 0.1])
fig5.suptitle(r'$T(t)$')
plt.tight_layout()
if show_figures:
    plt.show()

# %% T_index_vs_t
fig6, ax = plt.subplots(ncols=l, nrows=h, dpi=300, figsize=[3 * l, 3 * h])
ax = ax.ravel()
for i in range(l * h):
    ax[i].hlines(np.arange(copies), 0, len(T_index_vs_t), color='gray', linewidth=0.5)
    ax[i].plot(T_index_vs_t[:, i], linewidth=0.5, color='k')
    # ax[i].set_ylim([0.1, copies + 0.1])
fig6.suptitle('$T_{index}(t)$')
plt.tight_layout()
if show_figures:
    plt.show()

# %% T_vs_t distribution for each copy
fig7, ax = plt.subplots(ncols=l, nrows=h, dpi=300, figsize=[3 * l, 3 * h])
ax = ax.ravel()
for i in range(l * h):
    ax[i].hist(T_index_vs_t[:, i], copies, range=(0, copies), align='left', density=True, color='k')
    ax[i].set_yticks([])
    ax[i].set_ylim([0, 1])
fig7.suptitle('$P(T_{index}(t))$')
plt.tight_layout()
if show_figures:
    plt.show()

# %% Mean and std of T distribution for each copy vs uniform distribution
temper_mean_goodness = np.abs(1 - np.mean(T_index_vs_t, 0) / np.mean(np.arange(copies)))
temper_std_goodness = np.abs(1 - np.std(T_index_vs_t, 0) / np.sqrt(copies ** 2 / 12))
temper_goodness = (temper_std_goodness + temper_mean_goodness) / 2

fig8, ax = plt.subplots(ncols=3, dpi=300, figsize=[16 / 2, 9 / 2])
ax[0].bar(np.arange(copies), temper_mean_goodness, linewidth=0.5, color='k')
ax[0].title.set_text(r'$|1-\frac{\mu[T(t)]}{\mu_{uniform}}|$')
ax[1].bar(np.arange(copies), temper_std_goodness, linewidth=0.5, color='k')
ax[1].title.set_text(r'$|1-\frac{\sigma[T(t)]}{\sigma_{uniform}}|$')
ax[2].bar(np.arange(copies), temper_goodness, linewidth=0.5, color='k')
ax[2].title.set_text(
    r'$\frac{1}{2}\left(|1-\frac{\mu[T(t)]}{\mu_{uniform}}| + |1-\frac{\sigma[T(t)]}{\sigma_{uniform}}|\right)$')

[ax[i].set_xlabel('Copy index') for i in range(3)]
[ax[i].set_ylim([0, 1]) for i in range(3)]
[ax[i].set_xticks([]) for i in range(3)]
[ax[i].set_yticks([]) for i in range(1, 3)]
plt.tight_layout()
if show_figures:
    plt.show()

# %% Total number of swaps vs t
total_swaps_vs_t = [np.sum(T_index_vs_t[i + 1, :] - T_index_vs_t[i, :] != 0) / copies for i in
                    range(T_index_vs_t.shape[0] - 1)]

cumulative_swaps_per_copy_vs_t = np.zeros_like(T_index_vs_t)
cumulative_swaps_per_T_vs_t = np.zeros_like(T_index_vs_t)
swaps_low = np.zeros(copies)
swaps_high = np.zeros(copies)
for i in range(T_index_vs_t.shape[0] - 1):
    cumulative_swaps_per_copy_vs_t[i, :] = T_index_vs_t[i + 1, :] - T_index_vs_t[i, :] != 0
    for c in range(copies):
        c_index = np.argwhere(T_index_vs_t[i, :] == c)[0][0]
        cumulative_swaps_per_T_vs_t[i, c] = T_index_vs_t[i + 1, c_index] - T_index_vs_t[i, c_index] != 0
        if T_index_vs_t[i + 1, c_index] - T_index_vs_t[i, c_index] > 0:
            swaps_high[c] += 1
        elif T_index_vs_t[i + 1, c_index] - T_index_vs_t[i, c_index] < 0:
            swaps_low[c] += 1
cumulative_swaps_per_copy_vs_t = np.cumsum(cumulative_swaps_per_copy_vs_t, 0)
cumulative_swaps_per_T_vs_t = np.cumsum(cumulative_swaps_per_T_vs_t, 0)

fig9, ax = plt.subplots(dpi=300)
ax.plot(total_swaps_vs_t, linewidth=0.1, color='k')
ax.title.set_text('Total number of swaps')
ax.set_ylim([0, 1])
ax.set_xlabel('t')
plt.tight_layout()
if show_figures:
    plt.show()

# %% Cumulative number of swaps per copy vs t - BROKEN
fig10, ax = plt.subplots(dpi=300)
f.multiline_color_change(np.log10(cumulative_swaps_per_copy_vs_t[10:, :]), T_E_vs_t[10:, :], fig10, ax)
ax.title.set_text('Cumulative number of swaps per copy')
# ax.set_ylim([0, np.shape(T_index_vs_t)[0] - 1])
ax.set_xlabel('t')
ax.set_ylabel('log scale')
plt.tight_layout()
if show_figures:
    plt.show()

# %% Cumulative number of swaps per T vs t
fig10, ax = plt.subplots(dpi=300)
f.multiline(T, np.log10(cumulative_swaps_per_T_vs_t[10:, :]), ax=ax)
ax.title.set_text('Cumulative number of swaps per T')
# ax.set_ylim([0, np.shape(T_index_vs_t)[0] - 1])
ax.set_xlabel('t')
ax.set_ylabel('log scale')
# ax.set_yscale('log')
plt.tight_layout()
if show_figures:
    plt.show()

# %% Acceptance probability per T
fig12, ax = plt.subplots(dpi=300)
xs = np.zeros([2, copies])
for i in range(copies):
    xs[0, i] = i
    xs[1, i] = i + 1
# p_accept_T_theory = [2 - np.trapz(np.maximum(p_T[i], p_T[i + 1]), x) for i in range(copies - 1)]
p_accept_T = cumulative_swaps_per_T_vs_t[-1, :] / cumulative_swaps_per_T_vs_t.shape[0]
f.multiline(T, np.array([p_accept_T, p_accept_T]), xs=xs.T, ax=ax)
# f.multiline(T, np.array([p_accept_T_theory, p_accept_T_theory]), xs=xs.T, ax=ax, linestyle='--')
# ax.set_yscale('log')
ax.set_yticks([0.01, 0.1, 1])
ax.set_xticks([])
plt.tight_layout()
ax.title.set_text('Acceptance probability per T')
if show_figures:
    plt.show()
# %% Print multipage pdf
if save_pdf:
    path = '/home/gabriel/OneDrive/2021/Avaqus/Architecture/Figures/'
    pdf_name = f'Tempering tests {method}, connectivity {connectivity}, T0 = {T0}, Tf = {Tf}, copies = {copies}, T_dist = {T_dist}.pdf'
    figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10]
    pdf = matplotlib.backends.backend_pdf.PdfPages(path + pdf_name)
    for fig in figs:
        pdf.savefig(fig)
    pdf.close()
    plt.close('all')

# %% Option 1
'''
Returns an optimal temperature distribution 'T_opt' for parallel tempering where the acceptance probability
of a temper swap is accept_prob for each temperature.
:param T0: Initial value of temperature distribution.
:param Tf: Final value for temperature distribution.
:param J: Adjacency matrix of the problem.
:param init_steps: Number of MC steps not used for the calculation of accept probability.
:param avg_steps:
:param accept_prob: Target acceptance probability.
:param error_accept: Admissible error for the measured acceptance probability.
'''

total_steps = int(400e3)
steps_for_avg = int(200e3)
init_steps = total_steps - steps_for_avg

accept_prob_0 = 0.15
fig, ax = plt.subplots(ncols=2, dpi=300)
T0 = 0.01
Tf = 2
T = np.geomspace(T0, Tf, 8)
# T = np.linspace(T0, Tf, 8)
accept_prob_meas = np.zeros([len(T)])

accept_prob_0 = 0.15
accept_prob = np.ones([len(T)]) * accept_prob_0
accept_prob[0] = accept_prob_0 / 2
accept_prob[-1] = accept_prob_0 / 2
k = 0
check = 0
while np.any(accept_prob_meas < accept_prob) or check < 2:  # or  np.any(accept_prob_meas > accept_prob*4) :
    accept_prob = np.ones([len(T)]) * accept_prob_0
    accept_prob[0] = accept_prob_0 / 2
    accept_prob[-1] = accept_prob_0 / 2
    # ax[1].plot(T, accept_prob_meas, linewidth = 0.5 + 0.1*k )
    k += 1
    copies = len(T)
    *_, T_vs_t = mc.mc_evolution(1 / T, J, steps=total_steps, start=None, eq_steps=1, rng=rng, trajectories=True)
    temper_steps = int(steps_for_avg / 10)
    T_index_vs_t = np.argsort(np.argsort(T_vs_t[-temper_steps:, :]))
    swaps = np.zeros(copies)
    # for i in range(temper_steps - 1):
    #     for c in range(copies):
    #         c_index = np.argwhere(T_index_vs_t[i, :] == c)[0][0]
    #         if T_index_vs_t[i + 1, c_index] - T_index_vs_t[i, c_index] != 0:
    #             swaps[c] += 1
    for i in range(temper_steps - 1):
        for c in range(copies):
            c_index = np.argwhere(T_index_vs_t[i, :] == c)[0][0]
            if T_index_vs_t[i + 1, c_index] - T_index_vs_t[i, c_index] > 1:
                swaps_high[c] += 1
    accept_prob_meas = swaps / temper_steps
    print(f'T =  ' + np.array2string(T, precision=3, floatmode='fixed') +
          ', \nP =  ' + np.array2string((accept_prob_meas * 100).astype('int')) + ' \n' + str(check) + '\n')

    if np.all(accept_prob_meas > accept_prob):
        check += 1
    else:
        check = 0

    ax[0].plot(np.linspace(0, 1, len(T)), T, linewidth=0.5 + 0.1 * k, marker='.')
    ax[1].plot(T, accept_prob_meas, linewidth=0, marker='.', markersize=0.5 + 0.1 * k)

    dont_insert = []
    inserted_temps = 0
    for c in range(copies - 1):
        if accept_prob_meas[c] < accept_prob[c] and c not in dont_insert:
            T = np.insert(T, c + 1 + inserted_temps, (T[c + inserted_temps] + T[c + 1 + inserted_temps]) / 2)
            inserted_temps += 1
            dont_insert.append(c + 1)

# trim
kill = 0
for c in range(1, copies - 2):
    if kill == 1:
        kill = 0
        continue
    if c + 1 >= len(T):
        break
    if accept_prob_meas[c] > accept_prob[c] * 3 and accept_prob_meas[c + 1] > accept_prob[c + 1] * 3:
        T[c] = (T[c] + T[c + 1]) / 2
        T = np.delete(T, c + 1, 0)
        accept_prob_meas[c] = (T[c] + T[c + 1]) / 4
        accept_prob_meas = np.delete(accept_prob_meas, c + 1, 0)
        kill = 1

accept_prob = np.ones([len(T)]) * accept_prob_0
accept_prob[0] = accept_prob_0 / 2
accept_prob[-1] = accept_prob_0 / 2
# ax[1].plot(T, accept_prob_meas, linewidth = 0.5 + 0.1*k )
copies = len(T)
*_, T_vs_t = mc.mc_evolution(1 / T, J, steps=total_steps, start=None, eq_steps=1, rng=rng, trajectories=True)
temper_steps = int(steps_for_avg / 10)
T_index_vs_t = np.argsort(np.argsort(T_vs_t[-temper_steps:, :]))
swaps = np.zeros(copies)
for i in range(temper_steps - 1):
    for c in range(copies):
        c_index = np.argwhere(T_index_vs_t[i, :] == c)[0][0]
        if T_index_vs_t[i + 1, c_index] - T_index_vs_t[i, c_index] != 0:
            swaps[c] += 1
accept_prob_meas = swaps / temper_steps

ax[0].plot(np.linspace(0, 1, len(T)), T, linewidth=0.5 + 0.1 * k, marker='.')
ax[1].plot(T, accept_prob_meas, linewidth=1, marker='.', markersize=0.5 + 0.1 * k)
ax[0].set_yscale('log')
ax[1].set_xscale('log')
fig.suptitle(str(k))
plt.show()

# %% Option 1 high low
'''
Returns an optimal temperature distribution 'T_opt' for parallel tempering where the acceptance probability
of a temper swap is accept_prob for each temperature.
:param T0: Initial value of temperature distribution.
:param Tf: Final value for temperature distribution.
:param J: Adjacency matrix of the problem.
:param init_steps: Number of MC steps not used for the calculation of accept probability.
:param avg_steps:
:param accept_prob: Target acceptance probability.
:param error_accept: Admissible error for the measured acceptance probability.
'''

total_steps = int(400e3)
steps_for_avg = int(200e3)
init_steps = total_steps - steps_for_avg

fig, ax = plt.subplots(ncols=2, dpi=300)
T0 = 0.01
Tf = 2
T = np.geomspace(T0, Tf, 8)
accept_high_meas = np.zeros([len(T)])

accept_prob_0 = 0.15
accept_prob = np.ones([len(T)]) * accept_prob_0

k = 0
check = 0
while np.any(accept_high_meas[:-1] < accept_prob[:-1] / 2)  or check < 2:
    accept_prob = np.ones([len(T)]) * accept_prob_0

    k += 1
    copies = len(T)
    accept_high_meas  = mc.mc_evolution(1 / T, J, steps=total_steps, start=None, eq_steps=1, rng=rng, trajectories=False,
                    tempering=steps_for_avg, prob_accept=True)[::-1]
    print(f'T =  ' + np.array2string(T, precision=3, floatmode='fixed') +
          ', \nPh =  ' + np.array2string((accept_high_meas * 100).astype('int')) + ' \n' + str(check) + '\n')

    # if np.all(accept_high_meas[:-1] > accept_prob[:-1] / 2) and np.all(accept_low_meas[1:] > accept_prob[1:] / 2):
    if np.all(accept_high_meas[:-1] > accept_prob[:-1] / 2) :
        check += 1
    else:
        check = 0

    ax[0].plot(np.linspace(0, 1, len(T)), T, linewidth=0.5 + 0.1 * k, marker='.')
    ax[1].plot(T[:-1], accept_high_meas[:-1], linewidth=0, marker='.', markersize=0.5 + 0.1 * k)

    dont_insert = []
    inserted_temps = 0
    for c in range(copies-1):
        if accept_high_meas[c] < accept_prob[c] and c not in dont_insert:
            T = np.insert(T, c + 1 + inserted_temps, (T[c + inserted_temps] + T[c + 1 + inserted_temps]) / 2)
            inserted_temps += 1
            dont_insert.append(c + 1)

kill =0
for trim in range(4):
    for c in range( copies - 2):
        if kill == 1:
            kill = 0
            continue
        if c + 1 >= len(T):
            break
        if accept_high_meas[c] > accept_prob[c] * 3 and accept_high_meas[c + 1] > accept_prob[c + 1] * 3:
            T[c] = (T[c] + T[c + 1]) / 2
            T = np.delete(T, c + 1, 0)
            accept_high_meas = mc.mc_evolution(1 / T, J, steps=total_steps, start=None, eq_steps=1, rng=rng,
                                               trajectories=False, tempering=steps_for_avg, prob_accept=True)[::-1]
            kill = 1
    print(f'T =  ' + np.array2string(T, precision=3, floatmode='fixed') +
          ', \nP =  ' + np.array2string((accept_high_meas * 100).astype('int')) + ' \n' + str(check) + '\n')
    ax[0].plot(np.linspace(0, 1, len(T)), T, color='k', linewidth=trim)
    ax[1].plot(T[:-1], accept_high_meas[:-1], color='k', linewidth=trim)

# copies = len(T)
# accept_prob = np.ones([len(T)]) * accept_prob_0
# *_, T_vs_t = mc.mc_evolution(1 / T, J, steps=total_steps, start=None, eq_steps=1, rng=rng, trajectories=True)
# temper_steps = int(steps_for_avg / 10)
# T_index_vs_t = np.argsort(np.argsort(T_vs_t[-temper_steps:, :]))
# swaps_high = np.zeros(copies)
# swaps_low = np.zeros(copies)
# for i in range(temper_steps - 1):
#     for c in range(copies):
#         c_index = np.argwhere(T_index_vs_t[i, :] == c)[0][0]
#         if T_index_vs_t[i + 1, c_index] - T_index_vs_t[i, c_index] > 0:
#             swaps_high[c] += 1
#         if T_index_vs_t[i + 1, c_index] - T_index_vs_t[i, c_index] < 0:
#             swaps_low[c] += 1
# accept_high_meas = swaps_high / temper_steps
# accept_low_meas = swaps_low / temper_steps

ax[0].set_yscale('log')
ax[1].set_xscale('log')
fig.suptitle(str(k))
plt.show()

# %% Option 2
'''
Returns an optimal temperature distribution 'T_opt' for parallel tempering where the acceptance probability
of a temper swap is accept_prob for each temperature.
:param T0: Initial value of temperature distribution.
:param Tf: Final value for temperature distribution.
:param J: Adjacency matrix of the problem.
:param init_steps: Number of MC steps not used for the calculation of accept probability.
:param avg_steps:
:param accept_prob: Target acceptance probability.
:param error_accept: Admissible error for the measured acceptance probability.
'''

total_steps = int(400e3)
steps_for_avg = int(200e3)
init_steps = total_steps - steps_for_avg
accept_prob = 0.15

fig, ax = plt.subplots(ncols=2, dpi=300)
T0 = 0.01
Tf = 2
T = np.geomspace(T0, Tf, 8)
# T = np.linspace(T0, Tf, 8)
accept_prob_meas = np.zeros([len(T)])

accept_prob_target = np.ones([len(T)]) * accept_prob
accept_prob_target[0] = accept_prob / 2
accept_prob_target[-1] = accept_prob / 2
k = 0
check = 0
while np.any(accept_prob_meas < accept_prob_target) or check < 3:
    accept_prob_target = np.ones([len(T)]) * accept_prob
    accept_prob_target[0] = accept_prob / 2
    accept_prob_target[-1] = accept_prob / 2
    k += 1
    copies = len(T)
    accept_prob_meas = mc.mc_evolution(1 / T, J, steps=total_steps, start=None, eq_steps=1, rng=rng, trajectories=False,
                                       tempering=steps_for_avg, prob_accept=True)[::-1]
    print(f'T =  ' + np.array2string(T, precision=3, floatmode='fixed') +
          ', \nP =  ' + np.array2string((accept_prob_meas * 100).astype('int')) + ' \n' + str(check) + '\n')

    if np.all(accept_prob_meas > accept_prob_target):
        check += 1
    else:
        check = 0

    ax[0].plot(np.linspace(0, 1, len(T)), T, linewidth=0.5 + 0.1 * k, marker='.')
    ax[1].plot(T, accept_prob_meas, linewidth=0, marker='.', markersize=0.5 + 0.1 * k)

    dont_insert = []
    inserted_temps = 0
    for c in range(copies - 1):
        if (accept_prob_meas[c] < accept_prob_target[c] or accept_prob_meas[c + 1] < accept_prob_target[
            c + 1]) and c not in dont_insert:
            T = np.insert(T, c + 1 + inserted_temps, (T[c + inserted_temps] + T[c + 1 + inserted_temps]) / 2)
            inserted_temps += 1
            dont_insert.append(c + 1)

kill =0
for trim in range(4):
    for c in range(2, copies - 2):
        if kill == 1:
            kill = 0
            continue
        if c + 1 >= len(T):
            break
        if accept_prob_meas[c] > accept_prob_target[c] * 3 and accept_prob_meas[c + 1] > accept_prob_target[c + 1] * 3:
            T[c] = (T[c] + T[c + 1]) / 2
            T = np.delete(T, c + 1, 0)
            accept_prob_meas = mc.mc_evolution(1 / T, J, steps=total_steps, start=None, eq_steps=1, rng=rng,
                                               trajectories=False,
                                               tempering=steps_for_avg, prob_accept=True)[::-1]
            kill = 1
    print(f'T =  ' + np.array2string(T, precision=3, floatmode='fixed') +
          ', \nP =  ' + np.array2string((accept_prob_meas * 100).astype('int')) + ' \n' + str(check) + '\n')
    ax[0].plot(np.linspace(0, 1, len(T)), T, color='k', linewidth=trim)
    ax[1].plot(T, accept_prob_meas, color='k', linewidth=trim)

ax[0].set_yscale('log')
ax[1].set_xscale('log')
fig.suptitle(str(k))
plt.show()
