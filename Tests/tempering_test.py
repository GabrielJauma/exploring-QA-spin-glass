import Modules.spin_glass as sg
import Modules.monte_carlo as mc

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import Modules.figures as f


from importlib import reload

sg = reload(sg)
mc = reload(mc)
f = reload(f)

plt.rcParams.update({
    "text.usetex": True})

# %% Parameters
save_pdf = False
show_figures = True
T0 = 0.1
Tf = 25
copies = 200
add = 0

adjacency = 'SK'
distribution = 'gaussian_EA'
seed = 7235  # 152  # 1845
size = 14
MCS_eq = 10000

trajectories_start = 1000

rng = np.random.default_rng(seed)
J = sg.connectivity_matrix(size, adjacency, distribution, rng=rng, add=add)
J0 = sg.connectivity_matrix(size, adjacency, distribution, rng=rng, add=add, sparse = False)

Ti = np.linspace(T0, Tf, copies)
T = np.zeros(copies * 2)
T[0::2] = Ti.copy()
T[1::2] = Ti.copy()
copies = len(T)

plotted_trajectories = np.array([0, int((copies - 1) / 2), copies - 1])
plotted_trajectories = np.arange(0,copies, 20)
# %% Trajectories in energy and tempering
E_vs_t, T_vs_t = mc.mc_evolution_tests(1 / T, J, steps=MCS_eq, start=None, rng=rng, tempering=True, trajectories=True, tempering_probabilities=False)
print(E_vs_t.min())
print(np.where(np.isclose(E_vs_t,E_vs_t.min()))[0][0])

E_vs_t = np.delete(E_vs_t, 0, 0)
T_vs_t = np.delete(T_vs_t, 0, 0)

T_ind_vs_t = np.argsort(np.argsort(T_vs_t))
T_E_vs_t = np.zeros([E_vs_t.shape[0], copies])
for copy in range(copies):
    T_E_vs_t[:, copy] = np.array([[T_vs_t[i, copy]] * size for i in range(1, T_vs_t.shape[0])]).flatten()

T_E_index_vs_t = np.argsort(T_E_vs_t)
E_fixed_T_vs_t = np.zeros_like(E_vs_t)
for i in range(E_vs_t.shape[0]):
    E_fixed_T_vs_t[i, :] = E_vs_t[i, T_E_index_vs_t[i, :]]


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
# f.multiline(T[plotted_trajectories], E_fixed_T_vs_t[trajectories_start:, plotted_trajectories], ax=ax, linewidth=0.5)
f.multiline(T[1:2], E_fixed_T_vs_t[trajectories_start:, 1:2], ax=ax, linewidth=0.5)
fig2.suptitle(r'$E_i(t)$ for a fixed temperature')
plt.tight_layout()
if show_figures:
    plt.show()

#%%
E_min = E_vs_t.min()
jump = 1000
P_ground = np.zeros([MCS_eq*size//jump, copies])
for i in np.arange(0, MCS_eq*size, jump):
    print(100*i/(MCS_eq*size))
    if i == 0:
        P_ground[i//jump, :] = np.isclose(E_fixed_T_vs_t[0:i], E_min).sum(0)
    else:
        P_ground[i // jump, :] = (P_0*(i-jump) + np.isclose(E_fixed_T_vs_t[i-jump:i], E_min).sum(0))/i
    P_0 = P_ground[i // jump, :]


#%%
fig, ax = plt.subplots(dpi=300)
ax.plot(P_ground[:,0::2])
ax.set_ylabel('P for different T')
ax.set_xlabel('MCMC step / 1000')
fig.show()

#%% Calculate partition function
def all_bit_strings(N):
    """Return a matrix of shape (2**N, N) of all bit strings that
    can be constructed using 'N' bits. Each row is a different
    configuration, corresponding to the integers 0, 1, 2 up to (2**N)-1"""
    confs = np.arange(2 ** N, dtype=np.int32)
    return np.array([(confs >> i) & 1 for i in range(N)], dtype=np.uint32)

#%%
ss = all_bit_strings(size).astype('int').T
ss = ss*2-1
Es = mc.cost(J,ss)
Z_T = [ np.sum(np.exp(-Es/T_z) ) for T_z in Ti]

#%%
fig, ax = plt.subplots(dpi=300)
ax.plot(Ti[1:], P_ground[-1,2::2], '.')
ax.plot(Ti, 2*np.exp(-E_min/Ti)/Z_T)
ax.set_yscale('log')
ax.set_xlabel('T')
ax.set_ylabel('$P_{ground}$')
fig.suptitle(f'n={size}')

fig.show()
# %% Gaussian Fit of E(T), each temperature in one plot
# p_T = []
# x_T = []
# mu_T = []
# sigma_T = []
# E_vs_t_T = []
#
# fig3, ax = plt.subplots(ncols=h, nrows=l, dpi=300, figsize=[3 * h, 3 * l])
# ax = ax.ravel()
#
# for i in range(l * h):
#     E_vs_t_T.append(E_fixed_T_vs_t[-avg_steps:-1, i])
#
#     mu, sigma = norm.fit(E_vs_t_T[i])
#     # x = np.linspace(E_vs_t_T[i].min(), E_vs_t_T[i].max(), 200)
#     x = np.linspace(E_vs_t.min(), E_vs_t.max(), 10000)
#     p = norm.pdf(x, mu, sigma)
#     mu_T.append(mu)
#     sigma_T.append(sigma)
#     x_T.append(x)
#     p_T.append(p)
#
#     ax[i].hist(E_vs_t_T[i], 20, density=True)
#     ax[i].plot(x, p, 'k', linewidth=1)
#     ax[i].set_xlim([E_vs_t_T[i].min(), E_vs_t_T[i].max()])
# mu = np.array(mu_T)
# sigma = np.array(sigma_T)
#
# fig3.suptitle(r'$P(E)$ for fixed temperatures')
# plt.tight_layout()
# if show_figures:
#     plt.show()
# %% Gaussian Fit of E(T), all temperatures in one plot
# fig11, ax = plt.subplots(dpi=300, figsize=[16 / 3, 9 / 3])
# for i in range(copies):
#     ax.plot(x_T[i] - np.array(x_T).min(), p_T[i])
#     ax.set_xscale('log')
#     # ax.hist(E_vs_t_T[i], 20, density = True)
#
# plt.tight_layout()
# if show_figures:
#     plt.show()

# %% Mu and sigma for each temperature
# mu_fit = Polynomial.fit(T, mu, 2)
# sigma_fit = Polynomial.fit(T, sigma, 2)
#
# fig4, ax = plt.subplots(nrows=2, dpi=300)
# ax[0].plot(T, np.array(mu_T), label=r'$\mu$', color='k', marker='.')
# ax[0].plot(mu_fit.linspace(100)[0], mu_fit.linspace(100)[1], label=r'fit', linestyle=':', color='r')
# ax[0].legend()
# ax[1].plot(T, np.array(sigma_T), label=r'$\sigma$', color='k', marker='.')
# ax[1].plot(sigma_fit.linspace(100)[0], sigma_fit.linspace(100)[1], label=r'fit', linestyle=':', color='r')
# ax[1].set_xlabel('T')
# ax[1].legend()
# fig4.suptitle(r'$\mu(E)$ and $\sigma(E)$ for fixed temperatures')
#
# plt.tight_layout()
# if show_figures:
#     plt.show()

# %% T_vs_t
# fig5, ax = plt.subplots(ncols=h, nrows=l, dpi=300, figsize=[3 * h, 3 * l])
# ax = ax.ravel()
# for i in range(l * h):
#     ax[i].hlines(T, 0, len(T_vs_t[-int(avg_steps / 10):, i]), color='gray', linewidth=0.5)
#     ax[i].plot(T_vs_t[-int(avg_steps / 10):, i], linewidth=0.5, color='k')
#     # ax[i].set_ylim([0.1, copies + 0.1])
# fig5.suptitle(r'$T(t)$')
# plt.tight_layout()
# if show_figures:
#     plt.show()

#%% T_ind_vs_t
fig6, ax = plt.subplots(ncols=h, nrows=l, dpi=300, figsize=[3 * h, 3 * l])
ax = ax.ravel()
for i in range(l * h):
    ax[i].hlines(np.arange(copies), 0, len(T_ind_vs_t[::10, i]), color='gray', linewidth=0.5)
    ax[i].plot(T_ind_vs_t[::10, i], linewidth=0.5, color='k', marker='.', markersize=0.5)
    # ax[i].set_ylim([0.1, copies + 0.1])
fig6.suptitle('$T_{index}(t)$')
plt.tight_layout()
if show_figures:
    plt.show()

# %% T_vs_t distribution for each copy
fig7, ax = plt.subplots(ncols=h, nrows=l, dpi=300, figsize=[3 * h, 3 * l])
ax = ax.ravel()
for i in range(l * h):
    ax[i].hist(T_ind_vs_t[::10, i], copies, range=(0, copies), align='left', density=True, color='k')
    ax[i].set_yticks([])
    ax[i].set_ylim([0, 1])
    ax[i].set_xticks(np.arange(copies))
fig7.suptitle('$P(T_{index}(t))$')
plt.tight_layout()
if show_figures:
    plt.show()

# %% Mean and std of T distribution for each copy vs uniform distribution
temper_mean_goodness = np.abs(1 - np.mean(T_ind_vs_t, 0) / np.mean(np.arange(copies)))
temper_std_goodness = np.abs(1 - np.std(T_ind_vs_t, 0) / np.sqrt(copies ** 2 / 12))
# temper_goodness = (temper_std_goodness + temper_mean_goodness) / 2

fig8, ax = plt.subplots(ncols=2, dpi=300, figsize=[16 / 2, 9 / 2])
ax[0].bar(np.arange(copies), temper_mean_goodness, linewidth=0.5, color='k')
ax[0].title.set_text(r'$|1-\frac{\mu[T(t)]}{\mu_{uniform}}|$')
ax[1].bar(np.arange(copies), temper_std_goodness, linewidth=0.5, color='k')
ax[1].title.set_text(r'$|1-\frac{\sigma[T(t)]}{\sigma_{uniform}}|$')
# ax[2].bar(np.arange(copies), temper_goodness, linewidth=0.5, color='k')
# ax[2].title.set_text(
plt.tight_layout()
#     r'$\frac{1}{2}\left(|1-\frac{\mu[T(t)]}{\mu_{uniform}}| + |1-\frac{\sigma[T(t)]}{\sigma_{uniform}}|\right)$')

[ax[i].set_ylim([0, 1]) for i in range(2)]
[ax[i].set_xticks(np.arange(copies)) for i in range(2)]
[ax[i].set_yticks([]) for i in range(1, 2)]
if show_figures:
    plt.show()

 # %% Total number of swaps vs t
total_swaps_vs_t = [np.sum(T_ind_vs_t[i + 1, :] - T_ind_vs_t[i, :] != 0) / copies for i in
                    range(-int(avg_steps / 10), T_ind_vs_t.shape[0] - 1)]

total_swaps_vs_t = np.array(total_swaps_vs_t)
swaps_copy_vs_t = np.zeros([int(avg_steps / 10), copies]).astype('int32')
swaps_T_vs_t = np.zeros([int(avg_steps / 10), copies]).astype('int32')
swaps_low = np.zeros(copies).astype('int32')
swaps_high = np.zeros(copies).astype('int32')

for i in range(len(swaps_copy_vs_t)-1):
    swaps_copy_vs_t[i, :] = T_ind_vs_t[i + 1 - int(avg_steps / 10), :] - T_ind_vs_t[i - int(avg_steps / 10), :] != 0
    for c in range(copies):
        c_i = np.argwhere(T_ind_vs_t[i - int(avg_steps / 10), :] == c)[0][0]
        swaps_T_vs_t[i, c] = T_ind_vs_t[i + 1 - int(avg_steps / 10), c_i] - T_ind_vs_t[
            i - int(avg_steps / 10), c_i] != 0
        if T_ind_vs_t[i + 1, c_i] - T_ind_vs_t[i, c_i] > 0:
            swaps_high[c] += 1
        elif T_ind_vs_t[i + 1, c_i] - T_ind_vs_t[i, c_i] < 0:
            swaps_low[c] += 1
swaps_copy_vs_t = np.cumsum(swaps_copy_vs_t, 0)
swaps_T_vs_t = np.cumsum(swaps_T_vs_t, 0)

# fig9, ax = plt.subplots(dpi=300)
# ax.plot(total_swaps_vs_t, linewidth=0.1, color='k')
# ax.title.set_text('Total number of swaps')
# ax.set_ylim([0, 1])
# ax.set_xlabel('t')
# plt.tight_layout()
# if show_figures:
#     plt.show()

# %% Cumulative number of swaps per copy vs t - NOT USEFUL
# fig10, ax = plt.subplots(dpi=300)
# f.multiline_color_change(np.log10(swaps_T_vs_t[100:, :]), T_E_vs_t[-int(avg_steps / 10) + 100:, :], fig10, ax)
# ax.title.set_text('Cumulative number of swaps per copy')
# # ax.set_ylim([0, np.shape(T_ind_vs_t)[0] - 1])
# ax.set_xlabel('t')
# ax.set_ylabel('log scale')
# plt.tight_layout()
# if show_figures:
#     plt.show()

# %% Cumulative number of swaps per T vs t and Acceptance probability per T
fig10, ax = plt.subplots(dpi=300, ncols = 2, figsize = [16/2, 9/2])
f.multiline(T, np.log10(1 + swaps_T_vs_t), ax=ax[0])
# f.multiline(T, swaps_T_vs_t[100:, :], ax=ax)
ax[0].title.set_text('Cumulative number of swaps per T')
# ax.set_ylim([0, np.shape(T_ind_vs_t)[0] - 1])
ax[0].set_xlabel('t')
ax[0].set_ylabel('log scale')
# ax.set_yscale('log')
plt.tight_layout()
# if show_figures:
#     plt.show()

# fig12, ax = plt.subplots(dpi=300)
xs = np.zeros([2, copies])
for i in range(copies):
    xs[0, i] = i
    xs[1, i] = i + 1
# p_accept_T_theory = [2 - np.trapz(np.maximum(p_T[i], p_T[i + 1]), x) for i in range(copies - 1)]
p_accept_T = swaps_T_vs_t[-1, :] / swaps_T_vs_t.shape[0]
f.multiline(T, np.array([p_accept_T, p_accept_T]), xs=xs.T, ax=ax[1])
# f.multiline(T, np.array([p_accept_T_theory, p_accept_T_theory]), xs=xs.T, ax=ax, linestyle='--')
ax[1].set_yscale('log')
ax[1].set_yticks([0.01, 0.1, 1])
ax[1].set_xticks([])
plt.tight_layout()
ax[1].title.set_text('Acceptance probability per T')
if show_figures:
    plt.show()
# %% Print multipage pdf
if save_pdf:
    path = '/home/gabriel/OneDrive/2021/Avaqus/Architecture/Figures/'
    pdf_name = f'Tempering tests {method}, adjacency {adjacency} T0 = {T0}, Tf = {Tf}, copies = {copies}, T_dist = {T_dist}.pdf'
    figs = [fig7, fig8, fig10] #[fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10]
    pdf = matplotlib.backends.backend_pdf.PdfPages(path + pdf_name)
    for fig in figs:
        print(fig.texts)
        pdf.savefig(fig)
    pdf.close()
    plt.close('all')

# %% Temperature distributions
# if T_dists_fig:
#     def polynomial_fit(x, a, b):
#         return (x / a) ** b
#
#     T_dists = ['linear', 'geometric','optimallinear','optimalgeom','optimalhalf']
#     fig, ax = plt.subplots(dpi=300)
#
#     for T_dist in T_dists:
#         if T_dist == 'geometric':
#             T = np.geomspace(T0, Tf, copies)
#         elif T_dist == 'linear':
#             T = np.linspace(T0, Tf, copies)
#         elif T_dist == 'optimallinear':
#             t = time.perf_counter()
#             T = np.linspace(T0, Tf, copies)
#             T, P = mc.optimal_temperature_distribution(T, J, rng, init_steps=init_steps, avg_steps=avg_steps,
#                                                        accept_prob_min=0.2, accept_prob_max=0.6, plot=False)
#             print(t-time.perf_counter(), len(T))
#         elif T_dist == 'optimalgeom':
#             t = time.perf_counter()
#             T = np.geomspace(T0, Tf, copies)
#             T, P = mc.optimal_temperature_distribution(T, J, rng, init_steps=init_steps, avg_steps=avg_steps,
#                                                        accept_prob_min=0.2, accept_prob_max=0.6, plot=False)
#             print(t-time.perf_counter(), len(T))
#         elif T_dist == 'optimalhalf':
#             t = time.perf_counter()
#             T = np.concatenate( (np.geomspace(T0, Tf/2, copies//2, endpoint=False),  np.linspace(Tf/2, Tf, copies//2)) )
#             T, P = mc.optimal_temperature_distribution(T, J, rng, init_steps=init_steps, avg_steps=avg_steps,
#                                                        accept_prob_min=0.2, accept_prob_max=0.6, plot=False)
#             print(t-time.perf_counter(), len(T))
#
#         # elif T_dist = 'constant_entropy':
#
#         # elif T_dist == 'polynomial fit':
#         #     x = np.linspace(0, 1, len(T))
#         #     fit_params, _ = curve_fit(polynomial_fit, x, T - T0)
#         #     T = polynomial_fit(x, fit_params[0], fit_params[1]) + T0
#         #     T = np.delete(T, -1, 0)
#         #     T = np.append(T, Tf)
#
#         ax.plot(np.linspace(0, 1, len(T)), T, marker='.', label=T_dist)
#     ax.set_yscale('log')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
#     if save_pdf:
#         pdf_temps_fig = f'/home/gabriel/OneDrive/2021/Avaqus/Architecture/Figures/Temperature distribution {adjacency}.pdf'
#         plt.savefig(pdf_temps_fig)