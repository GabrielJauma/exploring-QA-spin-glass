import Modules.spin_glass as sg
import numpy as np
import matplotlib.pyplot as plt

# %%

adjacency = 'portafolio'

seed_list =  np.random.SeedSequence(4654245).spawn(20)
size_list = np.linspace(10, 100, 10, dtype=int)
λ_list = [0, 1, 10, 100, 1000]
P0_list = np.logspace(2,3,10)
end_date_list = ['2010-01-01', '2012-01-01', '2014-01-01', '2016-01-01', '2018-01-01']

seed = 5745123
size = 10
λ = 50
λN = 0
P0 = 1e5
end_date = '2016-01-01'

µs_mean, µs_std, µs_max, µs_min, Qs_mean, Qs_std, Qs_min, Qs_max, cs = [], [], [], [], [], [], [], [], []

changing_paramter = size_list

for size in changing_paramter:
    distribution = [λ, λN, P0, end_date]
    µ, Q, c = sg.connectivity_matrix(size, adjacency, distribution, rng=np.random.default_rng(seed))
    µs_mean.append(np.mean(µ))
    µs_std.append(np.std(µ))
    µs_min.append(np.min(µ))
    µs_max.append(np.max(µ))
    Qs_mean.append(np.mean(Q))
    Qs_std.append(np.std(Q))
    Qs_min.append(np.min(Q))
    Qs_max.append(np.max(Q))
    cs.append(c)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))

if changing_paramter == seed_list:
    changing_paramter = np.arange(len(changing_paramter))

ax[0].plot(np.array(changing_paramter), np.array(µs_min), 'b', label='min')
ax[0].plot(np.array(changing_paramter), np.array(µs_max), 'r', label='max')
ax[0].errorbar(np.array(changing_paramter), np.array(µs_mean), yerr=np.array(µs_std), label='mean')
ax[0].legend()
ax[0].set_title('$µ$')

ax[1].plot(np.array(changing_paramter), np.array(Qs_min), 'b', label='min')
ax[1].plot(np.array(changing_paramter), np.array(Qs_max), 'r', label='max')
ax[1].errorbar(np.array(changing_paramter), np.array(Qs_mean), yerr=np.array(Qs_std), label='mean')
ax[1].legend()
ax[1].set_title('$Q$')

plt.show()