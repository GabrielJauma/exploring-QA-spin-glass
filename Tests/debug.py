import numpy as np
import math
import Modules.monte_carlo as mc
import Modules.spin_glass as sg

seed = 7232455  # 152  # 1845
rng = np.random.default_rng(seed)

size = 8 ** 3
J = sg.connectivity_matrix(size, '3D', 'gaussian_EA', rng=rng)

T0 = 0.1
Tf = 2
copies = 12
T = np.geomspace(T0, Tf, copies)
# a = 10
# T = 1 / np.linspace((1 / T0) ** (1 / a), (1 / Tf) ** (1 / a), copies) ** (a)
# T = np.linspace(T0, Tf, copies)
β = 1 / T


_,E ,*_ = mc.mc_evolution(β, J, steps=100000, start=None, rng=rng)

ndx = np.arange(10) #np.array([5,1,2,0,9,8,6,4,3,7])
p = np.random.rand(10)


swapped = []
for j, (Ej, βj, pj) in enumerate(zip(E, β, p)):
    if np.any(np.array(swapped) == j):
        continue
    else:
        for i, (Ei, βi) in enumerate(zip(E, β)):
            if not np.any(np.array(swapped) == i) and i != j:
                r = (βi - βj) * (Ei - Ej)
                if r >= 0 or pj <= math.exp(r):
                    ndx[i], ndx[j] = ndx[j], ndx[i]
                    swapped.append(j)
                    swapped.append(i)
                    break
