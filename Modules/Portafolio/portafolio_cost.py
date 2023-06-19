import numpy as np
from numba import njit
import math


def cost(s, J):
    µ = J[0]
    Q = J[1]
    c = J[2]

    n = (s + 1) // 2

    return n @ µ + np.diag(n @ Q @ n.T) + c


@njit(fastmath=True, boundscheck=False)
def delta_cost_total(µ, Q, s, dE):
    copies = s.shape[0]
    size = s.shape[1]
    for c in range(copies):
        for i in range(size):
            # n = (s[c, :] + 1) // 2
            # ni = n[i] # Old value in binary
            # d = (ni + 1) % 2 - ni # Distance between old and new values in binary
            # dE[c, i] = µ[i] * d + 2 * d * np.sum(Q[i,:]*n) + Q[i, i]
            dE[c, i] = µ[i] * (-s[c, i]) + 2 * (-s[c, i]) * np.sum(Q[i, :] * ((s[c, :] + 1) // 2)) + Q[i, i]


@njit(fastmath=True, boundscheck=False)
def mc_step(beta, µ, Q, s, E, dE, flip_chances, flip_sites):
    size = s.shape[1]
    for c, (pi, i) in enumerate(zip(flip_chances, flip_sites)):
        if beta[c] * dE[c, i] < 0 or pi <= math.exp(-beta[c] * dE[c, i]):
            s[c, i] = -s[c, i]
            E[c] += dE[c, i]

            dE[c, i] = -dE[c, i]
            for j in range(size):
                if j == i :
                    continue
                dE[c, j] -= 2 * (s[c, j]) * Q[j, i] * ((-s[c, i] + 1) // 2)
