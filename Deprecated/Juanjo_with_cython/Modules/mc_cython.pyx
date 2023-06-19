from libc.math cimport exp #, abs
import numpy as np
import scipy.sparse as sp
cimport numpy as cnp
# import math
# cimport cython
# from numpy.typing import ArrayLike
# from typing import Tuple, Optional, Any, Callable

cdef class CSRSparse:

    cdef Py_ssize_t n
    cdef cnp.int32_t[::1] shape
    cdef double[::1] data
    cdef cnp.int32_t[::1] indices
    cdef cnp.int32_t[::1] indptr
    cdef double[::1] diag

    def __init__(self, Q):
        Q = sp.csr_matrix(Q)
        self.shape = np.asarray(Q.shape, np.int32)
        self.data = Q.data.astype(np.double)
        self.indices = Q.indices.astype(np.int32)
        self.indptr = Q.indptr.astype(np.int32)
        self.diag = np.diag(Q.todense()).copy()  # Why .copy() ?

cdef void sparse_qubo_cost_change(CSRSparse self, double[::1] dE,
                                  cnp.int8_t[:,::1] s, cnp.int32_t[::1] changes,
                                  Py_ssize_t n):
    cdef:
        double E
        Py_ssize_t m, c, i0, i1, i, N = self.shape[0]
        double[::1] data = self.data
        cnp.int32_t[::1] indices = self.indices
        cnp.int32_t[::1] indptr = self.indptr
        double[::1] diag = self.diag
    for j in range(n):
        c = changes[j]
        i0 = indptr[c]
        i1 = indptr[c+1]
        E = 0.0
        for i in range(i0,i1):
            E += data[i] * s[j,indices[i]]
        dE[j] = 2.0 * (s[j,c] * E + diag[c])

# def delta_cost(J, dE, s, changes, n):
#     cdef:
#         CSRSparse Q
#     Q = CSRSparse(J)
#     return sparse_qubo_cost_change(Q, dE, s, changes, n)

# cdef void tempering_step(cnp.int32_t[::1] ndx, double[::1] p,
#                          double[::1] beta, double[::1] E, Py_ssize_t n):
#     cdef:
#         double Ei, Ej, r, betai, betaj
#         Py_ssize_t i, j, ndxi, ndxj
#     for j in range(n):
#         Ej = E[j]
#         betaj = beta[j]
#         ndxj = ndx[j]
#         if j:
#             r = (betai - betaj) * (Ei - Ej)
#             if r >= 0 or p[j] <= exp(r):
#                 ndx[i], ndx[j] = ndx[j], ndx[i]
#                 Ej, betaj, ndxi, j = Ei, betai, ndxj, i
#                 continue
#         Ei, betai, ndxi, i = Ej, betaj, ndxj, j
#     ndx[i] = ndxi


cpdef void tempering_step(cnp.int32_t[::1] ndx, const double[::1] p, double[::1] beta,
                          const double[::1] E, Py_ssize_t n):
    """Possibly swap temperatures between states.

    We implement a tempering phase in which we run over all temperatures
    from hot (small beta) to cold (large beta), investigating whether we
    swap states. In practice `ndx` is the order of the temperatures,
    meaning that state beta[ndx[i]] <= beta[ndx[i+1]]. We run over the
    temperatures in order, computing the probabilities that two configurations
    are exchanged."""
    cdef:
        double Ei, Ej, r, betai, betaj
        Py_ssize_t j, ndxi, ndxj
    for j in range(1, n):
        # We compare the states with temperatures betai < betaj
        ndxi = ndx[j-1]
        betai, Ei = beta[ndxi], E[ndxi]
        ndxj = ndx[j]
        betaj, Ej = beta[ndxj], E[ndxj]
        # According to the temperature probability, we may
        # exchange the configurations associated to those
        # temperatures
        r = (betai - betaj) * (Ei - Ej)
        if r >= 0 or p[j] <= exp(r):
            ndx[j-1], ndx[j] = ndxj, ndxi
            beta[ndxi], beta[ndxj] = betaj, betai

cdef cost(Q, s):
    sT = s.T
    return -0.5 * np.sum(sT * (Q @ sT), 0)

# def cost_p(Q, s):
#     return cost(Q,s)

cdef void mc_step(double[::1] p, double[::1] beta, double[::1] E,
                  cnp.int8_t[:,::1] s, double[::1] dE,
                  cnp.int32_t[::1] changes, Py_ssize_t n):
    cdef:
        Py_ssize_t i, ci
        double betaEi
    for i in range(n):
        betaEi = beta[i] * dE[i]
        if betaEi < 0 or p[i] <= exp(-betaEi):
            ci = changes[i]
            s[i,ci] = -s[i,ci]
            E[i] += dE[i]

# def mc_step_p(p, beta, E, s, dE, changes, n):
#     return mc_step(p, beta, E, s, dE, changes, n)

cdef double[::1] random_real(rng: np.random.Generator, copies: int):
    return rng.random(copies)

cdef void mc_loop(cnp.int32_t[::1] beta_order, double[::1] beta, CSRSparse Q,
                  cnp.int8_t[:,::1] s, double[::1] E,
                  int steps, object rng):
    cdef:
        Py_ssize_t m = s.shape[1] # Number of bits in each configuration
        Py_ssize_t n = s.shape[0] # Number of configurations to stochastically sample
        int step # Iteration counter
        double[::1] dE = E.copy() # Arrays of the energies / energy change of those configurations
        # We generate all random numbers we consume
        cnp.int32_t[:,::1] change = rng.integers(0, m, (steps,n), np.int32)
        int random_counter = 0
        double[:,::1] mc_random = rng.random((steps+steps//10, n))
    #
    # Monte Carlo algorithm
    #
    for step in range(steps):
        #
        # Find which bit to change on each instance and compute the cost of flipping it
        sparse_qubo_cost_change(Q, dE, s, change[step,:], n)
        #
        # Accept the change with a probability determined by the
        # instance's temperature and change of energy
        mc_step(mc_random[random_counter,:], beta, E, s, dE, change[step,:], n)
        random_counter += 1
        #
        # If tempering, randomly swap the instances' temperatures
        # based on the tempering probability formula
        # if step % 10 == 0:
        #     tempering_step(beta_order, mc_random[random_counter,:], beta, E, n)
        #     random_counter += 1

# def mc_loop_p(beta_order, beta, J, s, E, steps, rng):
#     cdef:
#         CSRSparse Q
#     Q = CSRSparse(J)
#     return mc_loop(beta_order, beta, Q, s, E, steps, rng)

cdef double isclose(double a, double b):
    cdef:
        double atol = 1e-10
        double rtol = 1e-8
    return abs(b - a) * (rtol * abs(b) + atol)

def mc_evolution(beta, J, steps=None, start=None, eq_points: int = 1000, rng = None):
    cdef:
        int num_steps
        bint thermal_equilibrium = 0
        int eq_points_left = eq_points, eq_points_max = eq_points
        double Emin = np.nan
        # beta is destructively modified, as are s and E
        double[::1] thebeta = beta.copy()
        object order = np.argsort(beta).astype(np.int32)
        cnp.int32_t[::1] beta_order = order
        cnp.int8_t [:,::1] s
        double[::1] E
        CSRSparse Q

    size = np.shape(J)[0]
    copies = len(beta)
    if rng is None:
        rng = np.random.default_rng()

    # Generate the original instances that we are modifying
    if start is not None:
        s = scopy = start[0]
        E = Ecopy = start[1]
    else:
        s = scopy = 2*rng.integers(0, 2, (copies, size), np.int8)-1
        E = Ecopy = cost(J, s)

    thermal_equilibrium = False
    if steps is None:
        num_steps = size*size  # change to int(size**2 * 10)
    else:
        num_steps = steps

    Q = CSRSparse(J)
    while not thermal_equilibrium:
        mc_loop(beta_order, thebeta, Q, s, E, num_steps, rng)
        if eq_points_left <= 0:
            thermal_equilibrium = 1
        else:
            newEmin = np.min(Ecopy)
            if Emin < newEmin or isclose(Emin, newEmin):
                eq_points_left -= 1
            else:
                Emin = newEmin
                eq_points_left = eq_points_max
    return scopy[order,:], Ecopy[order]