from libc.math cimport exp, log
import numpy as np
import scipy.sparse as sp
cimport numpy as cnp
from Modules.random cimport MersenneTwister
from typing import Optional


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


cdef cost(Q, s):
    sT = s.T
    return -0.5 * np.sum(sT * (Q @ sT), 0)


cdef void sparse_qubo_cost_change(CSRSparse self, double[::1] dE,
                                  cnp.int8_t[:,::1] s, cnp.int32_t[::1] changes,
                                  Py_ssize_t n) nogil:
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


# Cython 1
# cdef void tempering_step(cnp.int32_t[::1] ndx, MersenneTwister crng, double[::1] beta,
#                           const double[::1] E, Py_ssize_t n):
#     """Possibly swap temperatures between states.
#
#     We implement a tempering phase in which we run over all temperatures
#     from hot (small beta) to cold (large beta), investigating whether we
#     swap states. In practice `ndx` is the order of the temperatures,
#     meaning that state beta[ndx[i]] <= beta[ndx[i+1]]. We run over the
#     temperatures in order, computing the probabilities that two configurations
#     are exchanged."""
#     cdef:
#         double Ei, Ej, r, betai, betaj
#         Py_ssize_t j, ndxi, ndxj
#     for j in range(1, n):
#         # We compare the states with temperatures betai < betaj
#         ndxi = ndx[j-1]
#         betai, Ei = beta[ndxi], E[ndxi]
#         ndxj = ndx[j]
#         betaj, Ej = beta[ndxj], E[ndxj]
#         # According to the temperature probability, we may
#         # exchange the configurations associated to those
#         # temperatures
#         r = (betai - betaj) * (Ei - Ej)
#         if r >= 0 or crng.random() <= exp(r):
#             ndx[j-1], ndx[j] = ndxj, ndxi
#             beta[ndxi], beta[ndxj] = betaj, betai


# Cython 2
cdef void tempering_step(cnp.int32_t[::1] ndx, MersenneTwister crng, double[::1] beta,
                          const double[::1] E, Py_ssize_t n) nogil:
    """
    Possibly swap temperatures between states.

    We implement a tempering phase in which we run over all temperatures
    from hot (small beta) to cold (large beta), investigating whether we
    swap states. In practice `ndx` is the order of the temperatures,
    meaning that state beta[ndx[i]] <= beta[ndx[i+1]]. We run over the
    temperatures in order, computing the probabilities that two configurations
    are exchanged.
    """
    cdef Py_ssize_t j
    for j in range(1, n, 2):
        maybe_temper(ndx, beta, E, crng.random(), j)
    for j in range(2, n, 2):
        maybe_temper(ndx, beta, E, crng.random(), j)


cdef inline void maybe_temper(cnp.int32_t[::1] ndx, double[::1] beta, const double[::1] E,
                              double pj, Py_ssize_t j) nogil:
    cdef:
        double Ei, Ej, r, betai, betaj
        Py_ssize_t ndxi, ndxj
    # We compare the states with temperatures betai < betaj
    ndxi = ndx[j-1]
    betai, Ei = beta[ndxi], E[ndxi]
    ndxj = ndx[j]
    betaj, Ej = beta[ndxj], E[ndxj]
    # According to the temperature probability, we may
    # exchange the configurations associated to those
    # temperatures
    r = (betai - betaj) * (Ei - Ej)
    if r >= 0 or pj <= exp(r):
        ndx[j-1], ndx[j] = ndxj, ndxi
        beta[ndxi], beta[ndxj] = betaj, betai


cdef void mc_step(MersenneTwister crng, double[::1] beta, double[::1] E, cnp.int8_t[:,::1] s,
                  double[::1] dE, cnp.int32_t[::1] changes, Py_ssize_t n) nogil:
    cdef:
        Py_ssize_t i, ci
        double dEi
    for i in range(n):
        dEi = dE[i]
        if dEi < 0 or dEi < -log(crng.random())/beta[i]:
            E[i] += dEi
            ci = changes[i]
            s[i,ci] = -s[i,ci]


cdef void mc_loop(cnp.int32_t[::1] beta_order, double[::1] beta, CSRSparse Q,
                  cnp.int8_t[:,::1] s, double[::1] E, int steps, int steps_until_temp, MersenneTwister crng):
    cdef:
        Py_ssize_t m = s.shape[1] # Number of bits in each configuration
        Py_ssize_t n = s.shape[0] # Number of configurations to stochastically sample
        int step # Iteration counter
        double[::1] dE = E.copy() # Arrays of the energies / energy change of those configurations
        cnp.int32_t[::1] changes

    # Monte Carlo algorithm
    changes = np.empty((n,), dtype=np.int32)
    for step in range(steps):

        crng.integers32(0,m,changes,n)
        sparse_qubo_cost_change(Q, dE, s, changes, n)
        mc_step(crng, beta, E, s, dE, changes, n)

        if step % steps_until_temp == 0 :
            tempering_step(beta_order, crng, beta, E, n)


def mc_evolution(beta, J, steps=None, start=None, steps_until_temp=10, rng=np.random.default_rng()):
    cdef:
        int steps_c, steps_until_temp_c
        double Emin = np.nan
        # beta is destructively modified, as are s and E
        double[::1] thebeta = beta.copy()
        object order = np.argsort(beta).astype(np.int32)
        cnp.int32_t[::1] beta_order = order
        cnp.int8_t [:,::1] s
        double[::1] E
        CSRSparse Q
        MersenneTwister crng = MersenneTwister(rng)
        double[:, ::1] E_loop
        double[:, ::1] beta_loop

    size = J.shape[0]
    copies = beta.size

    if start is not None:
        s = scopy = start[0]
        E = Ecopy = start[1]
    else:
        s = scopy = 2*rng.integers(0, 2, (copies, size), np.int8)-1
        E = Ecopy = cost(J, s)

    if steps is None:
        steps = int(size * 1e5)

    if steps * copies > 4e10:  # Hardcoded value to ensure that the random arrays (random_sites, ...) fit in Paulas PC.
        blocks = int(steps * copies / 4e10)
        steps = int(steps / blocks)
    else:
        blocks = 1

    Q = CSRSparse(J)
    steps_c = steps
    steps_until_temp_c = steps_until_temp
    for _ in range(blocks):
        mc_loop(beta_order, thebeta, Q, s, E, steps_c, steps_until_temp_c, crng)

    return scopy[order,:], Ecopy[order]


#
# cdef void mc_loop_trajectories(cnp.int32_t[::1] beta_order, double[::1] beta, CSRSparse Q, cnp.int8_t[:,::1] s,
#                                double[::1] E, int steps, int steps_until_temp, MersenneTwister crng,
#                                double[:,::1] E_loop, double[:,::1] beta_loop ):
#     cdef:
#         Py_ssize_t m = s.shape[1] # Number of bits in each configuration
#         Py_ssize_t n = s.shape[0] # Number of configurations to stochastically sample
#         Py_ssize_t temper_index
#         int step # Iteration counter
#         double[::1] dE = E.copy() # Arrays of the energies / energy change of those configurations
#         cnp.int32_t[::1] changes
#
#     # Monte Carlo algorithm
#     changes = np.empty((n,), dtype=np.int32)
#     for step in range(steps):
#
#         crng.integers32(0,m,changes,n)
#         sparse_qubo_cost_change(Q, dE, s, changes, n)
#         mc_step(crng, beta, E, s, dE, changes, n)
#
#         if step % steps_until_temp == 0:
#             temper_index = int(step/steps_until_temp)
#             tempering_step(beta_order, crng, beta, E, n)
#             beta_loop[temper_index, :] = beta
#             # beta_loop[temper_index, :] = beta_order
#
#         E_loop[step, :] = E
#         # beta_loop[step, :] = beta_order


# def mc_evolution_test(beta, J, steps=None, start=None, steps_until_temp: int = 10, rng: Optional[np.random.Generator] = None, trajectories: bool = False):
#     cdef:
#         int num_steps
#         double Emin = np.nan
#         # beta is destructively modified, as are s and E
#         double[::1] thebeta = beta.copy()
#         object order = np.argsort(beta).astype(np.int32)
#         cnp.int32_t[::1] beta_order = order
#         cnp.int8_t [:,::1] s
#         double[::1] E
#         CSRSparse Q
#         MersenneTwister crng = MersenneTwister(rng)
#         double[:, ::1] E_loop
#         double[:, ::1] beta_loop
#
#     size = np.shape(J)[0]
#     copies = len(beta)
#     if rng is None:
#         rng = np.random.default_rng()
#
#     # Generate the original instances that we are modifying
#     if start is not None:
#         s = scopy = start[0]
#         E = Ecopy = start[1]
#     else:
#         s = scopy = 2*rng.integers(0, 2, (copies, size), np.int8)-1
#         E = Ecopy = cost(J, s)
#
#     if steps is None:
#         steps = int(size * 1e5)
#
#     if steps * copies > 4e10:  # Hardcoded value to ensure that the random arrays (random_sites, ...) fit in Paulas PC.
#         blocks = int(steps * copies / 4e10)
#         steps = int(steps / blocks)
#     else:
#         blocks = 1
#
#     if trajectories:
#         E_vs_t = np.array([E])
#         beta_vs_t = np.array([beta])
#
#
#     Q = CSRSparse(J)
#     for _ in range(blocks):
#         if trajectories:
#             E_loop = np.zeros([num_steps, copies])
#             beta_loop = np.zeros([int(num_steps/10), copies])
#
#             mc_loop_trajectories(beta_order, thebeta, Q, s, E, num_steps, crng, E_loop, beta_loop)
#
#             E_vs_t = np.concatenate((E_vs_t, E_loop))
#             beta_vs_t = np.concatenate((beta_vs_t, beta_loop))
#
#     if trajectories:
#         return scopy[order,:], Ecopy[order], E_vs_t, 1/beta_vs_t
#
#
# def delta_cost(J, dE, s, changes, n):
#     cdef:
#         CSRSparse Q
#     Q = CSRSparse(J)
#     return sparse_qubo_cost_change(Q, dE, s, changes, n)
#
# def cost_p(Q, s):
#     return cost(Q,s)
#
# def mc_step_p(rng, beta, E, s, dE, changes, n):
#     cdef:
#         CSRSparse Q
#         MersenneTwister crng = MersenneTwister(rng)
#     return mc_step(crng, beta, E, s, dE, changes, n)
#
# def mc_loop_p(beta_order, beta, J, s, E, steps, rng):
#     cdef:
#         CSRSparse Q
#         MersenneTwister crng = MersenneTwister(rng)
#     Q = CSRSparse(J)
#     return mc_loop(beta_order, beta, Q, s, E, steps, crng)
