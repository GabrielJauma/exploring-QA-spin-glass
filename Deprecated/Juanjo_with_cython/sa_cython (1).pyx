from libc.math cimport exp, sqrt, log
import numpy as np
import scipy.sparse as sp
cimport numpy as cnp
import math
cimport cython
from .random cimport RandomInt32Pool, CRNG
from numpy.typing import ArrayLike
from typing import Tuple, Optional, Any, Callable
from .normalize import normalize, QUBOMatrix

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
        self.diag = np.diag(Q.todense()).copy()

cdef void sparse_qubo_cost_change(CSRSparse self, double[:,::1] dE, cnp.uint8_t[:,::1] s, Py_ssize_t n, Py_ssize_t m) nogil:
    cdef:
        double E
        Py_ssize_t c, i0, i1, i
        double[::1] data = self.data
        cnp.int32_t[::1] indices = self.indices
        cnp.int32_t[::1] indptr = self.indptr
        double[::1] diag = self.diag
    for j in range(n):
        for c in range(m):
            i0 = indptr[c]
            i1 = indptr[c+1]
            E = 0.0
            for i in range(i0,i1):
                E += data[i] * s[j,indices[i]]
            dE[j,c] = 2.0 * E * (1 - 2 * s[j,c]) + diag[c]


cdef double sparse_mc_step_full(CSRSparse self, CRNG p, double beta, double[::1] E, cnp.uint8_t[:,::1] s,
                           double[:,::1] dE, Py_ssize_t n, Py_ssize_t m, double Emin, cnp.uint8_t[::1] smin) nogil:
    cdef:
        Py_ssize_t i, ci, r, r0, r1, j
        double[::1] data = self.data
        cnp.int32_t[::1] indices = self.indices
        cnp.int32_t[::1] indptr = self.indptr
        double[::1] diag = self.diag
        int fac
        double dEi
    for i in range(n):
        for ci in range(m):
            dEi = dE[i, ci]
            if dEi < 0 or dEi < -log(p.random())/beta:
                E[i] += dEi
                s[i,ci] = 1-s[i,ci]
                #
                # Update the energy changes of this site and all neighbors
                r0 = indptr[ci]
                r1 = indptr[ci+1]
                fac = 2*(2*s[i,ci]-1)
                for r in range(r0, r1):
                    j = indices[r]
                    dE[i, j] += ((1-2*s[i,j])*fac)*data[r]
                dE[i,ci] = -dEi
                if E[i] < Emin:
                    smin[:] = s[i,:]
                    Emin = E[i]
    return Emin

cdef double sparse_mc_step(CSRSparse self, CRNG p, double beta, double[::1] E, cnp.uint8_t[:,::1] s,
                           double[:,::1] dE, Py_ssize_t n, Py_ssize_t m, double Emin, cnp.uint8_t[::1] smin) nogil:
    cdef:
        Py_ssize_t i, ci, r, r0, r1, j
        double[::1] data = self.data
        cnp.int32_t[::1] indices = self.indices
        cnp.int32_t[::1] indptr = self.indptr
        double[::1] diag = self.diag
        int fac
        double dEi
    for i in range(n):
        ci = p.integer32() % m
        dEi = dE[i, ci]
        if dEi < 0 or dEi < -log(p.random())/beta:
            E[i] += dEi
            s[i,ci] = 1-s[i,ci]
            #
            # Update the energy changes of this site and all neighbors
            r0 = indptr[ci]
            r1 = indptr[ci+1]
            fac = 2*(2*s[i,ci]-1)
            for r in range(r0, r1):
                j = indices[r]
                dE[i, j] += ((1-2*s[i,j])*fac)*data[r]
            dE[i,ci] = -dEi
            if E[i] < Emin:
                smin[:] = s[i,:]
                Emin = E[i]
    return Emin

cdef qubo_cost(Q, s):
    return np.sum(s * (s @ Q), 1)

cdef void qubo_cost_change(double [:,::1] Q, double[:,::1] dE, cnp.uint8_t[:,::1] s, Py_ssize_t n, Py_ssize_t m) nogil:
    cdef:
        double E
        int fac
        Py_ssize_t i, j, c
    for i in range(n):
        for c in range(m):
            E = 0.0
            for j in range(m):
                E += Q[c,j] * s[i,j]
            fac = 1 - 2 * s[i,c]
            dE[i, c] = (2 * E + Q[c,c] * fac) * fac

cdef double mc_step(double[:,::1] Q, CRNG p, double beta, double[::1] E, cnp.uint8_t[:,::1] s,
                    double[:,::1] dE, Py_ssize_t n, Py_ssize_t m, double Emin, cnp.uint8_t[::1] smin) nogil:
    cdef:
        Py_ssize_t i, ci, j
        int fac
        double dEi
    for i in range(n):
        ci = p.integer32() % m
        dEi = dE[i, ci]
        if dEi < 0 or dEi < -log(p.random())/beta:
            E[i] += dEi
            s[i,ci] = 1-s[i,ci]
            #
            # Update the energy changes of this site and all neighbors
            fac = 2*(2*s[i,ci]-1)
            for j in range(m):
                dE[i, j] += ((1-2*s[i,j])*fac)*Q[ci,j]
            dE[i,ci] = -dEi
            if E[i] < Emin:
                smin[:] = s[i,:]
                Emin = E[i]
    return Emin

cdef double mc_step_full(double[:,::1] Q, CRNG p, double beta, double[::1] E, cnp.uint8_t[:,::1] s,
                         double[:,::1] dE, Py_ssize_t n, Py_ssize_t m, double Emin, cnp.uint8_t[::1] smin) nogil:
    cdef:
        Py_ssize_t i, ci, j
        int fac
        double dEi
    for i in range(n):
        for ci in range(m):
            dEi = dE[i, ci]
            if dEi < 0 or dEi < -log(p.random())/beta:
                E[i] += dEi
                s[i,ci] = 1-s[i,ci]
                #
                # Update the energy changes of this site and all neighbors
                fac = 2*(2*s[i,ci]-1)
                for j in range(m):
                    dE[i, j] += ((1-2*s[i,j])*fac)*Q[ci,j]
                dE[i,ci] = -dEi
                if E[i] < Emin:
                    smin[:] = s[i,:]
                    Emin = E[i]
    return Emin

def calibrate_temperature_range(Q):
    # We estimate the minimum and maximum energy changes when changing a
    # variable in our problem.
    if sp.issparse(Q):
        Q = np.asarray(Q.todense())
        abs_J = abs(sp.triu(Q, 1).T + sp.tril(Q, -1)) / 4.0
        abs_h = abs((Q + Q.T).sum(0)) / 4.0
        min_energy_flip = min(
            abs_h[abs_h != 0].min(initial=np.inf),
            abs_J.data[abs_J.data != 0].min(initial=np.inf),
        )
        max_energy_flip = np.max(abs_h + abs_J.sum(0) + abs_J.T.sum(0))
    else:
        abs_J = abs(np.triu(Q, 1).T + np.tril(Q, -1)) / 4
        abs_h = abs(np.sum(Q + Q.T, 0)) / 4
        min_energy_flip = min(
            abs_h[abs_h != 0].min(initial=np.inf), abs_J[abs_J != 0].min(initial=np.inf)
        )
        max_energy_flip = np.max(abs_h + np.sum(abs_J + abs_J.T, 0))
    if min_energy_flip == 0 or min_energy_flip == np.inf:
        return [0.1, 1.0]
    # The temperature range is designed to make the probability of costly flips
    # around 50% initially and smaller at the end.
    return np.log(2.0) / max_energy_flip, np.log(100.0) / min_energy_flip


def mc_QUBO_solver(Q: QUBOMatrix,
                   beta: Optional[ArrayLike] = None,
                   steps: Optional[int] = None, replicas: int = 10, copies: int = 100,
                   start: Optional[ArrayLike] = None,
                   full_update: bool = True,
                   rng: Optional[np.random.Generator] = None
                   ) -> Tuple[np.ndarray, float]:
    """Monte Carlo solver for quadratic binary optimization problems.

    This function implements two Monte Carlo Algorithms for solving quadratic binary
    optimization problems, represented by real (dense or sparse) matrices `Q` of
    dimension `(m,m)`.
    
    If `temper` is False, it implements a simulated annealing where the
    schedule of inverse temperatures is dictated by `beta`. This variable can be a
    vector monotonously increasing real numbers, whose length determines the number
    of Monte Carlo steps. Alternatively, it can be a pair of numbers from which a
    schedule with `steps` phases is derived. During these stochastic phases, a total of
    `n=copies` different configurations are modfied stochastically.

    If `temper` is True, the algorithm uses parallel tempering, simulating the thermalization
    of a number of replicas at different temperatures. If `beta` is a vector of inverse
    temperatures, the number of replicas is derived from it. Alternatively, if `beta`
    is a pair of non-negative numbers, a vector of length = `replicas` is derived. In
    parallel tempering mode, each temperature can have associated more than one configuration
    which is determined by the variable `copies`. A total of `n=copies*replicas` states are
    simultaneously explored.
    
    The initial configurations are determined by the `start` variable, which must contain an
    array-like object with dimensions `(n, m)` or `(m,)`, in which case the
    array is broadcasted to the appropriate size. If `start` is `None`, a completely
    random set of configurations is created.

    This function returns the bit configuration with the lowest cost function. In addition to
    this, the function may be supplied a `callback` function that is called on every
    iteration as `callback(step, configurations)` a total of `steps+1` times.

    Parameters
    ----------
    Q : QUBOMatrix[(m,m)]
        Matrix of size `(m,m)` representing the QUBO problem
    beta : ArrayLike, optional
        Array-like sequence of inverse temperatures, by default None
    steps : int, optional
        Number of simulated annealing steps. Default is calibrated based on problem size.
    replicas : int, optional
        Number of different temperatures in parallel tempering, by default 10
    copies : int, optional
        Number of configurations per temperature in parallel tempering, by default 100
    start : ArrayLike, optional
        Initial configurations for the stochastic algorithm, by default None
    rng : numpy.random.Generator, optional
        Random number generator, by default `np.random.default_rng()`

    Returns
    -------
    s: ndarray[(size,), dtype=np.uint8]
        Sequence of bits with the lowest cost function
    E: float
        Value of the lowest cost function
    """
    if rng is None:
        rng = np.random.default_rng()
    cdef:
        double factor # Factors for automatic calibration of parameters
        bint use_sparse = sp.issparse(Q)
        bint use_full_update = bool(full_update)
        double[:,::1] Qubo # Matrix representation of the QUBO problem
        CSRSparse Qsparse = None
        Py_ssize_t m # Number of bits in each configuration
        Py_ssize_t n # Number of configurations to stochastically sample
        int step, num_steps # Iteration counter
        cnp.uint8_t[:,::1] s # Array of configurations
        double[::1] E # Arrays of the energies
        double[:,::1] dE # Array of energy changes for each spin flip
        double Emin # Lowest energy found so far
        cnp.uint8_t[::1] smin # Configuration for the lowest energy state
        CRNG crng # Random numbers [0,1) for Monte Carlo steps
        double[::1] beta_vector # Vector of temperatures for annealing
    #
    # Normalize problem and find out dimensions
    Q = normalize(Q)
    if use_sparse:
        Qsparse = CSRSparse(Q)
    else:
        Qubo = Q
    m = Q.shape[0]
    # Automatic calibration of thermalization steps. We use the heuristic
    # that L = N * exp(log(2)log(epsilon)sqrt(N)/(-30))
    # where epsilon = 1e-5 is the desired relative error in the energy
    if steps is None:
        steps = min(int(m * exp(0.26 * sqrt(m))), 10000000)
        if full_update:
            steps = steps // m
    if beta is None:
        if False:
            factor = sp.linalg.norm(Q) if use_sparse else np.linalg.norm(Q)
            factor = sqrt(m/128.0) / factor
            beta = (1e-6*factor, 200.0*factor)
        else:
            beta = calibrate_temperature_range(Q)
    if len(beta) == 2:
        beta = np.linspace(beta[0], beta[1], steps)
    steps = beta.size
    n = copies
    beta_vector = beta = np.asarray(beta, dtype=np.double)
    #
    # Generate the original instances that we are modifying
    if start is None:
        start = rng.integers(0, 2, (n, m), np.uint8)
    else:
        start = np.asarray(start, dtype=np.uint8)
        if start.ndim == 1:
            start = start.reshape(1,-1) * np.ones((n, 1), dtype=np.uint8)
        else:
            if start.ndim != 2 or start.shape[1] != m:
                raise Exception('`start` array has an incorrect shape')
            n = copies = start.shape[0]
    #
    s = start
    E = Ecopy = qubo_cost(Q, start)
    smin = output = start[0,:].copy()
    Emin = E[0]
    dE = np.empty((n, m), dtype=np.double)
    beta_vector = beta
    #
    # Generate all random numbers we need
    num_steps = steps
    crng = CRNG(rng)
    #
    # Monte Carlo algorithm
    #
    #with nogil:
    if use_sparse:
        sparse_qubo_cost_change(Qsparse, dE, s, n, m)
    else:
        qubo_cost_change(Qubo, dE, s, n, m)
    for step in range(num_steps):
        if use_full_update:
            if use_sparse:
                Emin = sparse_mc_step_full(Qsparse, crng, beta_vector[step], E, s, dE, n, m, Emin, smin)
            else:
                Emin = mc_step_full(Qubo, crng, beta_vector[step], E, s, dE, n, m, Emin, smin)
        else:
            if use_sparse:
                Emin = sparse_mc_step(Qsparse, crng, beta_vector[step], E, s, dE, n, m, Emin, smin)
            else:
                Emin = mc_step(Qubo, crng, beta_vector[step], E, s, dE, n, m, Emin, smin)
    return output, Emin
