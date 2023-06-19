from numba import njit
import optim.hubo.mc
import numpy as np
import math

@njit("i4[:](i4[:], f8[:], f8[:], f8[:])", fastmath=True)
def tempering_step(ndx, p, β, E):
    L = β.size
    Ei, βi, i = E[0], β[0], 0
    for j, (Ej, βj, pj) in enumerate(zip(E, β, p)):
        if j:
            r = (βi - βj) * (Ei - Ej)
            if r >= 0 or pj <= math.exp(r):
                ndx[j-1] = j
                Ej, βj, j = Ei, βi, i
                continue
            else:
                ndx[j-1] = i
        Ei, βi, i = Ej, βj, j
    ndx[-1] = i
    return ndx

@njit("void(f8[:], f8[:], f8[:], u1[:,:], f8[:], u1[:,:])",
      fastmath=True)
def mc_step(p, βE, E, s, newE, news):
    for i, (βEi, pi) in enumerate(zip(βE, p)):
        if βEi < 0 or pi <= math.exp(-βEi):
            s[i,:] = news[i,:]
            E[i] = newE[i]




def mc_HUBO_solver(cost, size, β=None, steps=2000, copies=1,
                   temper=False, replicas=10,
                   trajectories=False, direction=-1, start=None,
                   rng=None):
    """Solve a QUBO problem by classical annealing.

    Input:
    cost =          Energy function to minimize/maximize
    size =          Size of the binary vector that is an argument to the cost function
    β =             Annealing schedule. It can be vector of real nonnegative
                    values of the inverse temperature, to be used one after
                    another. It can be a pair (β0, βf), in which case we
                    use a vector np.linspace(β0, βf, steps). It can
                    also be None, in which case β0=0, βf=10.
    steps =         Number of MC steps to use, when β is not a vector
    copies =        Number of MC trajectories to simulate
    temper =        If True, use parallel tempering over from β[0] to β[-1]
    replicas =      Number of temperatures to use when `temper` is True
    trajectories =  If True, return a matrix of energies explored.
    direction =     If -1, minimize cost, otherwise, maximize it.

    Output:
    smin =       list/vector of integers that minimize the problem
    Emin =       minimum energy that was found
    Ematrix =    matrix of shape (copies, steps) with the energies explored
                 by the algorithm.
    """
    if β is None:
        β = (1e-6, 10.0)
    sign = +1 if direction < 0 else -1
    #
    # If we are tempering, β is a vector of temperatures for
    # each instance; otherwise, β is a vector of temperatures
    # for each annealing step
    if temper:
        if len(β) == 2:
            β = np.linspace(β[0], β[1], replicas)
        β = np.kron(np.ones(copies), β)
        copies = β.size
    else:
        if len(β) == 2:
            β = np.linspace(β[0], β[1], steps)
        steps = β.size
    if rng is None:
        rng = np.random.default_rng()
    #
    # Generate the original instances that we are modifying
    L = size
    if start is None:
        s = rng.integers(0, 2, (copies, L), np.uint8)
    else:
        s = np.array(start, dtype=np.uint8).reshape(1,-1) * np.ones((copies, 1), dtype=np.uint8)
    E = sign*cost(s)
    if trajectories:
        theE = np.zeros((steps+1, copies))
        theE[0] = E
    ndx = np.argmin(E)
    smin = s[ndx,:].copy()
    Emin = E[ndx]
    #
    # Monte Carlo algorithm
    #
    swap = np.arange(copies, dtype=np.int32)
    for n, change in enumerate(rng.integers(0, L, (steps, copies))):
        #
        # Find which bit to change on each instance and flip it
        news = s.copy()
        news[swap, change] = 1 - news[swap, change]
        newE = sign*cost(news)
        #
        # Record the lowest energy instance recorded so far
        ndx = np.argmin(newE)
        if newE[ndx] < Emin:
            Emin = newE[ndx]
            smin[:] = news[ndx,:].copy()
        #
        # Accept the change with a probability determined by the
        # instance's temperature and change of energy
        mc_step(rng.random(copies), (β[n] if not temper else β) * (newE - E),
                E, s, newE, news)
        #
        # If tempering, randomly swap the instances' temperatures
        # based on the tempering probability formula
        if temper:
            swap = tempering_step(swap, rng.random(β.size), β, E)
            E = E[swap]
            s = s[swap, :]
        if trajectories:
            theE[n+1,:] = E
    if trajectories:
        return smin, Emin, theE
    else:
        return smin, Emin
