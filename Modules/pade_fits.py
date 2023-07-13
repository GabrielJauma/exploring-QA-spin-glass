import numpy as np
from scipy.optimize import curve_fit, minimize_scalar, root_scalar
import itertools
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import scipy.stats as stats
from joblib import Parallel, delayed

rc('text', usetex=True)

color = ['turquoise', 'tab:olive', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
         'tab:blue', 'goldenrod', 'tab:orange', 'tab:red']

''''
# Fitting function:

Parameters:

rho  --> order of expansion in relevant field

ns   --> order of expansion in the function itself

nb   --> order of expansion in irrelevant field

We look for a scaling form in variable $f$ in Taylor expansion:

$$f\left(L^{1/\nu}\rho(W), L^{-|y|}u_i(W)\right)$$

Subroutine fitea_p is the main one to fit data. Inputs are:

xopt   --> [temperature, sizes]

yopt   --> Data to scale

y_error   --> errors in data

Outputs:

popt  --> Best fitting parameters

perr  --> errors in best fitting parameters

chisq --> chisq divided between degrees of freedom. Good fitting should give around 1

res   --> residuos. '''


def derivative(f, dx):
    """
    Input;
    f, dx ---> function to derive and finite difference
    Outputs:
    x_m ---> derivative via syummetryc first order formula.
    """
    df = lambda x: (f(x + dx / 2) - f(x - dx / 2)) / dx
    return df


class fit_pade:
    """
    Class to perform a pade fitting.
    Input;
    n,m ---> orders of the Padé
    """
    def __init__(self, n, m):
        self.n = n
        self.m = m

    #
    def    evalua_pade(self, x, *para):
        """
        Input;
        x     --->    value at which to evaluate
        *para ---> parameters of the padé, they depends on n, m variable of the class
        Outputs:
        x     --->  value of the padé function at x
        """
        a = para[0:self.n]
        b = para[self.n:]
        y = sum(a[i] * x ** i for i in range(len(a))) / (1 + sum(b[i] * x ** i for i in range(len(b))))

        return y

    #
    def func_pade(self, *para):
        """
        Input;
        *para ---> parameters of the padé, they depends on n, m variable of the class.
        Outputs:
        func     --->  the padé approximation as a labmda function
        """
        a = para[0:self.n]
        b = para[self.n:]
        func = lambda x: sum(a[i] * x ** i for i in range(len(a))) / (1 + sum(b[i] * x ** i for i in range(len(b))))
        return func

    def fitea_p(self, x, y, y_err, igu, verbose=False, maxfev=5000000):
        """
        Input;
        x, y, yerr ---> data to fit.
        Outputs:
        igu     --->  parameters of the fitting
        """
        try:
            popt, pcov = curve_fit(self.evalua_pade, x, y, igu, sigma=y_err, maxfev=maxfev)
        except:
            return [None], [None], 10, [None]
        perr = np.sqrt(np.abs(np.diag(pcov)))
        r = y - self.evalua_pade(x, *popt)
        res = r / y_err
        chisq = sum((res) ** 2) / (y_err.shape[0] - len(igu))
        if verbose: print("chisq/df", chisq)
        return popt, perr, chisq, res

    def pade_opt_rand(self, x, y, y_err, ntr=10, verbose=False, maxfev=5000000):
        """
        Fitea via Pade for many random initial parameters, so that the variance is extracted from all those
        fits in a similar fashion as for bootstrap.
        Input;
        x, y, yerr ---> data to fit.
        Outputs:
        igu     --->  parameters of the fitting
        """
        # Return opt_parameter, errors, reduced_chis, res
        igu = [np.random.rand(self.n + self.m) for i in range(ntr)]
        res = [self.fitea_p(x, y, y_err, p0, maxfev=maxfev) for p0 in igu]
        xchi = np.array([res[i][2] for i in range(ntr)])
        ic = (np.abs(xchi)).argmin()
        mu = np.sum(xchi) / ntr
        var = np.sqrt(np.sum(xchi ** 2) / ntr - mu ** 2)
        if verbose:
            print("best estimation has order", self.n, self.m, "  and rchi", xchi[ic])
        return res[ic]

    def sev_func_pade(self, x, y, y_err, ntr=100, verbose=False, maxfev=5000000):
        """
        Fitea via Pade for all curves
        Input;
        x, y, yerr ---> data to fit.
        Outputs:
        igu     --->  parameters of the fitting
        """
        if type(x) == list:
            n = len(x)
            yv = [self.pade_opt_rand(x[i], y[i], y_err[i], ntr=ntr, verbose=verbose, maxfev=maxfev) for i in range(n)]
            para = [yv[i][0] for i in range(n)]
            rchi = [yv[i][2] for i in range(n)]
            func = [self.func_pade(*para[i]) for i in range(len(yv))]
        else:
            yv = self.pade_opt_rand(x, y, y_err, ntr=ntr, maxfev=maxfev)
            para = yv[0]
            rchi = yv[2]
            func = self.func_pade(*para)
        return func, para, rchi

    def der_pade(self, para):
        a = para[:self.n]
        b = para[self.n:]
        # print(para)
        Af = lambda x: sum(a[i] * x ** i for i in range(len(a)))
        Bf = lambda x: (1 + sum(b[i] * x ** i for i in range(len(b))))
        Ap = lambda x: sum(a[i] * i * x ** (i - 1) for i in range(1, len(a)))
        Bp = lambda x: (sum(b[i] * i * x ** (i - 1) for i in range(1, len(b))))
        y = lambda x: Ap(x) / Bf(x) - (Bp(x) * Af(x)) / (Bf(x)) ** 2
        return y

    def der2_pade(self, para):
        a = para[:self.n]
        b = para[self.n:]
        # print(para)
        Af = lambda x: sum(a[i] * x ** i for i in range(len(a)))
        Bf = lambda x: (1 + sum(b[i] * x ** i for i in range(len(b))))
        Ap = lambda x: sum(a[i] * i * x ** (i - 1) for i in range(1, len(a)))
        Bp = lambda x: (sum(b[i] * i * x ** (i - 1) for i in range(1, len(b))))
        App = lambda x: sum(a[i] * i * (i - 1) * x ** (i - 2) for i in range(2, len(a)))
        Bpp = lambda x: (sum(b[i] * i * (i - 1) * x ** (i - 2) for i in range(2, len(b))))
        y = lambda x: App(x) / Bf(x) - 2 * Ap(x) * Bp(x) / (Bf(x)) ** 2 + 2 * Af(x) * (Bp(x)) ** 2 / (Bf(x)) ** 3 - \
                      Af(x) * Bpp(x) / (Bf(x)) ** 2
        return y


# from scipy.special import gamma, factorial
class fitea_order_man:
    def __init__(self, nrho, ns, nb):
        self.nrho = nrho
        self.ns = ns
        self.nb = nb
        self.nt = ns + nrho + nb
        if ns < 2:
            print("WARNING: the minimum ns should be 2 to include scaling variable!")

    """
    Inputs:
    x       ---> points where to evaluate function, like temperature and size
    para    ---> parameters of the function to minimize. We will always assume that the last two parameters are critical ex
                 ponent and disorder
    Outputs:
    f       ---> function evaluated at the points x. 
    """

    def func2(self, x, *para):
        a = para[0:self.ns]
        d = para[self.ns:self.ns + self.nrho]
        b = para[self.ns + self.nrho: self.ns + self.nrho + self.nb]
        nu = para[self.ns + self.nrho + self.nb]
        wc = para[self.ns + self.nrho + self.nb + 1]
        #         print(a,d,nu,wc)
        rho = (x[0] - wc) + sum(d[i] * (x[0] - wc) ** (i + 2) for i in range(self.nrho))
        xs = rho * x[1] ** (1 / nu)
        beta = sum(b[i] * (x[0] - wc) ** (i) for i in range(self.nb))
        # First, without irrelevant corrections
        if self.nb == 0:
            f = sum(a[i] * xs ** i for i in range(self.ns))
        # now with irrelevant corrections in simple form:
        elif self.nb > 0:
            yi = para[-1]
            # f =  f0(xs) + (1+ui * L^{-y})
            f = x[1] ** -1 * sum(a[i] * xs ** i for i in range(self.ns)) * (1 + beta * x[1] ** (-yi))
        return f

    """
    Inputs:
    x_opt  ---> x parameter of fitting function. That typically includes parameter that drive the 
                    transition and size of the system
    y_opt  ---> value of order parameter
    y_err  ---> errors in order parameter 
    Ouputs:
    popt, perr  ----> optimized values of parameters togehter with covarianze matrix
    chisq       ----> value of reduced chisq
    res         ----> residues
    """

    def fitea_p(self, x_opt, y_opt, y_err, nu_i=1, w_i=1, y_i=1, verbose=True):
        igu = [np.random.rand() / 1 for i in range((self.nrho + self.ns + self.nb))]
        #         print(igu)
        if self.nb == 0:
            igu.extend([nu_i, w_i])
        else:
            igu.extend([nu_i, w_i, y_i])
        popt, pcov = curve_fit(self.func2, x_opt, y_opt, igu, sigma=y_err, absolute_sigma="False", maxfev=1000000, check_finite=True)
        perr = np.sqrt(np.diag(pcov))
        r = y_opt - self.func2(x_opt, *popt)
        res = r / y_err
        df = y_err.shape[0] - len(igu)
        chisq = sum((res) ** 2)
        print("numero de puntos", y_err.shape[0])
        print("grados de libertad, ", df)
        fit_T_max = stats.chi2.cdf(chisq, df)
        print("value of 1-cdf", 1 - fit_T_max, "actual value", chisq * df)
        print("chisq/df", chisq / df)
        if verbose:
            for i in range(len(igu)):
                print(popt[i], "+-", perr[i])
        return popt, perr, chisq, res

def pade_best(x, y, y_err, ntr=10, ic=[4, 5], jc=[2, 3], verbose=True, maxfev=10000):
    ppade = []
    fpade = []
    dfpade = []
    st_chi = []
    st_ce = []
    od = []
    inx = itertools.product(ic, jc)
    for order in inx:
        A = fit_pade(*order)
        f_sev, pade_c, rchi = A.sev_func_pade(x[:], y[:], y_err[:], ntr=ntr, verbose=verbose, maxfev=maxfev)
        if type(x) == list:
            df = [A.der_pade(pade_c[i]) for i in range(len(pade_c))]
        else:
            df = A.der_pade(pade_c)
        ppade.append(pade_c)
        od.append(order)
        st_chi.append(rchi)
        fpade.append(f_sev)
        dfpade.append(df)
    st_chi = np.array(st_chi)
    if type(x) == list:
        inopt = [(np.abs(np.array(st_chi[:, i])-1)).argmin() for i in range(len(x))] # choose the pade with st_chi closer to 1
        fpade_out = [fpade[inopt[i]][i] for i in range(len(x))]
        dfpade_out = [dfpade[inopt[i]][i] for i in range(len(x))]
        if verbose:
            print([st_chi[inopt[i], i] for i in range(len(x))])
    else:
        inopt = (np.abs(np.array(st_chi)-1)).argmin()
        fpade_out = fpade[inopt]
        dfpade_out = dfpade[inopt]
        if verbose:
            print("best rchi", st_chi[inopt])

    return fpade_out, dfpade_out, st_chi[inopt], inopt


def pade_best_specific_ic_jc(x_vs_size, y_vs_size, y_err_vs_size, ntr, ic_vs_size, jc_vs_size, verbose=True, maxfev=10000):
    fpade = []
    dfpade = []
    st_chi = []

    for x, y, y_err, ic, jc in zip(x_vs_size, y_vs_size, y_err_vs_size, ic_vs_size, jc_vs_size):
        A = fit_pade(ic, jc)
        f_sev, pade_c, rchi = A.sev_func_pade(x, y, y_err, ntr=ntr, verbose=verbose, maxfev=maxfev)

        df = A.der_pade(pade_c)

        st_chi.append(rchi)
        fpade.append(f_sev)
        dfpade.append(df)
    st_chi = np.array(st_chi)

    return fpade, dfpade, st_chi, None


def pade_best_fast_different_ic_jc_each(x, y, y_err, ic=[5], jc=[6], ntr=10, maxfev=10000):
    pades = []
    for X, Y, Y_err, IC, JC in zip(x, y, y_err, ic, jc):
        A = fit_pade(IC, JC)
        pade, _, _ = A.sev_func_pade(X, Y, Y_err, ntr=ntr, verbose=False, maxfev=maxfev)
        pades.append( pade)

    return pades


def pade_fss(sizes,  T_vs_size_best, dg_dT_vs_size_best, error_dg_dT_vs_size_best, T_term_vs_size=False,
             ntr=10, ic=[5], jc=[6],method_ic_jc='best'):

    """
    This function applies the Pade approximant for finite-size scaling (FSS) and determines the peak position (Tc) and
    peak height of the system for a set of given sizes.

    Parameters:
    sizes (list or numpy.ndarray): Array-like object containing the sizes of the system.
    T_vs_size_best (list): List of temperatures for different system sizes at the best fit.
    dg_dT_vs_size_best (list): List of derivatives of the binder cumulant with respect to temperature for
                               different system sizes at the best fit.
    error_dg_dT_vs_size_best (list): List of uncertainties in the derivatives of the binder cumulant for
                                     different system sizes at the best fit.
    T_term_vs_size (bool or list, optional): List of terminal temperatures for different system sizes.
                                             If False, terminal temperature indices default to zero for all sizes.
                                             Default is False.
    ntr (int, optional): Number of tries used in the scipy optimize. Default is 10.
    ic (list, optional): Degrees of the numerator polynomial in the Pade approximant. Default is [5].
    jc (list, optional): Degrees of the denominator polynomial in the Pade approximant. Default is [6].
    method_ic_jc (str, optional): Method to select the degrees of the numerator and denominator polynomials.
                                  Can be 'best' or 'specific'. Default is 'best'.
                                  It will try from all the posible numerators and denominators and return the one
                                  that gives a chi square closer to 1

    Returns:
    tuple: Returns a tuple containing the Pade approximants, estimated critical temperatures and peak heights
           for all system sizes.
    """

    sizes = np.array(sizes)

    if T_term_vs_size:
        T_term_ind = [np.where(T[-1] > T_term)[0][0] for T_term, T in zip(T_term_vs_size, T_vs_size_best)]
    else:
        T_term_ind = [0] * len(sizes)

    Tfit = [T[k:] for T, k in zip(T_vs_size_best, T_term_ind)]
    Ofit = [B[k:] for B, k in zip(dg_dT_vs_size_best, T_term_ind)]
    Ofit_er = [err[k:] for err, k in zip(error_dg_dT_vs_size_best, T_term_ind)]

    T0 = Tfit[0][0]
    Tf = Tfit[0][-1]

    if method_ic_jc == 'best':
        dg_dT_pade, _, l_rchi, inopt = pade_best(Tfit, Ofit, np.array(Ofit_er), ntr*2, ic, jc)
    elif method_ic_jc == 'specific':
        dg_dT_pade, _, l_rchi, inopt = pade_best_specific_ic_jc(Tfit, Ofit, np.array(Ofit_er), ntr, ic, jc)

    # Data to extrapolate the maxima
    T_c = estimate_Tc_with_pade(sizes, T0, Tf, dg_dT_pade, return_T_max=True)

    peak_height = np.array([dg_dT_pade[i](T_c[i]) for i in range(len(sizes))])

    print(f'Tc = {np.polyfit(1 / sizes[-2:] ** (1 / 3), T_c[-2:], 1)[-1]}')
    print(f'Tc = {np.polyfit(1 / sizes ** (1 / 3), T_c, 2)[-1]}')

    return dg_dT_pade, T_c, peak_height

#%% Estimate Tc from pade approximant of dgdT
def estimate_Tc_with_pade(sizes, T0, Tf, dg_dT_pade, return_T_max=False):
    T_max = np.zeros(len(sizes))
    sizes_fit = np.array(sizes)
    for i in range(len(sizes)):
        T_max[i] = minimize_scalar(dg_dT_pade[i], method='brent', bracket=(T0 + 0.2, T0)).x
        if T_max[i] < T0 + 0.1 or T_max[i] > Tf - 0.5:
            T_max[i] = minimize_scalar(dg_dT_pade[i], method='bounded', bounds=(T0 + 0.2, Tf - 0.5)).x

    linear_fit = np.polyfit(1 / sizes_fit[-3:] ** (1 / 3), T_max[-3:], 1)

    if return_T_max:
        return T_max
    else:
        return linear_fit[-1]

def estimate_peak_width(sizes, T0, Tf, dg_dT_pade, Tc, peak_height):
    add_float = lambda f, float_value: lambda x: f(x) + float_value
    width_L = np.array(
        [root_scalar(add_float(dg_dT_pade[i], -peak_height[i] * 0.8), method='brentq',
                     bracket=(T0, Tc[i])).root
         for i in range(len(sizes))])
    width_R = np.array(
        [root_scalar(add_float(dg_dT_pade[i], -peak_height[i] * 0.8), method='brentq',
                     bracket=(Tc[i], Tf)).root
         for i in range(len(sizes))])
    peak_width_bootstrap = width_R - width_L
    return peak_width_bootstrap

# %%
def estimate_Tc_with_pade_bootstrap(sizes, T_vs_size_best, error_dg_dT_vs_size_best, dg_dT_bootstrap_vs_size_best, ic=[5], jc=[6],
                                    ntr=10, maxfev=10000):

    T0 = T_vs_size_best[0][0]
    Tf = T_vs_size_best[0][-1]

    n_bootstrap = len(dg_dT_bootstrap_vs_size_best[0])
    Tc_bootstrap = np.zeros([n_bootstrap, len(sizes)])
    inv_peak_height_bootstrap = np.zeros([n_bootstrap, len(sizes)])
    peak_width_bootstrap = np.zeros([n_bootstrap, len(sizes)])

    for i_b in range(n_bootstrap):
        dgdT_bootstrap = [dgdT[i_b]for dgdT in dg_dT_bootstrap_vs_size_best]

        try:
            dg_dT_pade = pade_best_fast_different_ic_jc_each(T_vs_size_best, dgdT_bootstrap, error_dg_dT_vs_size_best, ic=ic, jc=jc, ntr=ntr, maxfev=maxfev)
            Tc_bootstrap[i_b, :] = estimate_Tc_with_pade(sizes, T0, Tf, dg_dT_pade, return_T_max=True)
            inv_peak_height_bootstrap[i_b, :] = 1/np.array([dg_dT_pade[i](Tc_bootstrap[i_b, i]) for i in range(len(sizes))])
            peak_width_bootstrap[i_b, :] = estimate_peak_width(sizes, T0, Tf,dg_dT_pade, Tc_bootstrap[i_b, :], 1/inv_peak_height_bootstrap[i_b, :])

        except:
            Tc_bootstrap[i_b, :] = np.nan*np.ones(len(sizes))
            inv_peak_height_bootstrap[i_b, :] = np.nan*np.ones(len(sizes))
            peak_width_bootstrap[i_b, :] = np.nan*np.ones(len(sizes))


    return Tc_bootstrap, inv_peak_height_bootstrap, peak_width_bootstrap

# %%
def estimate_Tc_with_pade_bootstrap_parallel(sizes, T_vs_size, error_vs_size, dg_dT_bootstrap_vs_size, ic=[5], jc=[6],
                                             ntr=10, maxfev=10000, threads=8):
    """
     This function estimates the critical temperature (Tc) with the Pade approximant, utilizing a bootstrap method
     for error estimation. It determines the inverse of the peak height and the peak width for each bootstrap sample.

     Parameters:
     sizes (list or numpy.ndarray): Array-like object containing the sizes of the system.
     T_vs_size_best (list): List of temperatures for different system sizes at the best fit.
     error_dg_dT_vs_size_best (list): List of uncertainties in the derivatives of the binder cumulant for
                                      different system sizes at the best fit.
     dg_dT_bootstrap_vs_size_best (list): List of derivatives of the binder cumulant with respect to temperature for
                                          different system sizes at the best fit from bootstrap sampling.
     ic (list, optional): Degrees of the numerator polynomial in the Pade approximant. Default is [5].
     jc (list, optional): Degrees of the denominator polynomial in the Pade approximant. Default is [6].
     ntr (int, optional): Number of temperatures used in the Pade approximant. Default is 10.
     maxfev (int, optional): Maximum number of function evaluations for the Pade approximant. Default is 10000.

     Returns:
     tuple: Returns a tuple containing the critical temperatures, inverse peak heights and peak widths
            for all bootstrap samples.
     """
    n_bootstrap = len(dg_dT_bootstrap_vs_size[0])
    n_bootstrap_per_thread = n_bootstrap // threads
    dg_dT_bootstrap_vs_size_vs_size_per_thread = [
        [dg_dT[n_bootstrap_per_thread * i_b:(i_b + 1) * n_bootstrap_per_thread] for dg_dT in dg_dT_bootstrap_vs_size]
        for i_b in range(threads)]

    Tc_height_widht_bootstrap = Parallel(n_jobs=threads)(delayed(estimate_Tc_with_pade_bootstrap)
                                                  (sizes, T_vs_size, error_vs_size,
                                                   dg_dT_bootstrap_vs_size_vs_size_per_thread[i],
                                                   ic, jc, ntr, maxfev)
                                                  for i in range(threads))
    Tc_bootstrap = []
    inv_peak_height_bootstrap = []
    peak_width_bootstrap = []

    for Tc_height_widht in Tc_height_widht_bootstrap:
        Tc_bootstrap.append(Tc_height_widht[0])
        inv_peak_height_bootstrap.append(Tc_height_widht[1])
        peak_width_bootstrap.append(Tc_height_widht[2])


    Tc_bootstrap = np.array(Tc_bootstrap).reshape((-1, len(sizes)))
    inv_peak_height_bootstrap = np.array(inv_peak_height_bootstrap).reshape((-1, len(sizes)))
    peak_width_bootstrap = np.array(peak_width_bootstrap).reshape((-1, len(sizes)))

    # Remove outliers
    for Tc_bootstrap_vs_size in Tc_bootstrap.T:
        Tc_bootstrap_vs_size[np.abs(Tc_bootstrap_vs_size - np.nanmean(Tc_bootstrap_vs_size)) > 4 * np.nanstd(Tc_bootstrap_vs_size)] = np.nan

    for inv_peak_height_bootstrap_vs_size in inv_peak_height_bootstrap.T:
        inv_peak_height_bootstrap_vs_size[np.abs(inv_peak_height_bootstrap_vs_size - np.nanmean(inv_peak_height_bootstrap_vs_size)) > 4 * np.nanstd(inv_peak_height_bootstrap_vs_size)] = np.nan

    for peak_width_bootstrap_size in peak_width_bootstrap.T:
        peak_width_bootstrap_size[np.abs(peak_width_bootstrap_size - np.nanmean(peak_width_bootstrap_size)) > 4 * np.nanstd(peak_width_bootstrap_size)] = np.nan

    Tc = np.nanmean(Tc_bootstrap, 0)
    Tc_err = 2 * np.nanstd(Tc_bootstrap, 0)

    inv_peak_height = np.nanmean(inv_peak_height_bootstrap, 0)
    inv_peak_height_err = 2 * np.nanstd(inv_peak_height_bootstrap, 0)

    peak_width = np.nanmean(peak_width_bootstrap, 0)
    peak_width_err = 2 * np.nanstd(peak_width_bootstrap, 0)

    return Tc_bootstrap, inv_peak_height_bootstrap, peak_width_bootstrap, Tc, Tc_err, inv_peak_height, inv_peak_height_err, peak_width, peak_width_err

# %%
def extrapolate_thermodynamic_limit_mean_field_graphs(sizes, Tc_bootstrap, inv_peak_height_bootstrap, peak_width_bootstrap):
    """
    This function extrapolates the thermodynamic limit of the critical temperature (Tc),
    inverse peak height, and peak width for mean field graphs, based on the results from bootstrap sampling.

    Parameters:
    sizes (list or numpy.ndarray): Array-like object containing the sizes of the system.
    Tc_bootstrap (numpy.ndarray): Array containing the bootstrap samples for the critical temperatures.
    inv_peak_height_bootstrap (numpy.ndarray): Array containing the bootstrap samples for the inverse peak heights.
    peak_width_bootstrap (numpy.ndarray): Array containing the bootstrap samples for the peak widths.

    Returns:
    tuple: Returns a tuple containing the estimated thermodynamic limits for Tc, inverse peak height,
           and peak width, as well as their associated uncertainties (standard deviations).
    """

    # Tc_inf_bootstrap = np.zeros(len(Tc_bootstrap))
    # inv_peak_height_inf_bootstrap = np.zeros(len(inv_peak_height_bootstrap))
    # peak_width_inf_bootstrap = np.zeros(len(peak_width_bootstrap))
    #
    # for i, (Tc, inv_peak_height, peak_width) in enumerate(
    #         zip(Tc_bootstrap, inv_peak_height_bootstrap, peak_width_bootstrap)):
    #     try:
    #         Tc_inf_bootstrap[i] = np.polyfit((np.array(sizes) ** (-1 / 3))[-2:], Tc[-2:], 1)[1]
    #         inv_peak_height_inf_bootstrap[i] = \
    #         np.polyfit((np.array(sizes) ** (-1 / 3))[-2:], inv_peak_height[-2:], 1)[1]
    #         peak_width_inf_bootstrap[i] = \
    #         np.polyfit((np.array(sizes) ** (-1 / 3))[-2:], peak_width[-2:], 1)[1]
    #     except:
    #         Tc_inf_bootstrap[i] = np.nan
    #         inv_peak_height_inf_bootstrap[i] = np.nan
    #         peak_width_inf_bootstrap[i] = np.nan
    #
    # Tc_inf = np.nanmean(Tc_inf_bootstrap)
    # Tc_inf_err = 2 * np.nanstd(Tc_inf_bootstrap)
    #
    # inv_peak_height_inf = np.nanmean(inv_peak_height_inf_bootstrap)
    # inv_peak_height_inf_err = 2 * np.nanstd(inv_peak_height_inf_bootstrap)
    #
    # peak_width_inf = np.nanmean(peak_width_inf_bootstrap)
    # peak_width_inf_err = 2 * np.nanstd(peak_width_inf_bootstrap)

    # Tc_inf = np.polyfit((np.array(sizes) ** (-1 / 3))[-2:], np.nanmean(Tc_bootstrap,0)[-2:], 1)[1]
    # Tc_inf_err = np.abs(Tc_inf - np.polyfit((np.array(sizes) ** (-1 / 3))[:-2], np.nanmean(Tc_bootstrap,0)[:-2], 1)[1])
    #
    # inv_peak_height_inf = np.polyfit((np.array(sizes) ** (-1 / 3))[-2:], np.nanmean(inv_peak_height_bootstrap,0)[-2:], 1)[1]
    # inv_peak_height_inf_err = np.abs(inv_peak_height_inf - np.polyfit((np.array(sizes) ** (-1 / 3))[:-2], np.nanmean(inv_peak_height_bootstrap,0)[:-2], 1)[1])
    #
    # peak_width_inf = np.polyfit((np.array(sizes) ** (-1 / 3))[-2:], np.nanmean(peak_width_bootstrap,0)[-2:], 1)[1]
    # peak_width_inf_err = np.abs(peak_width_inf - np.polyfit((np.array(sizes) ** (-1 / 3))[:-2], np.nanmean(peak_width_bootstrap,0)[:-2], 1)[1])


    bootstrap_data = [Tc_bootstrap, inv_peak_height_bootstrap, peak_width_bootstrap]
    extrapolations = []
    extrapolations_errors = []
    sizes_power = (np.array(sizes) ** (-1 / 3))[-2:]

    def poly1(x, m, b):
        return m * x + b

    for bootstrap_sample in bootstrap_data:
        x = sizes_power
        y = np.nanmean(bootstrap_sample, 0)[-2:]
        yerr = 2 * np.nanstd(bootstrap_sample, 0)[-2:]

        popt, pcov = curve_fit(poly1, x, y, sigma=yerr, absolute_sigma=True)

        extrapolations.append( popt[-1] )
        extrapolations_errors.append( 2 * np.sqrt(np.diag(pcov))[-1] )

    Tc_inf, inv_peak_height_inf, peak_width_inf = extrapolations
    Tc_inf_err, inv_peak_height_inf_err, peak_width_inf_err = extrapolations_errors

    return Tc_inf, Tc_inf_err, inv_peak_height_inf, inv_peak_height_inf_err, peak_width_inf, peak_width_inf_err