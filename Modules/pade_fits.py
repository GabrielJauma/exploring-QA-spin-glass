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

    #
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

        # force the func to have <0 derivative at x=0
        # if a[1]*(1+b[0])- a[0]*b[1]>0:
        #     a0 = a[1]*(1+b[0])/b[1]
        #     y = (a0 + sum(a[i] * x ** i for i in range(1,len(a)))) / (1 + sum(b[i] * x ** i for i in range(len(b))))
        # else:
        #     y = sum(a[i] * x ** i for i in range(len(a))) / (1 + sum(b[i] * x ** i for i in range(len(b))))

        # force the func to have 0 derivative at x=0
        # a0 = a[1] * (1 + b[0]) / b[1]
        # a1 = a0 * b[1] / (1 + b[0])
        # y = (a0 + a1 * x + sum(a[i] * x ** i for i in range(2, len(a)))) / ( 1 + sum(b[i] * x ** i for i in range(len(b))))

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

        # force the func to have <0 derivative at x=0
        # if a[1]*(1+b[0])- a[0]*b[1]>0:
        #     a0 = a[1] * (1 + b[0]) / b[1]
        #     func = lambda x: (a0 + sum(a[i] * x ** i for i in range(1, len(a)))) / ( 1 + sum(b[i] * x ** i for i in range(len(b))))
        # else:
        #     func = lambda x: sum(a[i] * x ** i for i in range(len(a))) / (1 + sum(b[i] * x ** i for i in range(len(b))))

        # force the func to have 0 derivative at x=0 and go through 1 at x=0
        # a0 = 1 + b[0]
        # a1 = a0 * b[1] / (1 + b[0])
        # func = lambda x: (a0 + a1*x + sum(a[i] * x ** i for i in range(2, len(a)))) / ( 1 + sum(b[i] * x ** i for i in range(len(b))))
        return func

    def fitea_p(self, x, y, y_err, igu, verbose=False, maxfev=5000000, add_0=False):
        """
        Input;
        x, y, yerr ---> data to fit.
        Outputs:
        igu     --->  parameters of the fitting
        """
        try:
            if add_0:
                popt, pcov = curve_fit(self.evalua_pade, np.concatenate([[0],x]), np.concatenate([[1],y]), igu, sigma=np.concatenate([[add_0],y_err]), maxfev=maxfev)
            else:
                popt, pcov = curve_fit(self.evalua_pade, x, y, igu, sigma=y_err, maxfev=maxfev)
        except:
            return [None], [None], 10, [None]
        perr = np.sqrt(np.abs(np.diag(pcov)))
        r = y - self.evalua_pade(x, *popt)
        res = r / y_err
        chisq = sum((res) ** 2) / (y_err.shape[0] - len(igu))
        if verbose: print("chisq/df", chisq)
        return popt, perr, chisq, res

    def pade_opt_rand(self, x, y, y_err, ntr=10, verbose=False, maxfev=5000000, add_0=False):
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
        res = [self.fitea_p(x, y, y_err, p0, maxfev=maxfev, add_0=add_0) for p0 in igu]
        xchi = np.array([res[i][2] for i in range(ntr)])
        ic = (np.abs(xchi)).argmin()
        mu = np.sum(xchi) / ntr
        var = np.sqrt(np.sum(xchi ** 2) / ntr - mu ** 2)
        if verbose:
            print("best estimation has order", self.n, self.m, "  and rchi", xchi[ic])
        return res[ic]

    def sev_func_pade(self, x, y, y_err, ntr=100, verbose=False, maxfev=5000000, add_0=False):
        """
        Fitea via Pade for all curves
        Input;
        x, y, yerr ---> data to fit.
        Outputs:
        igu     --->  parameters of the fitting
        """
        if type(x) == list:
            n = len(x)
            yv = [self.pade_opt_rand(x[i], y[i], y_err[i], ntr=ntr, verbose=verbose, maxfev=maxfev, add_0=add_0) for i in range(n)]
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
    #
    #
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

    def give_c(self, x, *para):
        a = para[0:self.ns]
        d = para[self.ns:self.ns + self.nrho]
        b = para[self.ns + self.nrho: self.ns + self.nrho + self.nb]
        nu = para[self.ns + self.nrho + self.nb]
        wc = para[self.ns + self.nrho + self.nb + 1]
        #         print(a,d,nu,wc)
        rho = (x[0] - wc) + sum(d[i] * (x[0] - wc) ** (i + 2) for i in range(self.nrho))
        xs = rho * x[1] ** (1 / nu)
        beta = sum(b[i] * (x[0] - wc) ** (i) for i in range(self.nb))
        if self.nb == 0:
            f = 0
        elif self.nb > 0:
            #             y = para[-1]
            f = x[1] ** -1 * sum(a[i] * xs ** i for i in range(self.ns)) * (1 + beta * x[1] ** (-yi))
        return f, nu, wc

    def sc_function(self, *para):
        if self.nrho > 0:
            print("WARNING: the scaling function also dependes on W-W_c!")
        a = para[0:self.ns]
        #         print(a,d,nu,wc)
        f = lambda x: sum(a[i] * x ** i for i in range(self.ns))
        return f

    def give_scalingv(self, x, *para):
        a = para[0:self.ns]
        d = para[self.ns:self.ns + self.nrho]
        b = para[self.ns + self.nrho: self.ns + self.nrho + self.nb]
        nu = para[self.ns + self.nrho + self.nb]
        wc = para[self.ns + self.nrho + self.nb + 1]
        #         print(a,d,nu,wc)
        rho = (x - wc) + sum(d[i] * (x - wc) ** (i + 2) for i in range(self.nrho))
        return rho


def separa(x, s, y, y_e):
    xp = []
    sp = []
    yp = []
    yp_e = []
    nc = 0
    for i in range(len(x) - 1):
        if x[i] > x[i + 1]:
            xp.append(x[nc:i + 1])
            yp.append(y[nc:i + 1])
            yp_e.append(y_e[nc:i + 1])
            sp.append(s[nc:i + 1])
            nc = i + 1
    xp.append(x[nc:])
    yp.append(y[nc:])
    sp.append(s[nc:])
    yp_e.append(y_e[nc:])
    return xp, sp, yp, yp_e


# def pade_best(x, y, y_err, ntr=100, ic=[4, 5], jc=[2, 3], verbose=True, maxfev=500000):
def pade_best(x, y, y_err, ntr=10, ic=[4, 5], jc=[2, 3], verbose=True, maxfev=10000, add_0=False):
    ppade = []
    fpade = []
    dfpade = []
    st_chi = []
    st_ce = []
    od = []
    inx = itertools.product(ic, jc)
    for order in inx:
        A = fit_pade(*order)
        f_sev, pade_c, rchi = A.sev_func_pade(x[:], y[:], y_err[:], ntr=ntr, verbose=verbose, maxfev=maxfev, add_0=add_0)
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
        # inopt = [closer_to_one_but_below(st_chi[:, i]) for i in range(len(x))] # choose the pade with st_chi closer to 1 but <= 1
        inopt = [(np.abs(np.array(st_chi[:, i])-1)).argmin() for i in range(len(x))] # choose the pade with st_chi closer to 1
        # inopt = [(np.array(st_chi[:, i])).argmin() for i in range(len(x))] # choose the pade with st_chi closer to 0
        # inopt = [((st_chi-1)**2).sum(1).argmin()]*len(x) # choose one pade for all sizes, the pade that is in average closer to one for all the lines
        # print(((st_chi-1)**2).sum(1))
        fpade_out = [fpade[inopt[i]][i] for i in range(len(x))]
        dfpade_out = [dfpade[inopt[i]][i] for i in range(len(x))]
        # ppade_out = [ppade[inopt[i]][i] for i in range(len(x))]
        if verbose:
            print([st_chi[inopt[i], i] for i in range(len(x))])
    else:
        # inopt = closer_to_one_but_below(st_chi)
        inopt = (np.abs(np.array(st_chi)-1)).argmin()
        # inopt = ((st_chi-1)**2).sum(1).argmin()
        fpade_out = fpade[inopt]
        dfpade_out = dfpade[inopt]
        # ppade_out = ppade[inopt]
        if verbose:
            print("best rchi", st_chi[inopt])

    return fpade_out, dfpade_out, st_chi[inopt], inopt


def pade_best_specific_ic_jc(x_vs_size, y_vs_size, y_err_vs_size, ntr, ic_vs_size, jc_vs_size, verbose=True, maxfev=10000, add_0=False):
    fpade = []
    dfpade = []
    st_chi = []

    for x, y, y_err, ic, jc in zip(x_vs_size, y_vs_size, y_err_vs_size, ic_vs_size, jc_vs_size):
        A = fit_pade(ic, jc)
        f_sev, pade_c, rchi = A.sev_func_pade(x, y, y_err, ntr=ntr, verbose=verbose, maxfev=maxfev, add_0=add_0)

        df = A.der_pade(pade_c)

        st_chi.append(rchi)
        fpade.append(f_sev)
        dfpade.append(df)
    st_chi = np.array(st_chi)

    return fpade, dfpade, st_chi, None


def closer_to_one_but_below(st_chi):
    inopt = np.nan
    chi_0 = 10
    for i, chi in enumerate(st_chi):
        if np.abs(chi - 1) <= np.abs(chi_0 - 1) and chi < 1:
            inopt = i
            chi_0 = chi
    return inopt


def pade_best_fast(x, y, y_err, ic=[5], jc=[6], ntr=10, maxfev=10000):
    A = fit_pade(ic[0], jc[0])
    pade, _, _ = A.sev_func_pade(x[:], y[:], y_err[:], ntr=ntr, verbose=False, maxfev=maxfev)
    # df = [A.der_pade(pade_c[i]) for i in range(len(pade_c))]

    return pade

def pade_best_fast_different_ic_jc_each(x, y, y_err, ic=[5], jc=[6], ntr=10, maxfev=10000):
    pades = []
    for X, Y, Y_err, IC, JC in zip(x, y, y_err, ic, jc):
        A = fit_pade(IC, JC)
        pade, _, _ = A.sev_func_pade(X, Y, Y_err, ntr=ntr, verbose=False, maxfev=maxfev)
        pades.append( pade)

    return pades


def pade_fss(sizes,  T_vs_size_best, dg_dT_vs_size_best, error_dg_dT_vs_size_best, T_term_vs_size=False,
             ntr=10, ic=[5], jc=[6], add_0=False, method_ic_jc='best'):

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
    nf = len(sizes)

    if method_ic_jc == 'best':
        dg_dT_pade, _, l_rchi, inopt = pade_best(Tfit, Ofit, np.array(Ofit_er), ntr, ic, jc, add_0=add_0)
    elif method_ic_jc == 'specific':
        dg_dT_pade, _, l_rchi, inopt = pade_best_specific_ic_jc(Tfit, Ofit, np.array(Ofit_er), ntr, ic, jc, add_0=add_0)

    # Data to extrapolate the maxima
    T_c = estimate_Tc_with_pade(sizes, T0, Tf, dg_dT_pade, return_T_max=True)

    peak_height = np.array([dg_dT_pade[i](T_c[i]) for i in range(nf)])

    print(f'Tc = {np.polyfit(1 / sizes[-3:] ** (1 / 3), T_c[-3:], 1, cov=True)[0][-1]}')

    return dg_dT_pade, T_c, peak_height


def pade_fss_analysis_figures(sizes, T_vs_size, dg_dT_vs_size, error_vs_size, T_term_vs_size=False, ntr=10, ic=[5], jc=[6],
                      figsize_in=(16 / 4, 9 / 4), dpi_in=100, adjacency='random_regular_3', out='figures', add_0=False, method_ic_jc='best'):
    Lfit = sizes
    sizes = np.array(sizes)

    if T_term_vs_size:
        T_term_ind = [np.where(T[-1] > T_term)[0][0] for T_term, T in zip(T_term_vs_size, T_vs_size)]
    else:
        T_term_ind = [0] * len(sizes)

    try:
        Tfit = [T[-1][k:] for T, k in zip(T_vs_size, T_term_ind)]
    except:
        Tfit = [T[k:] for T, k in zip(T_vs_size, T_term_ind)]

    try:
        Ofit = [B[-1][k:] for B, k in zip(dg_dT_vs_size, T_term_ind)]
    except:
        Ofit = [B[k:] for B, k in zip(dg_dT_vs_size, T_term_ind)]
    try:
        Ofit_er = [err[-1][k:] for err, k in zip(error_vs_size, T_term_ind)]
    except:
        Ofit_er = [err[k:] for err, k in zip(error_vs_size, T_term_ind)]

    T0 = Tfit[0][0]
    Tf = Tfit[0][-1]
    nf = len(sizes)

    if method_ic_jc == 'best':
        dg_dT_pade, _, l_rchi, inopt = pade_best(Tfit, Ofit, np.array(Ofit_er), ntr, ic, jc, add_0=add_0)
    elif method_ic_jc == 'specific':
        dg_dT_pade, _, l_rchi, inopt = pade_best_specific_ic_jc(Tfit, Ofit, np.array(Ofit_er), ntr, ic, jc, add_0=add_0)

    if out == 'figures':
        T_pade = np.linspace(T0, Tf, 1000)

        fig, ax2 = plt.subplots(figsize=np.array(figsize_in) * 1.5 * [1 / 2, 1], dpi=dpi_in)
        for i in range(0, nf):
            ax2.plot(T_pade, -dg_dT_pade[i](T_pade), label=" size" + str(Lfit[i]), color=color[i])
            ax2.errorbar(Tfit[i], -Ofit[i], yerr=Ofit_er[i], linewidth=0, markerfacecolor="None", capsize=4, capthick=1,
                 elinewidth=1, color=color[i])
        ax2.set_xlabel('$T$')
        ax2.set_ylabel("$dg/dT$")
        ax2.set_title(f'ic={ic}, jc={jc}')
        ax2.legend()
        fig.tight_layout()
        fig.show()

    # Data to extrapolate the maxima
    T_max = estimate_Tc_with_pade(sizes, T0, Tf, dg_dT_pade, return_T_max=True)[1]
    dg_dT_pade_T_max = np.array([dg_dT_pade[i](T_max[i]) for i in range(nf)])

    add_float = lambda f, float_value: lambda x: f(x) + float_value
    width_L = np.array(
        [root_scalar(add_float(dg_dT_pade[i], -dg_dT_pade_T_max[i] * 0.8), method='brentq', bracket=(T0, T_max[i])).root
         for i in range(nf)])
    width_R = np.array(
        [root_scalar(add_float(dg_dT_pade[i], -dg_dT_pade_T_max[i] * 0.8), method='brentq', bracket=(T_max[i], Tf)).root
         for i in range(nf)])
    peak_width = width_R - width_L

    z, fit_T_max_params = np.polyfit(1 / sizes[-3:] ** (1 / 3), T_max[-3:], 1, cov=True)
    fit_T_max = np.poly1d(z)

    # z2 = np.polyfit(1/np.log(sizes), T_max, 1)
    # p2 = np.poly1d(z2)

    dg_dT_pade_T_max = np.abs(dg_dT_pade_T_max)
    z3, fit_1_over_dg_dT_T_max_params = np.polyfit(1 / sizes[-3:] ** (1 / 3), 1 / dg_dT_pade_T_max[-3:], 1, cov=True)
    fit_1_over_dg_dT_T_max = np.poly1d(z3)

    z4, fit_peak_width_params = np.polyfit(1 / sizes[-3:] ** (1 / 3), peak_width[-3:], 1, cov=True)
    fit_peak_width = np.poly1d(z4)

    print(f'Tc = {z[-1]}')

    x_fit_T_max = np.linspace(0, 1.5 / sizes[0] ** (1 / 3), 100)
    # xx2 = np.linspace(0, 1.5/np.log(sizes[0]), 100)

    if out == 'figures':

        fig, (ax1, ax3, ax4) = plt.subplots(ncols=3, figsize=figsize_in, dpi=dpi_in)
        if adjacency != 'chimera' and adjacency != 'pegasus' and adjacency != 'zephyr':
            ax1.plot(1 / sizes ** (1 / 3), T_max, "o")
            ax1.plot(x_fit_T_max, fit_T_max(x_fit_T_max), "-")
            ax1.set_xlabel('$1/N^{1/3}$')
            ax1.set_ylabel("$T_c$")

            # ax2.plot(1 / np.log(sizes), T_max, "o")
            # ax2.plot(xx2, p2(xx2), "-")
            # ax2.set_xlabel('$1/log(N)$')
            # ax2.set_ylabel("$T_c$")

            ax3.plot(1 / sizes ** (1 / 3), 1 / dg_dT_pade_T_max, "o")
            ax3.plot(x_fit_T_max, fit_1_over_dg_dT_T_max(x_fit_T_max), "-")
            ax3.set_xlabel('$1/N^{1/3}$')
            ax3.set_ylabel("$(dg/dT|_{T_c})^{-1}$")
            ax3.set_ylim(bottom=-0.1)

            ax4.plot(1 / sizes ** (1 / 3), peak_width, "o")
            ax4.plot(x_fit_T_max, fit_peak_width(x_fit_T_max), "-")
            ax4.set_xlabel('$1/N^{1/3}$')
            ax4.set_ylabel("Peak peak_width")

        else:
            ax1.plot(sizes, T_max, "o")
            ax1.set_xlabel('$N$')
            ax1.set_ylabel("$T_c$")

            ax3.plot(sizes, dg_dT_pade_T_max, "o")
            # ax3.plot(x_fit_T_max, fit_1_over_dg_dT_T_max(x_fit_T_max), "-")
            ax3.set_xlabel('$N$')
            ax3.set_ylabel("$(dg/dT|_{T_c})$")
            ax3.set_ylim(bottom=-0.1)

            ax4.plot(sizes, peak_width, "o")
            # ax4.plot(x_fit_T_max, fit_peak_width(x_fit_T_max), "-")
            ax4.set_xlabel('$N$')
            ax4.set_ylabel("Peak peak_width")

        fig.suptitle(f'Tc = {z[-1]}')
        fig.tight_layout()
        fig.show()

    # fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=figsize_in, dpi=dpi_in)
    # ax1.plot(1 / sizes ** (1 / 3), 1 / gdmax, "o")
    # ax1.plot([0, 3 / sizes[2] ** (1 / 3)], [0, 3 / gdmax[2]], "-")
    # ax1.set_xlabel('$1/N^{1/3}$')
    # ax1.set_ylabel("$\\left(dg/dT|_{Max}\\right)^{-1}$")
    #
    # ax2.plot(1 / np.log(sizes), 1 / gdmax, "o")
    # ax2.plot([0, 3 / np.log(sizes)[2]], [0, 3 / gdmax[2]], "-")
    # ax2.set_xlabel('$1/\log(N)$')
    # ax2.set_ylabel("$\\left(dg/dT|_{Max}\\right)^{-1}$")
    #
    # ax3.plot(1 / (np.log(sizes)) ** 2, 1 / gdmax, "o")
    # ax3.plot([0, 3 / (np.log(sizes)[2]) ** 2], [0, 3 / gdmax[2]], "-")
    # ax3.set_xlabel('$1/\log(N)^{2}$')
    # ax3.set_ylabel("$\\left(dg/dT|_{Max}\\right)^{-1}$")
    # fig.tight_layout()
    # fig.show()
    if out == 'fits':
        return dg_dT_pade, T_max, dg_dT_pade_T_max, peak_width, fit_T_max, fit_1_over_dg_dT_T_max, fit_peak_width, fit_T_max_params, fit_1_over_dg_dT_T_max_params, fit_peak_width_params
    else:
        return inopt

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

def estimate_Tc_with_pade_bootstrap(sizes, T_vs_size_best, error_vs_size_best, dg_dT_bootstrap_vs_size_best, ic=[5], jc=[6],
                                    ntr=10, maxfev=10000):

    T0 = T_vs_size_best[0]
    Tf = T_vs_size_best[-1]

    n_bootstrap = len(dg_dT_bootstrap_vs_size_best[0])
    Tc_bootstrap = np.zeros([n_bootstrap, len(sizes)])
    peak_height_bootstrap = np.zeros([n_bootstrap, len(sizes)])
    peak_width_bootstrap = np.zeros([n_bootstrap, len(sizes)])

    for i_b in range(n_bootstrap):
        dgdT_bootstrap = [dgdT[i_b]for dgdT in dg_dT_bootstrap_vs_size_best]

        try:
            # dg_dT_pade = pade_best_fast_different_ic_jc_each(T_vs_size_best, dgdT_bootstrap, np.array(error_vs_size_best), ic=ic, jc=jc, ntr=ntr, maxfev=maxfev)
            dg_dT_pade = pade_best_specific_ic_jc(T_vs_size_best, dgdT_bootstrap, np.array(error_vs_size_best), ntr, ic, jc, maxfev=maxfev)[0]
            Tc_bootstrap[i_b, :] = estimate_Tc_with_pade(sizes, T0, Tf, dg_dT_pade, return_T_max=True)
            peak_height_bootstrap[i_b, :] = np.array([dg_dT_pade[i](Tc_bootstrap[i_b, i]) for i in range(len(sizes))])
            peak_width_bootstrap[i_b, :] = estimate_peak_width(sizes, T0, Tf,dg_dT_pade, Tc_bootstrap[i_b, :], peak_height_bootstrap[i_b, :])

        except:
            continue
        #     Tc_bootstrap[i_b, :] = np.nan*np.ones(len(sizes))
        #     peak_height_bootstrap[i_b, :] = np.nan*np.ones(len(sizes))
        #     peak_width_bootstrap[i_b, :] = np.nan*np.ones(len(sizes))


    return Tc_bootstrap, peak_height_bootstrap, peak_width_bootstrap


def estimate_Tc_with_pade_bootstrap_parallel(sizes, T_vs_size, error_vs_size, dg_dT_bootstrap_vs_size, ic=[5], jc=[6],
                                             ntr=10, maxfev=10000, threads=8):
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
    peak_height_bootstrap = []
    peak_width_bootstrap = []

    for Tc_height_widht in Tc_height_widht_bootstrap:
        Tc_bootstrap.append(Tc_height_widht[0])
        peak_height_bootstrap.append(Tc_height_widht[1])
        peak_width_bootstrap.append(Tc_height_widht[2])


    Tc_bootstrap = np.array(Tc_bootstrap).reshape((-1, len(sizes)))
    peak_height_bootstrap = np.array(peak_width_bootstrap).reshape((-1, len(sizes)))
    peak_width_bootstrap = np.array(peak_width_bootstrap).reshape((-1, len(sizes)))

    # Remove outliers
    for Tc_bootstrap_vs_size in Tc_bootstrap.T:
        Tc_bootstrap_vs_size[np.abs(Tc_bootstrap_vs_size - np.nanmean(Tc_bootstrap_vs_size)) > 4 * np.nanstd(Tc_bootstrap_vs_size)] = np.nan

    for peak_height_bootstrap_vs_size in peak_height_bootstrap.T:
        peak_height_bootstrap_vs_size[np.abs(peak_height_bootstrap_vs_size - np.nanmean(peak_height_bootstrap_vs_size)) > 4 * np.nanstd(peak_height_bootstrap_vs_size)] = np.nan

    for peak_width_bootstrap_size in peak_width_bootstrap.T:
        peak_width_bootstrap_size[np.abs(peak_width_bootstrap_size - np.nanmean(peak_width_bootstrap_size)) > 4 * np.nanstd(peak_width_bootstrap_size)] = np.nan

    return Tc_bootstrap, peak_height_bootstrap, peak_width_bootstrap