import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import locale
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
# _locale_radix = locale.localeconv()['decimal_poi

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
            f = x[1] ** -1 * sum(a[i] * xs ** i for i in range(self.ns)) * (1 + beta * x[1] ** (-yi))
        return f

    def fitea_p(self, x_opt, y_opt, y_err, w_i=17.5, nu_i=1, y_i=-0.5, verbose=True):
        #        igu = [1] * (self.nrho+self.ns+self.nb+ self.nir)
        igu = [np.random.rand() / 10 ** i for i in range((self.nrho + self.ns + self.nb))]
        #         print(igu)
        igu.extend([nu_i, w_i, y_i])
        popt, pcov = curve_fit(self.func2, x_opt, y_opt, igu, sigma=y_err, absolute_sigma="False", maxfev=1000000)
        perr = np.sqrt(np.diag(pcov))
        r = y_opt - self.func2(x_opt, *popt)
        res = r / y_err
        chisq = sum((res) ** 2) / (y_err.shape[0] - len(igu))
        print("numero de puntos", y_err.shape[0])
        # print(y_err**2)
        if verbose:
            for i in range(len(igu)):
                print(popt[i], "+-", perr[i])
            print("chisq/df", chisq)
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
            f = sum(a[i] * xs ** i for i in range(self.ns)) * (beta * x[1] ** (-0.6))
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