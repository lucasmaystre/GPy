import numpy as np
import functools

from math import exp, log, pi, sqrt  # Faster than numpy equivalents.
from scipy.misc import logsumexp

from ..util.univariate_Gaussian import std_norm_cdf


SQRT2 = sqrt(2.0)
SQRT2PI = sqrt(2.0 * pi)


def logit_match_moments(mean_cav, cov_cav):
    # Adapted from the GPML function `likLogistic.m`.
    # First use a scale mixture.
    lambdas = sqrt(2) * np.array([0.44, 0.41, 0.40, 0.39, 0.36]);
    cs = np.array([
      1.146480988574439e+02,
      -1.508871030070582e+03,
      2.676085036831241e+03,
      -1.356294962039222e+03,
      7.543285642111850e+01
    ])
    arr1, arr2, arr3 = np.zeros(5), np.zeros(5), np.zeros(5)
    for i, x in enumerate(lambdas):
        arr1[i], arr2[i], arr3[i] = probit_match_moments(x * mean_cav,
                                                         x*x * cov_cav)
    logpart1 = logsumexp(arr1, b=cs)
    dlogpart1 = (np.dot(np.exp(arr1) * arr2, cs * lambdas)
                 / np.dot(np.exp(arr1), cs))
    d2logpart1 = (np.dot(np.exp(arr1) * (arr2 * arr2 + arr3),
                         cs * lambdas * lambdas)
                  / np.dot(np.exp(arr1), cs)) - (dlogpart1 * dlogpart1)
    # Tail decays linearly in the log domain (and not quadratically.)
    exponent = -10.0 * (abs(mean_cav) - (196.0 / 200.0) * cov_cav - 4.0) 
    if exponent < 500:
        lambd = 1.0 / (1.0 + exp(exponent))
        logpart2 = min(cov_cav / 2.0 - abs(mean_cav), -0.1)
        dlogpart2 = 1.0
        if mean_cav > 0:
            logpart2 = log(1 - exp(logpart2))
            dlogpart2 = 0.0
        d2logpart2 = 0.0
    else:
        lambd, logpart2, dlogpart2, d2logpart2 = 0.0, 0.0, 0.0, 0.0
    logpart = (1 - lambd) * logpart1 + lambd * logpart2
    dlogpart = (1 - lambd) * dlogpart1 + lambd * dlogpart2
    d2logpart = (1 - lambd) * d2logpart1 + lambd * d2logpart2
    return logpart, dlogpart, d2logpart


def probit_match_moments(mean_cav, cov_cav):
    # Adapted from the GPML function `likErf.m`.
    z = mean_cav / sqrt(1 + cov_cav)
    logpart, val = _logphi(z)
    dlogpart = val / sqrt(1 + cov_cav)  # 1st derivative w.r.t. mean.
    d2logpart = -val * (z + val) / (1 + cov_cav)
    return logpart, dlogpart, d2logpart


# Some magic constants for a stable computation of logphi(z).
CS = [
  0.00048204, -0.00142906, 0.0013200243174, 0.0009461589032, -0.0045563339802,
  0.00556964649138, 0.00125993961762116, -0.01621575378835404,
  0.02629651521057465, -0.001829764677455021, 2*(1-pi/3), (4-pi)/3, 1, 1,]
RS = [
  1.2753666447299659525, 5.019049726784267463450, 6.1602098531096305441,
  7.409740605964741794425, 2.9788656263939928886,]
QS = [
  2.260528520767326969592, 9.3960340162350541504, 12.048951927855129036034,
  17.081440747466004316, 9.608965327192787870698, 3.3690752069827527677,]

def _logphi(z):
    # Adapted from the GPML function `logphi.m`.
    if z * z < 0.0492:
        # First case: z close to zero.
        coef = -z / SQRT2PI
        val = functools.reduce(lambda acc, c: coef * (c + acc), CS, 0)
        res = -2 * val - log(2)
        dres = exp(-(z * z) / 2 - res) / SQRT2PI
    elif z < -11.3137:
        # Second case: z very small.
        num = functools.reduce(lambda acc, r: -z * acc / SQRT2 + r, RS,
                     0.5641895835477550741)
        den = functools.reduce(lambda acc, q: -z * acc / SQRT2 + q, QS, 1.0)
        res = log(num / (2 * den)) - (z * z) / 2
        dres = abs(den / num) * sqrt(2.0 / pi)
    else:
        res = log(std_norm_cdf(z))
        dres = exp(-(z * z) / 2 - res) / SQRT2PI
    return res, dres
