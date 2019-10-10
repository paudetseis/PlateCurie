# Copyright 2019 Pascal Audet
#
# This file is part of PlateCurie.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This :mod:`~platecurie` module contains the following functions: 

- :func:`~platecurie.estimate.bayes_estimate_cell`: Set up :mod:`~pymc` model and estimate the parameters
  of the magnetic layer model using a probabilistic Bayesian inference method.  
- :func:`~platecurie.estimate.get_bayes_estimates`: Explore the output of sampling the :mod:`~pymc` model
- :func:`~platecurie.estimate.L2_estimate_cell`: Set up non-linear curve fitting to estimate the parameters
  of the magnetic layer model using non-linear least-squares from the function :func:`scipy.optimize.curve_fit`.  
- :func:`~platecurie.estimate.get_L2_estimates`: Explore the output the non-linear inversion
- :func:`~platecurie.estimate.calculate_psd`: Calculate the analytical PSD function. 

An internal function is available to calculate predicted power-spectral density data
with a ``theano`` decorator to be incorporated as ``pymc`` variables. These functions are
used within :class:`~platecurie.classes.Project` methods as with :mod:`~platecurie.plotting`
functions.

"""

# -*- coding: utf-8 -*-
import numpy as np
import pymc3 as pm
from theano.compile.ops import as_op
import theano.tensor as tt
from scipy.optimize import curve_fit
import pandas as pd

def bayes_estimate_cell(k, psd, epsd, fix_beta=None, prior_zt=None, 
    draws=500, tunes=500, cores=4):
    """
    Function to estimate the parameters of the magnetic model at a single cell location
    of the input grids. 

    Can incorporate 2 ('A', and 'dz'), 3 (either 'A', 'zt' and 'dz' or 
    'A', 'dz', and 'beta') or 4 ('A', 'zt', 'dz', and 'beta') 
    stochastic variables.

    :type k: :class:`~numpy.ndarray`
    :param k: 1D array of wavenumbers
    :type psd: :class:`~numpy.ndarray`
    :param psd: 1D array of wavelet scalogram (wavelet PSD)
    :type epsd: :class:`~numpy.ndarray`
    :param epsd: 1D array of error on wavelet scalogram (wavelet PSD)
    :type fix_beta: float, optional
    :param fix_beta: Fixed ``beta`` parameter or estimate it if ``None``
    :type prior_zt: float, optional
    :param prior_zt: Fixed ``zt`` parameter or estimate it if ``None``
    :type draws: int
    :param draws: Number of samples to draw from the posterior
    :type tunes: int
    :param tunes: Number of samples to discard (burn-in)
    :type cores: int
    :param cores: Number of computer cores to use in calculating independent markov chains

    :return:
        (tuple): Tuple containing:
            * ``trace`` : :class:`~pymc3.backends.base.MultiTrace`
                Posterior samples from the MCMC chains
            * ``summary`` : :class:`~pandas.core.frame.DataFrame`
                Summary statistics from Posterior distributions
            * ``map_estimate`` : dict
                Container for Maximum a Posteriori (MAP) estimates
    """

    with pm.Model() as model:

        # k is an array - needs to be passed as distribution
        k_obs = pm.Normal('k', mu=k, sigma=1., observed=k)

        # Prior distributions
        A = pm.Uniform('A', lower=1., upper=30.)
        dz = pm.Uniform('dz', lower=1., upper=50.)

        if fix_beta is not None:
            # Pass `beta` variable as theano variable
            beta = tt.as_tensor_variable(np.float64(fix_beta))
        else:
            beta = pm.Uniform('beta', lower=0., upper=4.)

        if prior_zt is not None:
            # Pass `zt` variable as theano variable
            zt = tt.as_tensor_variable(np.float64(prior_zt))
        else:
            # Prior distribution of `zt`
            zt = pm.Uniform('zt', lower=0., upper=10.)

        # Predicted PSD
        psd_exp = calculate_psd_theano(k_obs, A, zt, dz, beta)

        # Uncertainty as observed distribution
        dpsd = 3.*0.434*epsd/psd
        sigma = pm.Normal('sigma', mu=epsd, sigma=1., \
            observed=dpsd)

        # Likelihood of observations
        psd_obs = pm.Normal('psd_obs', mu=psd_exp, \
            sigma=sigma, observed=np.log(psd))

        # Sample the Posterior distribution
        trace = pm.sample(draws, tune=tunes, cores=cores)

        # Get Max a porteriori estimate
        map_estimate = pm.find_MAP()

        # Get Summary
        summary = pm.summary(trace)

    return trace, summary, map_estimate

def get_bayes_estimates(summary, map_estimate):
    """
    Extract useful estimates from the Posterior distributions.

    :type summary: :class:`~pandas.core.frame.DataFrame`
    :param summary: Summary statistics from Posterior distributions
    :type map_estimate: dict
    :param map_estimate: Container for Maximum a Posteriori (MAP) estimates

    :return: 
        (tuple): tuple containing:
            * mean_A (float) : Mean value of the constant term ``A``
            * std_A (float)  : Standard deviation of the constant term ``A``
            * best_A (float) : Most likely value of the constant term ``A``
            * mean_dz (float)  : Mean value of the magnetic layer thickness ``dz``
            * std_dz (float)   : Standard deviation of magnetic layer thickness ``dz``
            * best_dz (float)  : Most likely magnetic layer thickness ``dz``
            * mean_zt (float, optional) : Mean value of the depth to top of layer ``zt``
            * std_zt (float, optional)  : Standard deviation of the depth to top of layer ``zt``
            * best_zt (float, optional) : Most likely value of the depth to top of layer ``zt``
            * mean_beta (float, optional)  : Mean value of the fractal parameter ``beta``
            * std_beta (float, optional)   : Standard deviation of fractal parameter ``beta``
            * best_beat (float, optional)  : Most likely fractal parameter ``beta``

    """

    mean_beta = None
    mean_zt = None

    # Go through all estimates
    for index, row in summary.iterrows():
        if index=='A':
            mean_A = row['mean']
            std_A = row['sd']
            C2_5_A = row['hpd_2.5']
            C97_5_A = row['hpd_97.5']
            best_A = np.float(map_estimate['A'])
        elif index=='zt':
            mean_zt = row['mean']
            std_zt = row['sd']
            C2_5_zt = row['hpd_2.5']
            C97_5_zt = row['hpd_97.5']
            best_zt = np.float(map_estimate['zt'])
        elif index=='dz':
            mean_dz = row['mean']
            std_dz = row['sd']
            C2_5_dz = row['hpd_2.5']
            C97_5_dz = row['hpd_97.5']
            best_dz = np.float(map_estimate['dz'])
        elif index=='beta':
            mean_beta = row['mean']
            std_beta = row['sd']
            C2_5_beta = row['hpd_2.5']
            C97_5_beta = row['hpd_97.5']
            best_beta = np.float(map_estimate['beta'])

    if mean_beta is not None and mean_zt is not None:
        return mean_A, std_A, C2_5_A, C97_5_A, best_A, \
            mean_zt, std_zt, C2_5_zt, C97_5_zt, best_zt, \
            mean_dz, std_dz, C2_5_dz, C97_5_dz, best_dz, \
            mean_beta, std_beta, C2_5_beta, C97_5_beta, best_beta
    elif mean_beta is None and mean_zt is not None:
        return mean_A, std_A, C2_5_A, C97_5_A, best_A, \
            mean_zt, std_zt, C2_5_zt, C97_5_zt, best_zt, \
            mean_dz, std_dz, C2_5_dz, C97_5_dz, best_dz
    elif mean_beta is not None and mean_zt is None:
        return mean_A, std_A, C2_5_A, C97_5_A, best_A, \
            mean_dz, std_dz, C2_5_dz, C97_5_dz, best_dz, \
            mean_beta, std_beta, C2_5_beta, C97_5_beta, best_beta
    else:
        return mean_A, std_A, C2_5_A, C97_5_A, best_A, \
            mean_dz, std_dz, C2_5_dz, C97_5_dz, best_dz      

def L2_estimate_cell(k, psd, epsd, fix_beta=None, prior_zt=None):
    """
    Function to estimate the parameters of the magnetic model at a single cell location
    of the input grids. 

    :type k: :class:`~numpy.ndarray`
    :param k: 1D array of wavenumbers
    :type psd: :class:`~numpy.ndarray`
    :param psd: 1D array of wavelet scalogram (wavelet PSD)
    :type epsd: :class:`~numpy.ndarray`
    :param epsd: 1D array of error on wavelet scalogram (wavelet PSD)
    :type fix_beta: float, optional
    :param fix_beta: Fixed ``beta`` parameter or estimate it if ``None``
    :type prior_zt: float, optional
    :param prior_zt: Fixed ``zt`` parameter or estimate it if ``None``

    :return:
        (tuple): Tuple containing:
            * ``summary`` : :class:`~pandas.core.frame.DataFrame`
                Summary statistics from L2 estimation

    """
    
    # Observed data and error
    y_obs = np.log(psd)
    y_err = 3.*0.434*epsd/psd

    if fix_beta is None and prior_zt is None:

        theta0 = np.array([20., 1., 10., 2.])
        function = lambda k, A, zt, dz, beta: calculate_psd(k, A, zt, dz, beta)
        p1fit, p1cov = curve_fit(function, k, y_obs, p0=theta0, \
            sigma=y_err, absolute_sigma=True, max_nfev=1000, \
            bounds=([1., 0., 1., 0.], [30., 10., 50., 4.]))

        # Calculate best fit function
        pred = calculate_psd(k, p1fit[0], p1fit[1], p1fit[2], p1fit[3])
        
        # calculate reduced chi-square
        rchi2 = np.sum((pred - y_obs)**2\
                /y_err**2)/(len(pred)-len(p1fit))

    elif fix_beta is None and prior_zt is not None:

        theta0 = np.array([20., 10., 2.])
        function = lambda k, A, dz, beta: calculate_psd(k, A, prior_zt, dz, beta)
        p1fit, p1cov = curve_fit(function, k, y_obs, p0=theta0, \
            sigma=y_err, absolute_sigma=True, max_nfev=1000, \
            bounds=([1., 1., 0.], [30., 50., 4.]))

        # Calculate best fit function
        pred = calculate_psd(k, p1fit[0], prior_zt, p1fit[1], p1fit[2])
        
        # calculate reduced chi-square
        rchi2 = np.sum((pred - y_obs)**2\
                /y_err**2)/(len(pred)-len(p1fit))

    elif fix_beta is not None and prior_zt is None:

        theta0 = np.array([20., 1., 10.])
        function = lambda k, A, zt, dz: calculate_psd(k, A, zt, dz, beta=fix_beta)
        p1fit, p1cov = curve_fit(function, k, y_obs, p0=theta0, \
            sigma=y_err, absolute_sigma=True, max_nfev=1000, \
            bounds=([1., 0., 1.], [30., 10., 50.]))

        # Calculate best fit function
        pred = calculate_psd(k, p1fit[0], p1fit[1], p1fit[2], fix_beta)
        
        # calculate reduced chi-square
        rchi2 = np.sum((pred - y_obs)**2\
                /y_err**2)/(len(pred)-len(p1fit))

    else:

        theta0 = np.array([20., 10.])
        function = lambda k, A, dz: calculate_psd(k, A, prior_zt, dz, beta=fix_beta)
        p1fit, p1cov = curve_fit(function, k, y_obs, p0=theta0, \
            sigma=y_err, absolute_sigma=True, max_nfev=1000, \
            bounds=([1., 1.], [30., 50.]))

        # Calculate best fit function
        pred = calculate_psd(k, p1fit[0], prior_zt, p1fit[1], fix_beta)
        
        # calculate reduced chi-square
        rchi2 = np.sum((pred - y_obs)**2\
                /y_err**2)/(len(pred)-len(p1fit))

    p1err = np.sqrt(np.diag(p1cov))

    if fix_beta is None and prior_zt is None:
        # Store summary
        summary = pd.DataFrame(data={'mean':[p1fit[0], p1fit[1], p1fit[2], p1fit[3]], \
            'std':[p1err[0], p1err[1], p1err[2], p1err[3]], \
            'chi2':[rchi2, rchi2, rchi2, rchi2]}, \
            index=['A', 'zt', 'dz', 'beta'])

    elif fix_beta is None and prior_zt is not None:
        # Store summary
        summary = pd.DataFrame(data={'mean':[p1fit[0], p1fit[1], p1fit[2]], \
            'std':[p1err[0], p1err[1], p1err[2]], 'chi2':[rchi2, rchi2, rchi2]}, \
            index=['A', 'dz', 'beta'])

    elif fix_beta is not None and prior_zt is None:
        # Store summary
        summary = pd.DataFrame(data={'mean':[p1fit[0], p1fit[1], p1fit[2]], \
            'std':[p1err[0], p1err[1], p1err[2]], 'chi2':[rchi2, rchi2, rchi2]}, \
            index=['A', 'zt', 'dz'])

    else:
        summary = pd.DataFrame(data={'mean':[p1fit[0], p1fit[1]], \
            'std':[p1err[0], p1err[1]], 'chi2':[rchi2, rchi2]}, \
            index=['A', 'dz'])

    return summary


def get_L2_estimates(summary):
    """
    Returns digestible estimates from the L2 estimates.

    :type summary: :class:`~pandas.core.frame.DataFrame`
    :param summary: Summary statistics from Posterior distributions

    :return: 
        (tuple): tuple containing:
            * mean_A (float) : Mean value of the constant term ``A``
            * std_A (float)  : Standard deviation of the constant term ``A``
            * mean_dz (float)  : Mean value of the magnetic layer thickness ``dz``
            * std_dz (float)   : Standard deviation of magnetic layer thickness ``dz``
            * mean_zt (float, optional) : Mean value of the depth to top of layer ``zt``
            * std_zt (float, optional)  : Standard deviation of the depth to top of layer ``zt``
            * mean_beta (float, optional)  : Mean value of the fractal parameter ``beta``
            * std_beta (float, optional)   : Standard deviation of fractal parameter ``beta``
            * rchi2 (float)   : Reduced chi-squared value
    """

    mean_beta = None
    mean_zt = None

    # Go through all estimates
    for index, row in summary.iterrows():
        if index=='A':
            mean_A = row['mean']
            std_A = row['std']
            rchi2 = row['chi2']
        elif index=='zt':
            mean_zt = row['mean']
            std_zt = row['std']
        elif index=='dz':
            mean_dz = row['mean']
            std_dz = row['std']
        elif index=='beta':
            mean_beta = row['mean']
            std_beta = row['std']

    if mean_beta is not None and mean_zt is not None:
        return mean_A, std_A, mean_zt, std_zt, mean_dz, std_dz, mean_beta, std_beta, rchi2
    elif mean_beta is not None and mean_zt is None:
        return mean_A, std_A, mean_dz, std_dz, mean_beta, std_beta, rchi2
    elif mean_beta is None and mean_zt is not None:
        return mean_A, std_A, mean_zt, std_zt, mean_dz, std_dz, rchi2
    else:
        return mean_A, std_A, mean_dz, std_dz, rchi2

def calculate_psd(k, A, zt, dz, beta):
    """
    Calculate analytical expression for the power-spectral density function. 

    :type k: :class:`~numpy.ndarray`
    :param k: 1D array of wavenumbers
    :type A: float
    :param A: Constant (i.e., DC) value for PSD [1., 30.]
    :type zt: float
    :param zt: Depth to top of magnetized layer (km) [0., 10.]
    :type dz: float
    :param dz: Thickness of magnetized layer (km) [1., 50.]
    :type beta: float
    :param beta: Power-law exponent for fractal magnetization

    :return:  
        (:class:`~numpy.ndarray`): Power-spectral density function (shape ``len(k)``)

    :rubric: References

        Audet, P., and Gosselin, J.M. (2019). Curie depth estimation from magnetic 
        anomaly data: a re-assessment using multitaper spectral analysis and Bayesian 
        inference. Geophysical Journal International, 218, 494-507. 
        https://doi.org/10.1093/gji/ggz166

        Blakely, R.J., 1995. Potential Theory in Gravity and Magnetic Applications,
        Cambridge Univ. Press.

    """
    kk = k*1.e3

    # Theoretical equation for magnetized layer 
    return A - beta*np.log(kk) - 2.*kk*zt + 2.*np.log(1. - np.exp(-kk*dz))


def calculate_bouligand(k, A, zt, dz, beta):
    """
    Calculate the synthetic power spectral density of
    magnetic anomalies Equation (4) of Bouligand et al. (2009)
    
    :type k: :class:`~numpy.ndarray`
    :param k: 1D array of wavenumbers
    :type A: float
    :param A: Constant (i.e., DC) value for PSD [1., 30.]
    :type zt: float
    :param zt: Depth to top of magnetized layer (km) [0., 10.]
    :type dz: float
    :param dz: Thickness of magnetized layer (km) [1., 50.]
    :type beta: float
    :param beta: Power-law exponent for fractal magnetization

    :return:  
        (:class:`~numpy.ndarray`): Power-spectral density function (shape ``len(k)``)

    :rubric: References

        Bouligand, C., J. M. G. Glen, and R. J. Blakely (2009), Mapping Curie
        temperature depth in the western United States with a fractal model for
        crustal magnetization, J. Geophys. Res., 114, B11104,
        doi:10.1029/2009JB006494
        
    """
    # from scipy.special import kv
    kh = k*1.e3
    khdz = kh * dz
    coshkhdz = np.cosh(khdz)

    Phi1d = A - 2.0 * kh * zt - (beta - 1.0) * np.log(kh) - khdz
    C = (np.sqrt(np.pi)/gamma(1.0 + 0.5 * beta)*( \
        0.5 * coshkhdz * gamma(0.5 * (1.0 + beta)) - \
        kv((-0.5 * (1.0 + beta)), khdz) * np.power(0.5 * khdz, \
            (0.5 * (1.0 + beta)))))

    Phi1d += np.log(C)
    return Phi1d


@as_op(itypes=[tt.dvector, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar], 
    otypes=[tt.dvector])
def calculate_psd_theano(k, A, zt, dz, beta):
    """
    Calculate analytical expression for the power-spectral density function. 
    """

    return calculate_psd(k, A, zt, dz, beta)
