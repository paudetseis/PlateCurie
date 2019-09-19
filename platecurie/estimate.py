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

- :func:`~platecurie.estimate.set_model`: Set up :mod:`~pymc` model
- :func:`~platecurie.estimate.get_estimates`: Explore the output of sampling the :mod:`~pymc` model
- :func:`~platecurie.estimate.calculate_psd`: Calculate the analytical PSD function. 

Internal function is available to calculate predicted power-spectral density data
with a ``theano`` decorator to be incorporated as ``pymc`` variables. These functions are
used within :class:`~platecurie.classes.MagGrid` methods as with :mod:`~platecurie.plotting`
functions.

"""

# -*- coding: utf-8 -*-
import numpy as np
import pymc3 as pm
from platecurie import conf as cf
from theano.compile.ops import as_op
import theano.tensor as tt

def set_model(k, psd, epsd, fix_beta=None):
    """
    Function to set up a ``pymc3`` model using default bounds on the prior
    distribution of parameters and observed wavelet scalogram data. 
    Can incorporate 3 ('C', 'zt' and 'dz') or 4 ('C', 'zt', 'dz', and 'beta') 
    stochastic variables.

    :type k: :class:`~numpy.ndarray`
    :param k: 1D array of wavenumbers
    :type psd: :class:`~numpy.ndarray`
    :param psd: 1D array of wavelet admittance (wavelet PSD)
    :type epsd: :class:`~numpy.ndarray`
    :param epsd: 1D array of error on wavelet admittance (wavelet PSD)
    :type fix_beta: float, optional
    :param fix_beta: Fixed ``beta`` parameter or estimate it if ``None``

    :return: 
        New :class:`~pymc3.model.Model` object to estimate via sampling
    """

    with pm.Model() as model:

        # k is an array - needs to be passed as distribution
        k_obs = pm.Normal('k', mu=k, sigma=1., observed=k)

        # Prior distributions
        A = pm.Uniform('A', lower=1., upper=30.)
        zt = pm.Uniform('zt', lower=0., upper=10.)
        dz = pm.Uniform('dz', lower=1., upper=50.)

        if fix_beta is not None:

            # Pass `beta` variable as theano variable
            var_beta = tt.as_tensor_variable(np.float64(fix_beta))
            psd_exp = calculate_psd_theano(k_obs, A, zt, dz, var_beta)

        else:

            # Prior distribution of `beta`
            beta = pm.Uniform('beta', lower=0., upper=4.)
            psd_exp = calculate_psd_theano(k_obs, A, zt, dz, beta)

        # Uncertainty as observed distribution
        dpsd = 3.*0.434*epsd/psd

        sigma = pm.Normal('sigma', mu=epsd, sigma=1., \
            observed=dpsd)

        # Likelihood of observations
        psd_obs = pm.Normal('psd_obs', mu=psd_exp, \
            sigma=sigma, observed=np.log(psd))

    return model


def estimate_cell(k, psd, epsd, fix_beta=None):
    """
    Function to estimate the parameters of the flexural model at a single cell location
    of the input grids. 

    :type k: :class:`~numpy.ndarray`
    :param k: 1D array of wavenumbers
    :type psd: :class:`~numpy.ndarray`
    :param psd: 1D array of wavelet scalogram (wavelet PSD)
    :type epsd: :class:`~numpy.ndarray`
    :param epsd: 1D array of error on wavelet scalogram (wavelet PSD)
    :type fix_beta: float, optional
    :param fix_beta: Fixed ``beta`` parameter or estimate it if ``None``

    :return:
        (tuple): Tuple containing:
            * ``trace`` : :class:`~pymc3.backends.base.MultiTrace`
                Posterior samples from the MCMC chains
            * ``summary`` : :class:`~pandas.core.frame.DataFrame`
                Summary statistics from Posterior distributions
            * ``map_estimate`` : dict
                Container for Maximum a Posteriori (MAP) estimates

    """

    # Use model returned from function ``set_model``
    with set_model(k, psd, epsd, fix_beta):

        # Sample the Posterior distribution
        trace = pm.sample(cf.samples, tune=cf.tunes, cores=cf.cores)

        # Get Max a porteriori estimate
        map_estimate = pm.find_MAP()

        # Get Summary
        summary = pm.summary(trace).round(2)

    return trace, summary, map_estimate

def get_estimates(summary, map_estimate):
    """
    Extract useful estimates from the Posterior distributions.

    :type summary: :class:`~pandas.core.frame.DataFrame`
    :param summary: Summary statistics from Posterior distributions
    :type map_estimate: dict
    :param map_estimate: Container for Maximum a Posteriori (MAP) estimates

    :return: 
        (tuple): tuple containing:
            * mean_te (float) : Mean value of elastic thickness from posterior (km)
            * std_te (float)  : Standard deviation of elastic thickness from posterior (km)
            * best_te (float) : Most likely elastic thickness value from posterior (km)
            * mean_F (float)  : Mean value of load ratio from posterior
            * std_F (float)   : Standard deviation of load ratio from posterior
            * best_F (float)  : Most likely load ratio value from posterior

    .. rubric:: Example

    >>> from plateflex import estimate
    >>> # MAKE THIS FUNCTION FASTER

    """

    mean_beta = None

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

    if mean_beta is not None:
        return mean_A, std_A, C2_5_A, C97_5_A, best_A, \
            mean_zt, std_zt, C2_5_zt, C97_5_zt, best_zt, \
            mean_dz, std_dz, C2_5_dz, C97_5_dz, best_dz, \
            mean_beta, std_beta, C2_5_beta, C97_5_beta, best_beta
    else:
        return mean_A, std_A, C2_5_A, C97_5_A, best_A, \
            mean_zt, std_zt, C2_5_zt, C97_5_zt, best_zt, \
            mean_dz, std_dz, C2_5_dz, C97_5_dz, best_dz


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
