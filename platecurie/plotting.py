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
This :mod:`~platecurie`` module contains the following functions for plotting:

- :func:`~platecurie.plotting.plot_stats`
- :func:`~platecurie.plotting.plot_fitted`

"""

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde as kde
import seaborn as sns
sns.set()


def plot_stats(trace, summary, map_estimate, title=None, save=None):
    """
    Extract results from variables ``trace``, ``summary`` and ``map_estimate`` to 
    plot marginal and joint posterior distributions. Automatically determines
    how to plot results from those variables.

    :type trace: :class:`~pymc3.backends.base.MultiTrace`
    :param trace: Posterior samples from the MCMC chains
    :type summary: :class:`~pandas.core.frame.DataFrame`
    :param summary: Summary statistics from Posterior distributions
    :type map_estimate: dict
    :param map_estimate: Container for Maximum a Posteriori (MAP) estimates
    :type title: str, optional 
    :param title: Title of plot
    :type save: str, optional
    :param save: Name of file for to save figure

    """

    from platecurie import estimate

    # Extract results from summary and map_estimate
    results = estimate.get_bayes_estimates(summary, map_estimate)

    # Collect keys in trace object
    keys = []
    for var in trace.varnames:
        if var[-1]=='_':
            continue
        keys.append(var)

    # This means we searched for A and dz only
    if len(keys)==2:

        # Collect pymc chains as ``pandas.DataFrame`` object
        data = np.array([trace['A'], trace['dz']]).transpose()
        data = pd.DataFrame(data, columns=[r'$A$', r'$dz$'])

        # Plot marginal and joint distributions as histograms and kernel density functions
        g = sns.PairGrid(data)
        g.map_diag(plt.hist, lw=1)
        g.map_lower(sns.kdeplot)

        # Set unused plot axes to invisible
        ax = g.axes[0][1]
        ax.set_visible(False)

        # Text for A statistics
        Atext = '\n'.join((
            r'$\mu$ = {0:.0f}'.format(results[0]),
            r'$\sigma$ = {0:.0f}'.format(results[1]),
            r'$95\%$ CI = [{0:.0f}, {1:.0f}]'.format(results[2], results[3]),
            r'MAP = {0:.0f}'.format(results[4])))

        # Insert text as box
        ax1 = g.axes[0][0]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(1.05, 0.9, Atext, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

        # Text for dz statistics
        dztext = '\n'.join((
            r'$\mu$ = {0:.1f} km'.format(results[5]),
            r'$\sigma$ = {0:.1f} km'.format(results[6]),
            r'$95\%$ CI = [{0:.1f}, {1:.1f}] km'.format(results[7], results[8]),
            r'MAP = {0:.1f} km'.format(results[9])))

        # Insert text as box
        ax2 = g.axes[1][1]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.135, 1.4, dztext, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

    # This means we searched for A, zt and dz
    elif 'zt' in keys and 'beta' not in keys:

        # Collect pymc chains as ``pandas.DataFrame`` object
        data = np.array([trace['A'], trace['zt'], trace['dz']]).transpose()
        data = pd.DataFrame(data, columns=[r'$A$', r'$z_t$ (km)', r'$dz$ (km)'])

        # Plot marginal and joint distributions as histograms and kernel density functions
        g = sns.PairGrid(data)
        g.map_diag(plt.hist, lw=1)
        g.map_lower(sns.kdeplot)

        # Set unused plot axes to invisible
        ax = g.axes[0][1]
        ax.set_visible(False)
        ax = g.axes[0][2]
        ax.set_visible(False)
        ax = g.axes[1][2]
        ax.set_visible(False)

        # Text for A statistics
        Atext = '\n'.join((
            r'$\mu$ = {0:.0f}'.format(results[0]),
            r'$\sigma$ = {0:.0f}'.format(results[1]),
            r'$95\%$ CI = [{0:.0f}, {1:.0f}]'.format(results[2], results[3]),
            r'MAP = {0:.0f}'.format(results[4])))

        # Insert text as box
        ax1 = g.axes[0][0]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(1.05, 0.9, Atext, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

        # Text for zt statistics
        zttext = '\n'.join((
            r'$\mu$ = {0:.1f} km'.format(results[5]),
            r'$\sigma$ = {0:.1f} km'.format(results[6]),
            r'$95\%$ CI = [{0:.1f}, {1:.1f}] km'.format(results[7], results[8]),
            r'MAP = {0:.1f} km'.format(results[9])))

        # Insert text as box
        ax2 = g.axes[1][1]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.135, 1.4, zttext, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

        # Text for alpha statistics
        dztext = '\n'.join((
            r'$\mu$ = {0:.1f} km'.format(results[10]),
            r'$\sigma$ = {0:.1f} km'.format(results[11]),
            r'$95\%$ CI = [{0:.1f}, {1:.1f}] km'.format(results[12], results[13]),
            r'MAP = {0:.1f} km'.format(results[14])))

        # Insert text as box
        ax3 = g.axes[2][2]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax3.text(0.135, 1.4, dztext, transform=ax3.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

    # This means we searched for A, zt and dz
    elif 'zt' not in keys and 'beta' in keys:

        # Collect pymc chains as ``pandas.DataFrame`` object
        data = np.array([trace['A'], trace['dz'], trace['beta']]).transpose()
        data = pd.DataFrame(data, columns=[r'$A$', r'$dz$ (km)', r'$\beta$'])

        # Plot marginal and joint distributions as histograms and kernel density functions
        g = sns.PairGrid(data)
        g.map_diag(plt.hist, lw=1)
        g.map_lower(sns.kdeplot)

        # Set unused plot axes to invisible
        ax = g.axes[0][1]
        ax.set_visible(False)
        ax = g.axes[0][2]
        ax.set_visible(False)
        ax = g.axes[1][2]
        ax.set_visible(False)

        # Text for A statistics
        Atext = '\n'.join((
            r'$\mu$ = {0:.0f}'.format(results[0]),
            r'$\sigma$ = {0:.0f}'.format(results[1]),
            r'$95\%$ CI = [{0:.0f}, {1:.0f}]'.format(results[2], results[3]),
            r'MAP = {0:.0f}'.format(results[4])))

        # Insert text as box
        ax1 = g.axes[0][0]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(1.05, 0.9, Atext, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

        # Text for alpha statistics
        dztext = '\n'.join((
            r'$\mu$ = {0:.1f} km'.format(results[5]),
            r'$\sigma$ = {0:.1f} km'.format(results[6]),
            r'$95\%$ CI = [{0:.1f}, {1:.1f}] km'.format(results[7], results[8]),
            r'MAP = {0:.1f} km'.format(results[9])))

        # Insert text as box
        ax2 = g.axes[1][1]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.135, 1.4, dztext, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

        # Text for beta statistics
        btext = '\n'.join((
            r'$\mu$ = {0:.1f}'.format(results[10]),
            r'$\sigma$ = {0:.1f}'.format(results[11]),
            r'$95\%$ CI = [{0:.1f}, {1:.1f}]'.format(results[12], results[13]),
            r'MAP = {0:.1f}'.format(results[14])))

        # Insert text as box
        ax3 = g.axes[2][2]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax3.text(0.135, 1.4, btext, transform=ax3.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

    # This means we searched for A, zt, dz and beta
    elif len(keys)==4:

        # Collect pymc chains as ``pandas.DataFrame`` object
        data = np.array([trace['A'], trace['zt'], trace['dz'], trace['beta']]).transpose()
        data = pd.DataFrame(data, columns=[r'$A$', r'$z_t$ (km)', r'$dz$ (km)', r'$\beta$'])

        # Plot marginal and joint distributions as histograms and kernel density functions
        g = sns.PairGrid(data)
        g.map_diag(plt.hist, lw=1)
        g.map_lower(sns.kdeplot)

        # Set unused plot axes to invisible
        ax = g.axes[0][1]
        ax.set_visible(False)
        ax = g.axes[0][2]
        ax.set_visible(False)
        ax = g.axes[0][3]
        ax.set_visible(False)
        ax = g.axes[1][2]
        ax.set_visible(False)
        ax = g.axes[1][3]
        ax.set_visible(False)
        ax = g.axes[2][3]
        ax.set_visible(False)

        # Text for A statistics
        Atext = '\n'.join((
            r'$\mu$ = {0:.0f}'.format(results[0]),
            r'$\sigma$ = {0:.0f}'.format(results[1]),
            r'$95\%$ CI = [{0:.0f}, {1:.0f}]'.format(results[2], results[3]),
            r'MAP = {0:.0f}'.format(results[4])))

        # Insert text as box
        ax1 = g.axes[0][0]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(1.05, 0.9, Atext, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

        # Text for zt statistics
        zttext = '\n'.join((
            r'$\mu$ = {0:.1f} km'.format(results[5]),
            r'$\sigma$ = {0:.1f} km'.format(results[6]),
            r'$95\%$ CI = [{0:.1f}, {1:.1f}] km'.format(results[7], results[8]),
            r'MAP = {0:.1f} km'.format(results[9])))

        # Insert text as box
        ax2 = g.axes[1][1]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.135, 1.4, zttext, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

        # Text for dz statistics
        dztext = '\n'.join((
            r'$\mu$ = {0:.1f} km'.format(results[10]),
            r'$\sigma$ = {0:.1f} km'.format(results[11]),
            r'$95\%$ CI = [{0:.1f}, {1:.1f}] km'.format(results[12], results[13]),
            r'MAP = {0:.1f} km'.format(results[14])))

        # Insert text as box
        ax3 = g.axes[2][2]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax3.text(0.135, 1.4, dztext, transform=ax3.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

        # Text for beta statistics
        btext = '\n'.join((
            r'$\mu$ = {0:.1f}'.format(results[15]),
            r'$\sigma$ = {0:.1f}'.format(results[16]),
            r'$95\%$ CI = [{0:.1f}, {1:.1f}]'.format(results[17], results[18]),
            r'MAP = {0:.1f}'.format(results[19])))

        # Insert text as box
        ax4 = g.axes[3][3]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax4.text(0.135, 1.4, btext, transform=ax4.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

    else:
        raise(Exception('there are less than 3 or more than 4 variables in pymc3 chains'))

    # Plot title if requested
    if title is not None:
        plt.suptitle(title)

    # Save figure
    if save is not None:
        plt.savefig(save+'.png')

    plt.show()


def plot_functions(k, psd, epsd, ppsd=None, title=None, save=None):
    """
    Function to plot observed and fitted PSD function using 
    one of ``MAP`` or ``mean`` estimates. The observed PSD function is plotted
    regardless of method to estimate the model paramters. 

    :type k: :class:`~numpy.ndarray`
    :param k: 1D array of wavenumbers
    :type psd: :class:`~numpy.ndarray`
    :param psd: 1D array of wavelet scalogram (wavelet PSD)
    :type epsd: :class:`~numpy.ndarray`
    :param epsd: 1D array of error on wavelet scalogram (wavelet PSD)
    :type ppsd: :class:`~numpy.ndarray`, optional
    :param ppsd: 1D array of predicted PSD
    :type title: str, optional 
    :param title: Title of plot
    :type save: str, optional
    :param save: Name of file for to save figure
    """

    # Plot as subplot
    f, ax = plt.subplots(1, 1)

    # Plot observed PSD with error bars
    ax.errorbar(k*1.e3, np.log(psd), yerr=3.*0.434*epsd/psd)

    if ppsd is not None:        
        # Plot predicted PSD
        ax.plot(k*1.e3, ppsd)

    # Add all labels
    ax.set_ylabel('Power spectral density log(nT$^2$/|k|)')
    ax.set_xscale('log')

    # Plot title if requested
    if title is not None:
        plt.suptitle(title)

    # Save figure
    if save is not None:
        plt.savefig(save+'.png')

    plt.show()
