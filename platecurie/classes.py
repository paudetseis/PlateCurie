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

This :mod:`~platecurie` module contains the following ``Grid`` classe:

- :class:`~platecurie.classes.MagGrid`

These classes can be initiatlized with a grid of magnetic anomaly data, 
and contain methods inherited from :class:`~plateflex.classes.Grid` 
for the following functionality:

- Performing a wavelet transform using a Morlet wavelet
- Obtaining the wavelet scalogram from the wavelet transform
- Plotting the input grids, wavelet transform components, and scalograms

- Estimate model parameters at single grid cell
- Estimate model parameters at every (or decimated) grid cell
- Plot the statistics of the estimated parameters at single grid cell
- Plot the fitted admittance and coherence functions at single grid cell
- Plot the final grids of model parameters

"""

# -*- coding: utf-8 -*-
import numpy as np
from plateflex.cpwt import cpwt
from plateflex import conf as cf
from plateflex import Grid
from plateflex import plotting as flexplot
from platecurie import plotting as curieplot
from platecurie import estimate
import seaborn as sns
sns.set()


class MagGrid(Grid):
    """
    Basic grid class of ``platecurie`` for Magnetic anomaly data that inherits 
    from :class:`~plateflex.classes.Grid`

    Contains method to plot the grid data with default title and units using function 
    :func:`~plateflex.plotting.plot_real_grid`.

    .. rubric:: Additional Attributes

    ``height`` : float
        Height of measurement above surface (meters)
    ``units`` : str
        Units of Magnetic data ('nTesla')
    ``sg_units`` : str
        Units of wavelet PSD (scalogram) ('nTesla^2/|k|')
    ``logsg_units`` : str
        Units of log of wavelet PSD (log(scalogram)) ('log(nTesla^2/|k|)')
    ``title``: str
        Descriptor for Magnetic anomaly data

    .. rubric: Example
    
    >>> import numpy as np
    >>> from plateflex import Grid
    >>> from platecurie import MagGrid
    >>> nn = 200; dd = 10.
    >>> maggrid = MagGrid(np.random.randn(nn, nn), 0., (nn-1)*dd, 0., (nn-1)*dd)
    >>> isinstance(maggrid, Grid)
    True

    """

    def __init__(self, grid, dx, dy):

        super().__init__(grid, dx, dy)
        # self.height = height
        self.units = 'nTesla'
        self.sg_units = r'nTesla$^2$/|k|'
        self.logsg_units = r'log(nTesla$^2$/|k|)'
        self.title = 'Magnetic anomaly'


    def estimate_cell(self, cell=(0,0), fix_beta=None, returned=False):
        """
        Method to estimate the parameters of the magnetized layer at a single cell location
        of the input grid. 

        :type cell: tuple
        :param cell: Indices of cell location within grid
        :type fix_beta: float, optional
        :param fix_beta: If not None, fix ``beta`` parameter - otherwise estimate it
        :type returned: bool, optional
        :param returned: Whether to use to return the estimates

        .. rubric:: Additional Attributes

        ``trace`` : :class:`~pymc3.backends.base.MultiTrace`
            Posterior samples from the MCMC chains
        ``summary`` : :class:`~pandas.core.frame.DataFrame`
            Summary statistics from Posterior distributions
        ``map_estimate`` : dict
            Container for Maximum a Posteriori (MAP) estimates
        ``cell`` : tuple 
            Indices of cell location within grid
        
        Results are stored as attributes of :class:`~platecurie.classes.MagGrid` 
        object.
        """

        # Extract admittance and coherence at cell indices
        psd = self.wl_sg[cell[0], cell[1], :]
        epsd = self.ewl_sg[cell[0], cell[1], :]

        trace, summary, map_estimate = estimate.estimate_cell( \
            self.k, psd, epsd, fix_beta)

        # Return estimates if requested
        if returned:
            return summary, map_estimate

        # Otherwise store as object attributes
        else:
            self.fix_beta = fix_beta
            self.cell = cell
            self.trace = trace
            self.map_estimate = map_estimate
            self.summary = summary

    def estimate_grid(self, nn=10, fix_beta=None, parallel=False):
        """
        Method to estimate the parameters of the magnetized layer at all grid point locations.
        It is also possible to decimate the number of grid cells at which to estimate parameters. 

        :type nn: int
        :param nn: Decimator. If grid shape is ``(nx, ny)``, resulting grids will have 
            shape of ``(int(nx/nn), int(ny/nn))``. 
        :type fix_beta: float, optional
        :param fix_beta: If not None, fix ``beta`` parameter - otherwise estimate it

        .. rubric:: Additional Attributes

        ``mean_A_grid`` : :class:`~numpy.ndarray` 
            Grid with mean A estimates (shape ``(nx, ny``))
        ``MAP_A_grid`` : :class:`~numpy.ndarray` 
            Grid with MAP A estimates (shape ``(nx, ny``))
        ``std_A_grid`` : :class:`~numpy.ndarray` 
            Grid with std A estimates (shape ``(nx, ny``))
        ``mean_zt_grid`` : :class:`~numpy.ndarray` 
            Grid with mean zt estimates (shape ``(nx, ny``))
        ``MAP_zt_grid`` : :class:`~numpy.ndarray` 
            Grid with MAP zt estimates (shape ``(nx, ny``))
        ``std_zt_grid`` : :class:`~numpy.ndarray` 
            Grid with std zt estimates (shape ``(nx, ny``))
        ``mean_dz_grid`` : :class:`~numpy.ndarray` 
            Grid with mean dz estimates (shape ``(nx, ny``))
        ``MAP_dz_grid`` : :class:`~numpy.ndarray` 
            Grid with MAP dz estimates (shape ``(nx, ny``))
        ``std_dz_grid`` : :class:`~numpy.ndarray` 
            Grid with std dz estimates (shape ``(nx, ny``))

        .. rubric:: Optional Additional Attributes

        ``mean_beta_grid`` : :class:`~numpy.ndarray` 
            Grid with mean beta estimates (shape ``(nx, ny``))
        ``MAP_beta_grid`` : :class:`~numpy.ndarray` 
            Grid with MAP beta estimates (shape ``(nx, ny``))
        ``std_beta_grid`` : :class:`~numpy.ndarray` 
            Grid with std beta estimates (shape ``(nx, ny``))

        """

        # Import garbage collector
        import gc

        # Delete attributes to release some memory
        try:
            del self.mean_A_grid
            del self.MAP_A_grid
            del self.std_A_grid            
            del self.mean_zt_grid
            del self.MAP_zt_grid
            del self.std_zt_grid
            del self.mean_dz_grid
            del self.MAP_dz_grid
            del self.std_dz_grid
            try:        
                del self.mean_beta_grid
                del self.MAP_beta_grid
                del self.std_beta_grid
            except:
                print("parameter 'beta' was not previously estimated")
        except:
            pass

        self.fix_beta = fix_beta

        # Initialize result grids to zoroes
        mean_A_grid = np.zeros((int(self.ny/nn),int(self.nx/nn)))
        MAP_A_grid = np.zeros((int(self.ny/nn),int(self.nx/nn)))
        std_A_grid = np.zeros((int(self.ny/nn),int(self.nx/nn)))
        mean_zt_grid = np.zeros((int(self.ny/nn),int(self.nx/nn)))
        MAP_zt_grid = np.zeros((int(self.ny/nn),int(self.nx/nn)))
        std_zt_grid = np.zeros((int(self.ny/nn),int(self.nx/nn)))
        mean_dz_grid = np.zeros((int(self.ny/nn),int(self.nx/nn)))
        MAP_dz_grid = np.zeros((int(self.ny/nn),int(self.nx/nn)))
        std_dz_grid = np.zeros((int(self.ny/nn),int(self.nx/nn)))
        if self.fix_beta is None:
            mean_beta_grid = np.zeros((int(self.ny/nn),int(self.nx/nn)))
            MAP_beta_grid = np.zeros((int(self.ny/nn),int(self.nx/nn)))
            std_beta_grid = np.zeros((int(self.ny/nn),int(self.nx/nn)))

        if parallel:

            raise(Exception('Parallel implementation does not work - check again later'))
            # from joblib import Parallel, delayed
            # cf.cores=1

            # # Run nested for loop in parallel to cover the whole grid
            # results = Parallel(n_jobs=4)(delayed(self.estimate_cell) \
            #     (cell=(i,j), alph=alph, atype=atype, returned=True) \
            #     for i in range(0, self.nx-nn, nn) for j in range(0, self.ny-nn, nn))

        else:
            for i in range(0, self.ny-nn, nn):
                for j in range(0, self.nx-nn, nn):
                    
                    # For reference - index values
                    print(i,j)

                    # tuple of cell indices
                    cell = (i,j)

                    # Carry out calculations by calling the ``estimate_cell`` method
                    summary, map_estimate = self.estimate_cell(cell=cell, \
                        fix_beta=self.fix_beta, returned=True)

                    # Extract estimates from summary and map_estimate
                    res = estimate.get_estimates(summary, map_estimate)

                    # Distribute the parameters back to space
                    mean_A = res[0]; std_A = res[1]; MAP_A = res[4]
                    mean_zt = res[5]; std_zt = res[6]; MAP_zt = res[9]
                    mean_dz = res[10]; std_dz = res[11]; MAP_dz = res[14]
                    if self.fix_beta is None:
                        mean_beta = res[15]; std_beta = res[16]; MAP_beta = res[19]

                    # Store values in smaller arrays
                    mean_A_grid[int(i/nn),int(j/nn)] = mean_A
                    MAP_A_grid[int(i/nn),int(j/nn)] = MAP_A
                    std_A_grid[int(i/nn),int(j/nn)] = std_A
                    mean_zt_grid[int(i/nn),int(j/nn)] = mean_zt
                    MAP_zt_grid[int(i/nn),int(j/nn)] = MAP_zt
                    std_zt_grid[int(i/nn),int(j/nn)] = std_zt
                    mean_dz_grid[int(i/nn),int(j/nn)] = mean_dz
                    MAP_dz_grid[int(i/nn),int(j/nn)] = MAP_dz
                    std_dz_grid[int(i/nn),int(j/nn)] = std_dz
                    if self.fix_beta is None:
                        mean_beta_grid[int(i/nn),int(j/nn)] = mean_beta
                        MAP_beta_grid[int(i/nn),int(j/nn)] = MAP_beta
                        std_beta_grid[int(i/nn),int(j/nn)] = std_beta

                    # Release garbage collector
                    gc.collect()

        # Store grids as attributes
        self.mean_A_grid = mean_A_grid
        self.MAP_A_grid = MAP_A_grid
        self.std_A_grid = std_A_grid
        self.mean_zt_grid = mean_zt_grid
        self.MAP_zt_grid = MAP_zt_grid
        self.std_zt_grid = std_zt_grid
        self.mean_dz_grid = mean_dz_grid
        self.MAP_dz_grid = MAP_dz_grid
        self.std_dz_grid = std_dz_grid
        if self.fix_beta is None:
            self.mean_beta_grid = mean_beta_grid
            self.MAP_beta_grid = MAP_beta_grid
            self.std_beta_grid = std_beta_griod


    def plot_stats(self, title=None, save=None):
        """
        Method to plot the marginal and joint distributions of samples drawn from the 
        posterior distribution as well as the extracted statistics. Calls the function 
        :func:`~platecurie.plotting.plot_stats` with attributes as arguments.

        :type title: str, optional 
        :param title: Title of plot
        :type save: str, optional
        :param save: Name of file for to save figure

        """

        try:
            curieplot.plot_stats(self.trace, self.summary, \
                self.map_estimate, title=title, save=save)
        except:
            raise(Exception("No 'cell' estimate available"))


    def plot_fitted(self, est='MAP', title=None, save=None):
        """
        Method to plot observed and fitted admittance and coherence functions using 
        one of ``MAP`` or ``mean`` estimates. Calls the function :func:`~platecurie.plotting.plot_fitted`
        with attributes as arguments.

        :type est: str, optional
        :param est: Type of inference estimate to use for predicting admittance and coherence
        :type title: str, optional 
        :param title: Title of plot
        :type save: str, optional
        :param save: Name of file for to save figure
        
        """

        if est not in ['mean', 'MAP']:
            raise(Exception("Choose one among: 'mean', or 'MAP'"))
            
        try:
            cell = self.cell
            k = self.k
            psd = self.wl_sg[cell[0], cell[1], :]
            epsd = self.ewl_sg[cell[0], cell[1], :]

            # Call function from ``plotting`` module
            curieplot.plot_fitted(k, psd, epsd, self.summary, \
                self.map_estimate, fix_beta=self.fix_beta, \
                est=est, title=title, save=save)

        except:
            raise(Exception("No estimate yet available"))


    def plot_results(self, mean_A=False, MAP_A=False, std_A=False, \
        mean_zt=False, MAP_zt=False, std_zt=False, mean_dz=False, MAP_dz=False, \
        std_dz=False, mean_beta=False, MAP_beta=False, std_beta=False, \
        mask=None, save=None, **kwargs):
        """
        Method to plot grids of estimated parameters with fixed labels and titles. 
        To have more control over the plot rendering, use the function 
        :func:`~plateflex.plotting.plot_real_grid` with the relevant quantities and 
        plotting options.

        :type mean/MAP/std_A/zt/dz/beta: bool
        :param mean/MAP/std_A/zt/dz/beta: Type of plot to produce. 
            All variables default to False (no plot generated)
        """

        if mean_A:
            flexplot.plot_real_grid(self.mean_A_grid, mask=mask, \
                title='Mean of posterior', clabel='$A$', save=save, **kwargs)
        if MAP_A:
            flexplot.plot_real_grid(self.MAP_A_grid, mask=mask, \
                title='MAP estimate', clabel='$A$', save=save, **kwargs)
        if std_A:
            flexplot.plot_real_grid(self.std_A_grid, mask=mask, \
                title='Std of posterior', clabel='$A$', save=save, **kwargs)
        if mean_zt:
            flexplot.plot_real_grid(self.mean_zt_grid, mask=mask, \
                title='Mean of posterior', clabel='$z_t$ (km)', save=save, **kwargs)
        if MAP_zt:
            flexplot.plot_real_grid(self.MAP_zt_grid, mask=mask, \
                title='MAP estimate', clabel='$z_t$ (km)', save=save, **kwargs)
        if std_zt:
            flexplot.plot_real_grid(self.std_zt_grid, mask=mask, \
                title='Std of posterior', clabel='$z_t$ (km)', save=save, **kwargs)
        if mean_dz:
            flexplot.plot_real_grid(self.mean_dz_grid, mask=mask, \
                title='Mean of posterior', clabel='$dz$ (km)', save=save, **kwargs)
        if MAP_dz:
            flexplot.plot_real_grid(self.MAP_dz_grid, mask=mask, \
                title='MAP estimate', clabel='$dz$ (km)', save=save, **kwargs)
        if std_dz:
            flexplot.plot_real_grid(self.std_dz_grid, mask=mask, \
                title='Std of posterior', clabel='$dz$ (km)', save=save, **kwargs)
        if mean_beta:
            try:
                flexplot.plot_real_grid(self.mean_beta_grid, mask=mask, \
                    title='Mean of posterior', clabel=r'$\beta$', save=save, **kwargs)
            except:
                print("parameter 'beta' was not estimated")
        if MAP_beta:
            try:
                flexplot.plot_real_grid(self.MAP_beta_grid, mask=mask, \
                    title='MAP estimate', clabel=r'$\beta$', save=save, **kwargs)
            except:
                print("parameter 'beta' was not estimated")
        if std_beta:
            try:
                flexplot.plot_real_grid(self.std_beta_grid, mask=mask, \
                    title='Std of posterior', clabel=r'$\beta$', save=save, **kwargs)
            except:
                print("parameter 'beta' was not estimated")


