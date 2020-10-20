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


"""

# -*- coding: utf-8 -*-
import sys
import numpy as np
from plateflex.cpwt import cpwt
from plateflex import Grid
from plateflex import plotting as flexplot
from platecurie import plotting as curieplot
from platecurie import estimate


class MagGrid(Grid):
    """
    Basic grid class of :mod:`~platecurie` for Magnetic anomaly data that inherits 
    from :class:`~plateflex.classes.Grid`

    Contains method to plot the grid data with default title and units using function 
    :func:`~plateflex.plotting.plot_real_grid`.

    .. rubric:: Additional Attributes

    ``units`` : str
        Units of Magnetic data (':math:`nTesla`')
    ``sg_units`` : str
        Units of wavelet PSD (scalogram) (':math:`nTesla^2/|k|`')
    ``logsg_units`` : str
        Units of log of wavelet PSD (log(scalogram)) (':math:`log(nTesla^2/|k|)`')
    ``title``: str
        Descriptor for Magnetic anomaly data

    .. rubric: Example
    
    >>> import numpy as np
    >>> from plateflex import Grid
    >>> from platecurie import MagGrid
    >>> maggrid = MagGrid(np.zeros((100, 100)), 10, 10)
    >>> isinstance(maggrid, Grid)
    True

    """

    def __init__(self, grid, dx, dy):

        super().__init__(grid, dx, dy)
        # self.height = height
        self.units = 'nT'
        self.sg_units = r'nT$^2$/|k|'
        self.logsg_units = r'log(nT$^2$/|k|)'
        self.title = 'Magnetic anomaly'

class ZtGrid(Grid):
    """
    Basic grid class of :mod:`~platecurie` for the top of the magnetic layer that inherits 
    from :class:`~plateflex.classes.Grid`

    Contains method to plot the grid data with default title and units using function 
    :func:`~plateflex.plotting.plot_real_grid`.

    .. rubric:: Additional Attributes

    ``units`` : str
        Units of depth ('m')
    ``title``: str
        Descriptor for Depth to top of magnetic layer

    .. note::

        This class should only be used to specify the depth to the top of the magnetic
        layer at each cell location obtained from independent data. 
        Although the :class:`~plateflex.classes.Grid`
        methods are still available, they are not useful in this context.

    """

    def __init__(self, grid, dx, dy):

        super().__init__(grid, dx, dy)
        self.units = 'm'
        self.sg_units = None
        self.logsg_units = None
        self.title = r'Top of magnetic layer ($z_t$)'

class SigZtGrid(Grid):
    """
    Basic grid class of :mod:`~platecurie` for the estimated uncertainty in top
    of magnetic layer that inherits from :class:`~plateflex.classes.Grid`

    Contains method to plot the grid data with default title and units using function 
    :func:`~plateflex.plotting.plot_real_grid`.

    .. rubric:: Additional Attributes

    ``units`` : str
        Units of depth ('m')
    ``title``: str
        Descriptor for Depth to top of magnetic layer

    .. note::

        This class should only be used to specify the depth to the top of the magnetic
        layer at each cell location. Although the :class:`~plateflex.classes.Grid`
        methods are still available, they are not useful in this context.

    """

    def __init__(self, grid, dx, dy):

        super().__init__(grid, dx, dy)
        self.units = 'm'
        self.sg_units = None
        self.logsg_units = None
        self.title = r'Uncertainty in $z_t$'

        if np.std(self.data) < 10.:
            self.data *= 1.e3

class Project(object):
    """
    Container for :class:`~platecurie.classes.MagGrid` and/or 
    :class:`~platecurie.classes.ZtGrid` objects, with
    methods to estimate model parameters of the magnetic layer
    as well as plot various results. 

    :type grids: list of :class:`~plateflex.classes.Grid`, optional
    :param grids: Initial list of grid objects.

    .. rubric:: Default Attributes

    ``grids`` : List
        List of :class:`~platecurie.classes.MagGrid` objects
    ``mask`` : Array
        2D array of boolean values determined independently
    ``initialized`` : Bool
        Whether or not the project has been initialized and is ready for 
        the estimation steps. By default this parameter is ``False``, 
        unless the method :func:`~platecurie.classes.Project.init` 
        has been executed.

    .. note::

        A Project can hold a list of any length with any type of 
        :class:`~plateflex.classes.Grid` or those defined in 
        :mod:`~platecurie` - however the estimation 
        will only proceed if the project holds exactly one 
        :class:`~platecurie.classes.MagGrid` object. 

    .. rubric:: Examples

    """

    def __init__(self, grids=None):

        self.grids = []
        self.mask = None
        self.initialized = False

        if isinstance(grids, Grid):
            grids = [grids]
        if grids:
            self.grids.extend(grids)

    def __add__(self, other):
        """
        Add two grid objects or a 
        :class:`~platecurie.classes.Project` object with a single grid.

        """
        if isinstance(other, Grid):
            other = Project([other])
        if not isinstance(other, Project):
            raise TypeError
        grids = self.grids + other.grids
        return self.__class__(grids=grids)

    def __iter__(self):
        """
        Return a robust iterator for grid objects in Project

        """
        return list(self.grids).__iter__()

    def append(self, grid):
        """
        Append a single grid object to the 
        current `:class:`~platecurie.classes.Project` object.

        :type grid: :class:`~plateflex.classes.Grid`
        :param grid: object to append to project

        .. rubric:: Example
            
        >>> import numpy as np
        >>> from platecurie import MagGrid, Project
        >>> nn = 200; dd = 10.
        >>> grid = MagGrid(np.random.randn(nn, nn), dd, dd)
        >>> project = Project()
        >>> project.append(grid)
        """
        
        if isinstance(grid, Grid):
            self.grids.append(grid)
        else:
            msg = 'Append only supports a single Grid object as an argument.'
            raise TypeError(msg)

        return self

    def extend(self, grid_list):
        """
        Extend the current Project object with a list of Grid objects.

        :param trace_list: list of grid objects or
            :class:`~platecurie.classes.Project`.

        .. rubric:: Example

        >>> import numpy as np
        >>> from platecurie import MagGrid, ZtGrid, Project
        >>> nn = 200; dd = 10.
        >>> maggrid = MagGrid(np.random.randn(nn, nn), dd, dd)
        >>> ztgrid = ZtGrid(np.random.randn(nn, nn), dd, dd)
        >>> project = Project()
        >>> project.extend(grids=[maggrid, ztgrid])

        """
        if isinstance(grid_list, list):
            for _i in grid_list:
                # Make sure each item in the list is a Grid object.
                if not isinstance(_i, Grid):
                    msg = 'Extend only accepts a list of Grid objects.'
                    raise TypeError(msg)
            self.grids.extend(grid_list)
        elif isinstance(grid_list, Project):
            self.grids.extend(grid_list.grids)
        else:
            msg = 'Extend only supports a list of Grid objects as argument.'
            raise TypeError(msg)
        return self


    def init(self, inverse='L2'):
        """
        Method to initialize a project. This step is required before estimating the 
        model parameters. The method checks that the project contains
        exactly one :class:`~platecurie.classes.MagGrid`. 
        It also ensures that all grids have the same shape and sampling intervals. 
        If a grid of type :class:`~platecurie.classes.ZtGrid` (and possibly 
        :class:`~platecurie.classes.SigZtGrid`)
        is present, the project attributes will be updated with data from the grid to be
        used in the estimation part.  

        :type: str, optional
        :param: Type of inversion to perform. By default the type is `'L2'` for non-linear least-squares. Options are: `'L2'` or `'bayes'`

        .. rubric:: Additional Attributes

        ``nx`` : int 
            Number of grid cells in the x-direction
        ``ny`` : int 
            Number of grid cells in the y-direction
        ``ns`` int 
            Number of wavenumber samples
        ``k`` : np.ndarray 
            1D array of wavenumbers
        ``initialized`` : bool
            Set to ``True`` when method is called successfully
            
        .. rubric:: Optional Attributes

        ``zt`` : :class:`~numpy.ndarray`
            Grid of depth to top of magnetic anomaly (km) (shape (`nx,ny`))
        ``szt`` : :class:`~numpy.ndarray`
            Grid of uncertainty in depth to top of magnetic anomaly (km) (shape (`nx,ny`))

        .. rubric:: Example

        >>> from platecurie import MagGrid, ZtGrid, Project
        >>> nn = 200; dd = 10.
        >>> maggrid = MagGrid(np.random.randn(nn, nn), dd, dd)
        >>> ztgrid = ZtGrid(np.random.randn(nn, nn), dd, dd)
        >>> project = Project[grids=[maggrid, ztgrid]]
        >>> project.init()

        """

        self.inverse = inverse

        # Methods will fail if there is no ``MagGrid`` object in list
        if not any(isinstance(g, MagGrid) for g in self.grids):
            raise(Exception('There needs to be one MagGrid object in Project'))

        # Abort if there is more than one MagGrid
        maggrid = [grid for grid in self.grids if isinstance(grid, MagGrid)]
        if not (len(maggrid)==1):
            raise(Exception('There is more than one MagGrid in Project - aborting'))
        maggrid = maggrid[0]

        # Check that all grids have the same shape 
        shape = [grid.data.shape for grid in self.grids]
        if not (len(set(shape))==1):
            raise(Exception('Grids do not have the same shape - aborting:'+str(shape)))

        # Check that all grids have the same sampling intervals
        dd = [(grid.dx, grid.dy) for grid in self.grids]
        if not (len(set(dd))==1):
            raise(Exception('Grids do not have the same sampling intervals - aborting:'+str(dd)))

        # Check that wavelet scalogram coefficients exist
        try:
            psd = maggrid.wl_sg
        except:
            print('Wavelet scalogram ')
            maggrid.wlet_scalogram()

        # # Initialize model attributes to None (i.e., default values will be used)
        self.zt = None
        self.szt = None

        # Identify the ``Grid`` types and set new attributes if available
        for grid in self.grids:
            if isinstance(grid, MagGrid):
                try:
                    self.wl_sg = grid.wl_sg
                    self.ewl_sg = grid.ewl_sg
                except:
                    grid.wlet_scalogram()
                    self.wl_sg = grid.wl_sg
                    self.ewl_sg = grid.ewl_sg
            if isinstance(grid, ZtGrid):
                self.zt = grid.data
            if isinstance(grid, SigZtGrid):
                self.szt = grid.data

        if self.inverse == 'bayes':
            if self.zt is None and self.szt is None:
                print('Initialization: estimating zt using uniform (uniformative) prior')
            elif self.zt is not None and self.szt is not None:
                print('Inititilization: using log-normal distribution with parameters zt and szt as prior for zt')
                self.zt_prior = 'lognormal'
            elif self.zt is not None and self.szt is None:
                print('Initialization Warning: zt is set but szt is not. Using zt as max range of uniform prior')
            else:
                print('Initialization Warning: szt is set but zt is not. Estimating zt using uniform (uniformative) prior')
        elif self.inverse == 'L2':
            if self.zt is not None:
                print('Initialization: Using zt as fixed parameter')
            else:
                print('Initialization: success')

        # Now set project attributes from first grid object
        self.k = self.grids[0].k
        self.ns = self.grids[0].ns
        self.nx = self.grids[0].nx
        self.ny = self.grids[0].ny
        self.dx = self.grids[0].dx
        self.dy = self.grids[0].dy
        self.initialized = True

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
        
        Results are stored as attributes of :class:`~platecurie.classes.Project` 
        object.

        """

        if not self.initialized: 
            raise(Exception('Project not initialized. Aborting'))

        # Set model parameters if they are availble as attributes
        self.fix_beta = fix_beta
        if self.zt is not None:
            self.zt_cell = self.zt[cell[0], cell[1]]
        else:
            self.zt_cell = None

        # Extract PSD and error at cell indices
        psd = self.wl_sg[cell[0], cell[1], :]
        epsd = self.ewl_sg[cell[0], cell[1], :]

        if self.inverse=='L2':
            summary = estimate.L2_estimate_cell( \
                self.k, psd, epsd, fix_beta=fix_beta, prior_zt=self.zt_cell)

            # Return estimates if requested
            if returned:
                return summary

            # Otherwise store as object attributes
            else:
                self.cell = cell
                self.summary = summary

        elif self.inverse=='bayes':

            trace, summary, map_estimate = estimate.bayes_estimate_cell( \
                self.k, psd, epsd, fix_beta=fix_beta, prior_zt=self.zt_cell)

            # Return estimates if requested
            if returned:
                return summary, map_estimate

            # Otherwise store as object attributes
            else:
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
        ``mean_dz_grid`` : :class:`~numpy.ndarray` 
            Grid with mean dz estimates (shape ``(nx, ny``))
        ``MAP_dz_grid`` : :class:`~numpy.ndarray` 
            Grid with MAP dz estimates (shape ``(nx, ny``))
        ``std_dz_grid`` : :class:`~numpy.ndarray` 
            Grid with std dz estimates (shape ``(nx, ny``))

        .. rubric:: Optional Additional Attributes

        ``mean_zt_grid`` : :class:`~numpy.ndarray` 
            Grid with mean zt estimates (shape ``(nx, ny``))
        ``MAP_zt_grid`` : :class:`~numpy.ndarray` 
            Grid with MAP zt estimates (shape ``(nx, ny``))
        ``std_zt_grid`` : :class:`~numpy.ndarray` 
            Grid with std zt estimates (shape ``(nx, ny``))
        ``mean_beta_grid`` : :class:`~numpy.ndarray` 
            Grid with mean beta estimates (shape ``(nx, ny``))
        ``MAP_beta_grid`` : :class:`~numpy.ndarray` 
            Grid with MAP beta estimates (shape ``(nx, ny``))
        ``std_beta_grid`` : :class:`~numpy.ndarray` 
            Grid with std beta estimates (shape ``(nx, ny``))

        """

        self.nn = nn

        if fix_beta is not None and not isinstance(fix_beta, float):
            raise(Exception("'fix_beta' should be a float: defaults to None"))
        self.fix_beta = fix_beta

        # Initialize result grids to zeroes
        if self.mask is not None:
            new_mask_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)), dtype=bool)

        mean_A_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
        std_A_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
        mean_dz_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
        std_dz_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
        if self.zt is None:
            mean_zt_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
            std_zt_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
        if self.fix_beta is None:
            mean_beta_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
            std_beta_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))

        if self.inverse == 'bayes':
            MAP_A_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
            MAP_dz_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
            if self.zt is None:
                MAP_zt_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))
            if self.fix_beta is None:
                MAP_beta_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))

        elif self.inverse == 'L2':
            chi2_grid = np.zeros((int(self.nx/nn),int(self.ny/nn)))

        else:
            raise(Exception('Inverse method invalid: '+str(self.inverse)))

        if parallel:

            raise(Exception('Parallel implementation does not work - check again later'))
            # from joblib import Parallel, delayed
            # cf.cores=1

            # # Run nested for loop in parallel to cover the whole grid
            # results = Parallel(n_jobs=4)(delayed(self.estimate_cell) \
            #     (cell=(i,j), alph=alph, atype=atype, returned=True) \
            #     for i in range(0, self.nx-nn, nn) for j in range(0, self.ny-nn, nn))

        else:
            for i in _progressbar(range(0, self.nx-nn, nn), 'Computing: ', 10):
                for j in range(0, self.ny-nn, nn):
                    
                    # # For reference - index values
                    # print(i,j)

                    # tuple of cell indices
                    cell = (i,j)

                    # Skip masked cells
                    if self.mask is not None:
                        new_mask_grid[int(i/nn),int(j/nn)] = self.mask[i,j]
                        if self.mask[i,j]:
                            continue

                    if self.inverse=='bayes':

                        # Carry out calculations by calling the ``estimate_cell`` method
                        summary, map_estimate = self.estimate_cell(cell=cell, \
                            fix_beta=self.fix_beta, returned=True)

                        # Extract estimates from summary and map_estimate
                        res = estimate.get_bayes_estimates(summary, map_estimate)

                        # Distribute the parameters back to space
                        mean_A_grid[int(i/nn),int(j/nn)] = res[0]
                        std_A_grid[int(i/nn),int(j/nn)] = res[1]
                        MAP_A_grid[int(i/nn),int(j/nn)] = res[4]

                        if self.fix_beta is None and self.zt is None:
                            mean_zt_grid[int(i/nn),int(j/nn)] = res[5]
                            std_zt_grid[int(i/nn),int(j/nn)] = res[6]
                            MAP_zt_grid[int(i/nn),int(j/nn)] = res[9]
                            mean_dz_grid[int(i/nn),int(j/nn)] = res[10]
                            std_dz_grid[int(i/nn),int(j/nn)] = res[11]
                            MAP_dz_grid[int(i/nn),int(j/nn)] = res[14]
                            mean_beta_grid[int(i/nn),int(j/nn)] = res[15]
                            MAP_beta_grid[int(i/nn),int(j/nn)] = res[16]
                            std_beta_grid[int(i/nn),int(j/nn)] = res[19]
                        elif self.fix_beta is not None and self.zt is None:
                            mean_zt_grid[int(i/nn),int(j/nn)] = res[5]
                            std_zt_grid[int(i/nn),int(j/nn)] = res[6]
                            MAP_zt_grid[int(i/nn),int(j/nn)] = res[9]
                            mean_dz_grid[int(i/nn),int(j/nn)] = res[10]
                            std_dz_grid[int(i/nn),int(j/nn)] = res[11]
                            MAP_dz_grid[int(i/nn),int(j/nn)] = res[14]
                        elif self.fix_beta is None and self.zt is not None:
                            mean_dz_grid[int(i/nn),int(j/nn)] = res[5]
                            std_dz_grid[int(i/nn),int(j/nn)] = res[6]
                            MAP_dz_grid[int(i/nn),int(j/nn)] = res[9]
                            mean_beta_grid[int(i/nn),int(j/nn)] = res[10]
                            std_beta_grid[int(i/nn),int(j/nn)] = res[11]
                            MAP_beta_grid[int(i/nn),int(j/nn)] = res[14]
                        else:
                            mean_dz_grid[int(i/nn),int(j/nn)] = res[5]
                            std_dz_grid[int(i/nn),int(j/nn)] = res[6]
                            MAP_dz_grid[int(i/nn),int(j/nn)] = res[9]

                    elif self.inverse=='L2':

                        # Carry out calculations by calling the ``estimate_cell`` method
                        summary = self.estimate_cell(cell=cell, \
                            fix_beta=self.fix_beta, returned=True)

                        # Extract estimates from summary and map_estimate
                        res = estimate.get_L2_estimates(summary)

                        # Distribute the parameters back to space
                        mean_A_grid[int(i/nn),int(j/nn)] = res[0]
                        std_A_grid[int(i/nn),int(j/nn)] = res[1]

                        if self.fix_beta is None and self.zt is None:
                            mean_zt_grid[int(i/nn),int(j/nn)] = res[2]
                            std_zt_grid[int(i/nn),int(j/nn)] = res[3]
                            mean_dz_grid[int(i/nn),int(j/nn)] = res[4]
                            std_dz_grid[int(i/nn),int(j/nn)] = res[5]
                            mean_beta_grid[int(i/nn),int(j/nn)] = res[6]
                            std_beta_grid[int(i/nn),int(j/nn)] = res[7]
                            chi2_grid[int(i/nn),int(j/nn)] = res[8]
                        elif self.fix_beta is not None and self.zt is None:
                            mean_zt_grid[int(i/nn),int(j/nn)] = res[2]
                            std_zt_grid[int(i/nn),int(j/nn)] = res[3]
                            mean_dz_grid[int(i/nn),int(j/nn)] = res[4]
                            std_dz_grid[int(i/nn),int(j/nn)] = res[5]
                            chi2_grid[int(i/nn),int(j/nn)] = res[6]
                        elif self.fix_beta is None and self.zt is not None:
                            mean_dz_grid[int(i/nn),int(j/nn)] = res[2]
                            std_dz_grid[int(i/nn),int(j/nn)] = res[3]
                            mean_beta_grid[int(i/nn),int(j/nn)] = res[4]
                            std_beta_grid[int(i/nn),int(j/nn)] = res[5]
                            chi2_grid[int(i/nn),int(j/nn)] = res[6]
                        else:
                            mean_dz_grid[int(i/nn),int(j/nn)] = res[2]
                            std_dz_grid[int(i/nn),int(j/nn)] = res[3]
                            chi2_grid[int(i/nn),int(j/nn)] = res[4]

        if self.mask is not None:
            self.new_mask_grid = new_mask_grid

        self.mean_A_grid = mean_A_grid
        self.std_A_grid = std_A_grid
        self.mean_dz_grid = mean_dz_grid
        self.std_dz_grid = std_dz_grid

        if self.inverse=='bayes':

            # Store grids as attributes
            self.MAP_A_grid = MAP_A_grid
            self.MAP_dz_grid = MAP_dz_grid
            if self.fix_beta is None:
                self.mean_beta_grid = mean_beta_grid
                self.MAP_beta_grid = MAP_beta_grid
                self.std_beta_grid = std_beta_grid
            if self.zt is None:
                self.mean_zt_grid = mean_zt_grid
                self.MAP_zt_grid = MAP_zt_grid
                self.std_zt_grid = std_zt_grid
        elif self.inverse=='L2':
            self.chi2_grid = chi2_grid
            if self.fix_beta is None:
                self.mean_beta_grid = mean_beta_grid
                self.std_beta_grid = std_beta_grid
            if self.zt is None:
                self.mean_zt_grid = mean_zt_grid
                self.std_zt_grid = std_zt_grid


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


    def plot_functions(self, est='mean', title=None, save=None):
        """
        Method to plot observed and fitted PSD function using 
        one of ``MAP`` or ``mean`` estimates. Calls the function 
        :func:`~platecurie.plotting.plot_fitted` with attributes as arguments.

        :type est: str, optional
        :param est: Type of inference estimate to use for predicting th PSD
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

        except:
            raise(Exception("No estimate yet available"))

        try:
            if self.inverse=='bayes':
                if est=='mean':
                    mA = self.summary.loc['A', est]
                    mdz = self.summary.loc['dz', est]
                    if sum(self.summary.index.isin(['zt']))==1:
                        mzt = self.summary.loc['zt', est]
                    else:
                        mzt = self.zt_cell
                    if sum(self.summary.index.isin(['beta']))==1:
                        mb = self.summary.loc['beta', est]
                    else:
                        mb = self.fix_beta

                elif est=='MAP':
                    mA = np.float(self.map_estimate['A'])
                    mdz = np.float(self.map_estimate['dz'])
                    if 'zt' in self.map_estimate:
                        mzt = np.float(self.map_estimate['zt'])
                    else:
                        mzt = self.zt_cell
                    if 'beta' in self.map_estimate:
                        mb = np.float(self.map_estimate['beta'])
                    else:
                        mb = self.fix_beta
                else:
                    raise(Exception("estimate does not exist. Choose among: 'mean' or 'MAP'"))

            elif self.inverse=='L2':

                # Extract statistics from summary object
                mA = self.summary.loc['A', 'mean']
                mdz = self.summary.loc['dz', 'mean']
                if sum(self.summary.index.isin(['zt']))==1:
                    mzt = self.summary.loc['zt', 'mean']
                else:
                    mzt = self.zt_cell
                if sum(self.summary.index.isin(['beta']))==1:
                    mb = self.summary.loc['beta', 'mean']
                else:
                    mb = self.fix_beta

            # Calculate predicted PSD from estimates
            ppsd = estimate.calculate_psd(k, mA, mzt, mdz, mb)

            # Call function from ``plotting`` module
            curieplot.plot_functions(k, psd, epsd, ppsd=ppsd, title=title, save=save)

        except:

            # Call function from ``plotting`` module
            curieplot.plot_functions(k, psd, epsd, title=title, save=save)


    def plot_results(self, mean_A=False, MAP_A=False, std_A=False, \
        mean_zt=False, MAP_zt=False, std_zt=False, mean_dz=False, MAP_dz=False, \
        std_dz=False, mean_beta=False, MAP_beta=False, std_beta=False, \
        chi2=False, mask=False, contours=None, filter=False, sigma=1, save=None, **kwargs):
        """
        Method to plot grids of estimated parameters with fixed labels and titles. 
        To have more control over the plot rendering, use the function 
        :func:`~plateflex.plotting.plot_real_grid` with the relevant quantities and 
        plotting options.

        :type mean/MAP/std_A/zt/dz/beta: bool
        :param mean/MAP/std_A/zt/dz/beta: Type of plot to produce. 
            All variables default to False (no plot generated)
        """

        from skimage.filters import gaussian

        if mask:
            try:
                new_mask = self.new_mask_grid
            except:
                new_mask = None
                print('No new mask found. Plotting without mask')
        else:
            new_mask = None

        if contours is not None:
            contours = np.array(contours)/self.nn

        if mean_A:
            if filter:
                mean_A_grid = gaussian(self.mean_A_grid, sigma=sigma)
            else:
                mean_A_grid = self.mean_A_grid
            flexplot.plot_real_grid(
                mean_A_grid, 
                mask=new_mask, 
                title='Constant term', 
                clabel=r'A (nT$^2$)', 
                contours=contours, 
                save=save, 
                **kwargs)
        if MAP_A:
            if filter:
                MAP_A_grid = gaussian(self.MAP_A_grid, sigma=sigma)
            else:
                MAP_A_grid = self.MAP_A_grid
            flexplot.plot_real_grid(
                MAP_A_grid, 
                mask=new_mask, 
                title='MAP estimate of $A$', 
                clabel=r'A (nT$^2$)', 
                contours=contours, 
                save=save, 
                **kwargs)
        if std_A:
            if filter:
                std_A_grid = gaussian(self.std_A_grid, sigma=sigma)
            else:
                std_A_grid = self.std_A_grid
            flexplot.plot_real_grid(
                std_A_grid, 
                mask=new_mask, 
                title='Error on $A$', 
                clabel=r'A (nT$^2$)', 
                contours=contours, 
                save=save, 
                **kwargs)
        if mean_dz:
            if filter:
                mean_dz_grid = gaussian(self.mean_dz_grid, sigma=sigma)
            else:
                mean_dz_grid = self.mean_dz_grid
            flexplot.plot_real_grid(
                mean_dz_grid, 
                mask=new_mask, 
                title='Magnetic layer thickness', 
                clabel=r'$dz$ (km)', 
                contours=contours, 
                save=save, 
                **kwargs)
        if MAP_dz:
            if filter:
                MAP_dz_grid = gaussian(self.MAP_dz_grid, sigma=sigma)
            else:
                MAP_dz_grid = self.MAP_dz_grid
            flexplot.plot_real_grid(
                MAP_dz_grid, mask=new_mask, 
                title='MAP estimate of $dz$', 
                clabel=r'$dz$ (km)', 
                contours=contours, 
                save=save, 
                **kwargs)
        if std_dz:
            if filter:
                std_dz_grid = gaussian(self.std_dz_grid, sigma=sigma)
            else:
                std_dz_grid = self.std_dz_grid
            flexplot.plot_real_grid(
                std_dz_grid, 
                mask=new_mask, 
                title='Error on $dz$', 
                clabel=r'$dz$ (km)', 
                contours=contours, 
                save=save, 
                **kwargs)
        if mean_beta:
            try:
                if filter:
                    mean_beta_grid = gaussian(self.mean_beta_grid, sigma=sigma)
                else:
                    mean_beta_grid = self.mean_beta_grid
                flexplot.plot_real_grid(
                    mean_beta_grid, 
                    mask=new_mask, 
                    title='Fractal magnetization', 
                    clabel=r'$\beta$', 
                    contours=contours, 
                    save=save, 
                    **kwargs)
            except:
                print("parameter 'beta' was not estimated")
        if MAP_beta:
            try:
                if filter:
                    MAP_beta_grid = gaussian(self.MAP_beta_grid, sigma=sigma)
                else:
                    MAP_beta_grid = self.MAP_beta_grid
                flexplot.plot_real_grid(
                    MAP_beta_grid, 
                    mask=new_mask, 
                    title=r'MAP estimate of $\beta$', 
                    clabel=r'$\beta$', 
                    contours=contours, 
                    save=save, 
                    **kwargs)
            except:
                print("parameter 'beta' was not estimated")
        if std_beta:
            try:
                if filter:
                    std_beta_grid = gaussian(self.std_beta_grid, sigma=sigma)
                else:
                    std_beta_grid = self.std_beta_grid
                flexplot.plot_real_grid(
                    self.std_beta_grid, 
                    mask=new_mask, 
                    title=r'Error on $\beta$', 
                    clabel=r'$\beta$', 
                    contours=contours, 
                    save=save, 
                    **kwargs)
            except:
                print("parameter 'beta' was not estimated")
        if mean_zt:
            try:
                if filter:
                    mean_zt_grid = gaussian(self.mean_zt_grid, sigma=sigma)
                else:
                    mean_zt_grid = self.mean_zt_grid
                flexplot.plot_real_grid(
                    mean_zt_grid, 
                    mask=new_mask, 
                    title='Depth to top of layer', 
                    clabel=r'$z_t$ (km)', 
                    contours=contours, 
                    save=save, 
                    **kwargs)
            except:
                print("parameter 'zt' was not estimated")
        if MAP_zt:
            try:
                if filter:
                    MAP_zt_grid = gaussian(self.MAP_zt_grid, sigma=sigma)
                else:
                    MAP_zt_grid = self.MAP_zt_grid
                flexplot.plot_real_grid(
                    MAP_zt_grid, 
                    mask=new_mask, 
                    title=r'MAP estimate of $z_t$', 
                    clabel=r'$z_t$ (km)', 
                    contours=contours, 
                    save=save, 
                    **kwargs)
            except:
                print("parameter 'zt' was not estimated")
        if std_zt:
            try:
                if filter:
                    std_zt_grid = gaussian(self.std_zt_grid, sigma=sigma)
                else:
                    std_zt_grid = self.std_zt_grid
                flexplot.plot_real_grid(
                    std_zt_grid, 
                    mask=new_mask, 
                    title=r'Error on $z_t$', 
                    clabel=r'$z_t$ (km)', 
                    contours=contours, 
                    save=save, 
                    **kwargs)
            except:
                print("parameter 'zt' was not estimated")
        if chi2:
            try:
                if filter:
                    chi2_grid = gaussian(self.chi2_grid, sigma=sigma)
                else:
                    chi2_grid = self.chi2_grid
                flexplot.plot_real_grid(
                    chi2_grid, 
                    mask=new_mask, 
                    title='Reduced chi-squared error', 
                    clabel=r'$\chi_{\nu}^2$', 
                    contours=contours, 
                    save=save, 
                    **kwargs)
            except:
                print("parameter 'chi2' was not estimated")


def _progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()