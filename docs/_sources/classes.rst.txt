Grid Classes
++++++++++++

This :mod:`~platecurie` module contains the following ``Grid`` classes:

- :class:`~platecurie.classes.MagGrid`
- :class:`~platecurie.classes.ZtGrid`

These classes can be initiatlized with a grid of corresponding data type, 
and contain methods inherited from :class:`~plateflex.classes.Grid` 
for the following functionality:

- Performing a wavelet transform using a Morlet wavelet
- Obtaining the wavelet scalogram from the wavelet transform
- Plotting the input grids, wavelet transform components, and scalograms

MagGrid
-------

.. autoclass:: platecurie.classes.MagGrid
   :members: 

ZtGrid
------

.. autoclass:: platecurie.classes.ZtGrid
   :members: 

Project Class
+++++++++++++

This module further contains the class :class:`~platecurie.classes.Project`, 
which itself is a container of grid objects 
(with at least one :class:`~platecurie.classes.MagGrid`). Methods are 
available to:

- Estimate model parameters at single grid cell
- Estimate model parameters at every (or decimated) grid cell
- Plot the statistics of the estimated parameters at single grid cell
- Plot the fitted power-spectral density of the magnetic anomaly at single grid cell
- Plot the final grids of model parameters
- Save the results to disk as .csv files

Project
-------

.. autoclass:: platecurie.classes.Project
   :members: 

