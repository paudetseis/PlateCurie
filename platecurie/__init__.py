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

PlateFlex is a software for estimating the effective elastic thickness of the lithosphere
from the inversion of flexural isostatic response functions calculated from a wavelet
analysis of gravity and topography data.

Licence
-------

Copyright 2019 Pascal Audet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Dependencies
++++++++++++

- A fortran compiler
- ``pymc`` 
- ``seaborn``
- ``scikit-image`` (https://scikit-image.org)
- ``plateflex``

See below for full installation details. 

Conda environment
+++++++++++++++++

We recommend creating a custom ``conda`` environment
where ``platecurie`` can be installed along with its dependencies. This will ensure
that all packages are compatible.

.. Note::
    In theory, you could use your own fortran compiler. However, to ensure a proper installation,
    it is recommended to install `fortran-compiler` in the `pflex` environment.

.. sourcecode:: bash

   conda create -n pcurie -c conda-forge python=3.12 fortran-compiler pymc seaborn scikit-image

Activate the newly created environment:

.. sourcecode:: bash

   conda activate pcurie

Install ``plateflex`` from the development on GitHub

.. sourcecode:: bash

    pip install plateflex@git+https://github.com/paudetseis/plateflex

Installing development branch from GitHub
+++++++++++++++++++++++++++++++++++++++++

Install the latest version of ``platecurie`` from the GitHub repository with the following command:

.. sourcecode:: bash

    pip install platecurie@git+https://github.com/paudetseis/platecurie

Jupyter Notebooks
+++++++++++++++++

Included in this package is a set of Jupyter Notebooks, which give examples on how to call the 
various routines The Notebooks describe how to reproduce published examples of synthetic data from 
Gaudreau et al., (2019)

After installing ``platecurie``, these notebooks can be locally installed (i.e., in a local folder ``Examples``) 
from the package by running:

.. sourcecode:: python

    from plateflex import doc
    doc.install_doc(path='Examples')

To run the notebooks you will have to further install ``jupyter``:

.. sourcecode:: bash

    conda install jupyter

Then:

.. sourcecode:: bash

    unzip data.zip
    cd Examples
    jupyter notebook

You can then save the notebooks as ``python`` scripts and you should be good to go!

"""

__version__ = '0.2.0'

__author__ = 'Pascal Audet'

# -*- coding: utf-8 -*-
from . import estimate
from . import plotting
from .classes import MagGrid, ZtGrid, SigZtGrid, Project
from plateflex.cpwt import conf_cpwt as cf_wt

def set_conf_cpwt(k0=5.336):
    cf_wt.k0 = k0

set_conf_cpwt()

def get_conf_cpwt():
    """
    Print global variable that controls the spatio-spectral resolution of the
    wavelet transform

    .. rubric:: Example

    >>> import platecurie
    >>> platecurie.get_conf_cpwt()
    Wavelet parameter used in platecurie.cpwt:
    ------------------------------------------
    [Internal wavenumber]      k0 (float):     5.336
    """

    print('\n'.join((
        'Wavelet parameter used in platecurie.cpwt:',
        '------------------------------------------',
        '[Internal wavenumber]      k0 (float):     {0:.3f}'.format(cf_wt.k0))))
