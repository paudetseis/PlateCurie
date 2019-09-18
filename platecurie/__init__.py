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

Installation
------------

Dependencies
++++++++++++

The current version was developed using **Python3.7** \
Also, the following package is required:

- ``plateflex`` (https://github.com/paudetseis/PlateFlex)

See below for full installation details. 

Download the software
+++++++++++++++++++++

- Clone the repository:

.. sourcecode:: bash

   git clone https://github.com/paudetseis/PlateCurie.git
   cd PlateCurie

Conda environment
+++++++++++++++++

We recommend creating a custom ``conda`` environment
where ``platecurie`` can be installed along with its dependencies.

.. sourcecode:: bash

   conda create -n pcurie python=3.7 numpy pymc3 matplotlib seaborn -c conda-forge

or create it from the ``pcurie_env.yml`` file:

.. sourcecode:: bash

   conda env create -f pcurie_env.yml

Activate the newly created environment:

.. sourcecode:: bash

   conda activate pcurie

Installing using pip
++++++++++++++++++++

Once the previous steps are performed, you can install ``platecurie`` using pip:

.. sourcecode:: bash

   pip install plateflex
   pip install .

.. note::

   Please note, if you are actively working on the code, or making frequent edits, it is advisable
   to perform the pip installation with the ``-e`` flag. This enables an editable installation, where
   symbolic links are used rather than straight copies. This means that any changes made in the
   local folders will be reflected in the packages available on the system.


"""
# -*- coding: utf-8 -*-
from . import conf as cf
from . import estimate
from . import plotting
from .classes import MagGrid
from plateflex.cpwt import conf_cpwt as cf_wt

def set_conf_cpwt():
    cf_wt.k0 = 5.336

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
