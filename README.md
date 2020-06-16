# PlateCurie: Software for mapping Curie depth from a wavelet analysis of magnetic anomaly data

<!-- ![](./plateflex/examples/picture/tws_logo.png)
 -->
Crustal magnetic anomalies carry information on the source distribution of magnetization
in the Earth's crust [(Blakely, 1995)](#references). 
The Curie point corresponds to the depth at which crustal rocks loose
their magnetization where they reach their Curie temperature, and is obtained by fitting
the power spectral density (PSD) of magnetic anomaly data with a model where magnetic anomalies
are confined within a layer 
[(Bouligand et al., 2009; Audet and Gosselin, 2019; Mather and Fullea, 2019)](#references). 
Mapping the Curie point provides important information on 
geothermal gradients in the Earth; however, mapping Curie depth is a spatio-spectral 
localization problem because the PSD needs to be calculated within moving windows at 
wavelengths long enough to capture the greatest possible depth to the bottom of the
magnetic layer. The wavelet transform is particularly well suited to overcome 
this problem because it avoids splitting the grids into small windows and can therefore 
produce PSD functions at each point of the input grid [(Gaudreau et al., 2019)](#references).

This package extends the package `plateflex`, which contains `python` modules to calculate 
the wavelet transform and scalogram of 2D gridded data, by providing a new class 
`MagGrid` that inherits from `plateflex.classes.Grid` with methods to estimate the properties
of the magnetic layer (depth to top of layer (<i>z<sub>t</sub></i>), thickness
of layer (<i>dz</i>), and power-law exponent of fractal magnetization (<i>&beta;</i>))
using Bayesian inference. Common computational workflows are covered in the Jupyter 
notebooks bundled with this package. The software contains methods to make beautiful and
insightful plots using the `seaborn` package.

Authors: [`Pascal Audet`](https://www.uogeophysics.com/authors/admin/) (Developer and Maintainer)

## Installation

### Dependencies

The current version was developed using **Python3.7**
Also, the following packages are required:

- [`plateflex`](https://github.com/paudetseis/PlateFlex)

> **_NOTE:_**  All dependencies are installed from `plateflex`

### Installing using pip

You can install `platecurie` using the [pip package manager](https://pypi.org/project/pip/):

```bash
pip install platecurie
```
All the dependencies will be automatically installed by `pip`.

### Installing with conda

You can install `platecurie` using the [conda package manager](https://conda.io).
Its required dependencies can be easily installed with:

```bash
conda install numpy pymc3 matplotlib seaborn -c conda-forge
```

Then `platecurie` can be installed with `pip`:

```bash
pip install platecurie
```

#### Conda environment

We recommend creating a custom 
[conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)
where `platecurie` can be installed along with its dependencies. 

- Create a environment called `curie` and install all dependencies:

```bash
conda create -n curie python=3.7 numpy pymc3 matplotlib seaborn -c conda-forge
```

- or create it from the `curie_env.yml` file by first cloning the repository:

```bash
git clone https://github.com/paudetseis/PlateCurie.git
cd PlateCurie
conda env create -f curie_env.yml
```

Activate the newly created environment:

```bash
conda activate curie
```

Install `platecurie` with `pip`:

```bash
pip install plateflex
pip install platecurie
```

### Installing from source

Download or clone the repository:
```bash
git clone https://github.com/paudetseis/PlateCurie.git
cd PlateCurie
```

Next we recommend following the steps for creating a `conda` environment (see [above](#conda-environment)). Then install using `pip`:

```bash
pip install plateflex
pip install .
``` 

---
**_NOTE:_**

If you are actively working on the code, or making frequent edits, it is advisable to perform 
installation from source with the `-e` flag: 

```bash
pip install -e .
```

This enables an editable installation, where symbolic links are used rather than straight 
copies. This means that any changes made in the local folders will be reflected in the 
package available on the system.

---

## Usage 

### Jupyter Notebooks

Included in this package is a set of Jupyter Notebooks, which give examples on how to call the various routines 
The Notebooks describe how to reproduce published examples of synthetic data from [Gaudreau et al., (2019)](#references).

<!-- - [sim_obs_Audet2016.ipynb](./plateflex/examples/Notebooks/sim_obs_Audet2016.ipynb): Example plane wave seismograms and P receiver functions for OBS data from [Audet (2016)](#Audet).
- [sim_Prfs_Porter2011.ipynb](./plateflex/examples/Notebooks/sim_Prfs_Porter2011.ipynb): Example P receiver functions from [Porter et al. (2011)](#Porter)
- [sim_SKS.ipynb](./plateflex/examples/Notebooks/sim_SKS.ipynb): Example plane wave seismograms for SKS splitting studies.
 -->
After [installing `platecurie`](#installation), these notebooks can be locally installed (i.e., in a local folder `Notebooks`) from the package by running:

```python
from platecurie import doc
doc.install_doc(path='Notebooks')
```

To run the notebooks you will have to further install `jupyter`:

```bash
conda install jupyter
```

Then ```cd Notebooks``` and type:

```bash
jupyter notebook
```

You can then save the notebooks as `python` scripts and you should be good to go!

### Testing

A series of tests are located in the ``tests`` subdirectory. In order to perform these tests, clone the repository and run `pytest` (`conda install pytest` if needed):

```bash
git checkout https://github.com/paudetseis/PlateCurie.git
cd PlateCurie
pytest -v
```

### Documentation

The documentation for all classes and functions in `platecurie` can be accessed from https://paudetseis.github.io/PlateCurie/.

## References

- Audet, P. and Gosselin, J.M. (2019). Curie depth estimation from magnetic anomaly data: a re-assessment using multitaper spectral analysis and Bayesian inference. Geophysical Journal International, 218, 494-507. https://doi.org/10.1093/gji/ggz166

- Blakely, R.J. (1995). Potential Theory in Gravity and Magnetic Applications, Cambridge University Press.

- Bouligand, C., Glen, J.M.G. and Blakely, R.J. (2009). Mapping Curie temperature depth in the western United States with a fractal model for crustal magnetization, Journal of Geophysical Research, 114, B11104, https://doi.org/10.1029/2009JB006494

- Gaudreau, E., Audet, P., and Schneider, D.A. (2019). Mapping Curie depth across western Canada from a wavelet analysis of magnetic anomaly data. Journal of Geophysical Research, 124, 4365-4385. https://doi.org/10.1029/
2018JB016726

- Mather, B., and Fullea, J. (2019). Constraining the geotherm beneath the British Isles from Bayesian inversion of Curie depth: integrated modelling of magnetic, geothermal, and seismic data. Solid Earth, 10, 839-850. https://doi.org/10.5194/se-10-839-2019
