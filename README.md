
![](./plateflex/examples/picture/logo_platecurie.png)

## Software for mapping Curie depth from a wavelet analysis of magnetic anomaly data

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

Installation, Usage, API documentation and Jupyter Notebooks are described at https://paudetseis.github.io/PlateCurie/

Author: [`Pascal Audet`](https://www.uogeophysics.com/authors/admin/) (Developer and Maintainer)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3905424.svg)](https://doi.org/10.5281/zenodo.3905424)


#### Citing

If you use `PlateCurie` in your work, please cite the [Zenodo DOI](https://doi.org/10.5281/zenodo.3905424).

#### Contributing

All constructive contributions are welcome, e.g. bug reports, discussions or suggestions for new features. You can either [open an issue on GitHub](https://github.com/paudetseis/PlateCurie/issues) or make a pull request with your proposed changes. Before making a pull request, check if there is a corresponding issue opened and reference it in the pull request. If there isn't one, it is recommended to open one with your rationale for the change. New functionality or significant changes to the code that alter its behavior should come with corresponding tests and documentation. If you are new to contributing, you can open a work-in-progress pull request and have it iteratively reviewed.

Examples of straightforward contributions include notebooks that describe published examples of elastic thickness
results. Suggestions for improvements (speed, accuracy, etc.) are also welcome.

#### References

- Audet, P. and Gosselin, J.M. (2019). Curie depth estimation from magnetic anomaly data: a re-assessment using multitaper spectral analysis and Bayesian inference. Geophysical Journal International, 218, 494-507. https://doi.org/10.1093/gji/ggz166

- Blakely, R.J. (1995). Potential Theory in Gravity and Magnetic Applications, Cambridge University Press.

- Bouligand, C., Glen, J.M.G. and Blakely, R.J. (2009). Mapping Curie temperature depth in the western United States with a fractal model for crustal magnetization, Journal of Geophysical Research, 114, B11104, https://doi.org/10.1029/2009JB006494

- Gaudreau, E., Audet, P., and Schneider, D.A. (2019). Mapping Curie depth across western Canada from a wavelet analysis of magnetic anomaly data. Journal of Geophysical Research, 124, 4365-4385. https://doi.org/10.1029/
2018JB016726

- Mather, B., and Fullea, J. (2019). Constraining the geotherm beneath the British Isles from Bayesian inversion of Curie depth: integrated modelling of magnetic, geothermal, and seismic data. Solid Earth, 10, 839-850. https://doi.org/10.5194/se-10-839-2019
