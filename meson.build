project('platecurie', 'c',
  version : '0.2.0',
  license: 'MIT',
  meson_version: '>=0.64.0',
)

py_mod = import('python')
py = py_mod.find_installation()
py_dep = py.dependency()

py.install_sources(
  'platecurie/classes.py',
  'platecurie/estimate.py',
  'platecurie/plotting.py',
  'platecurie/doc.py',
  'platecurie/__init__.py',
  pure: false,
  subdir: 'platecurie'
  )

install_data(
    'platecurie/examples/data/Mag_NWCAN.xyz',
    'platecurie/examples/data/Sed_NWCAN.xyz',
    install_dir: py.get_install_dir(pure: false) / 'platecurie/examples/data',
)
install_data(
    'platecurie/examples/Notebooks/Ex1_making_grids.ipynb',
    'platecurie/examples/Notebooks/Ex2_wavelet_analysis.ipynb',
    'platecurie/examples/Notebooks/Ex3_estimate_curie_parameters_cell.ipynb',
    'platecurie/examples/Notebooks/Ex4_estimate_curie_parameters_grid.ipynb',
    'platecurie/examples/Notebooks/Ex5_estimate_curie_variable_zt.ipynb',
    install_dir: py.get_install_dir(pure: false) / 'platecurie/examples/Notebooks',

)