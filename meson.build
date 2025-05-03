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
  'platecurie/__init__.py',
  pure: false,
  subdir: 'platecurie'
  )
