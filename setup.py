import setuptools
from numpy.distutils.core import setup, Extension

setup(
    name                = 'platecurie',
    version             = '0.0.1',
    description         = 'Python package for estimating Curie depth',
    author              = 'Pascal Audet',
    maintainer          = 'Pascal Audet',
    author_email        = 'pascal.audet@uottawa.ca',
    url                 = 'https://github.com/paudetseis/PlateCurie', 
    classifiers         = [
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'],
    install_requires    = ['numpy>=1.15', 'pymc3', 'matplotlib', 'seaborn'],
    python_requires     = '>=3.6',
    tests_require       = ['pytest'],
    packages            = ['platecurie'],
    package_data        = {
        'platecurie': [
            # 'examples/*.ipynb',
            'examples/data/*.txt',
            'examples/Notebooks/*.py']
    }
)
