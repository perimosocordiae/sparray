#!/usr/bin/env python
from setuptools import setup

try:
  from Cython.Build import cythonize
  import numpy as np
except ImportError:
  use_cython = False
else:
  use_cython = True

setup_kwargs = dict(
    name='sparray',
    version='0.0.2',
    author='CJ Carey',
    author_email='perimosocordiae@gmail.com',
    description='Sparse representation for ndarrays',
    url='http://github.com/perimosocordiae/sparray',
    license='MIT',
    packages=['sparray'],
    package_data = {'': ['*.pyx']},
    install_requires=[
        'numpy >= 1.9',
        'scipy >= 0.15',
        'Cython >= 0.21',
    ],
)
if use_cython:
  setup_kwargs['include_dirs'] = [np.get_include()]
  setup_kwargs['ext_modules'] = cythonize('sparray/*.pyx')

setup(**setup_kwargs)
