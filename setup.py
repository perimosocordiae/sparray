#!/usr/bin/env python
from setuptools import setup, Extension

try:
  from Cython.Build import cythonize
  import numpy as np
except ImportError:
  use_cython = False
else:
  use_cython = True

setup_kwargs = dict(
    name='sparray',
    version='0.0.4',
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
  exts = [Extension('*', ['sparray/*.pyx'], include_dirs=[np.get_include()])]
  setup_kwargs['ext_modules'] = cythonize(exts)

setup(**setup_kwargs)
