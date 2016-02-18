#!/usr/bin/env python
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='sparray',
    version='0.0.2',
    author='CJ Carey',
    author_email='perimosocordiae@gmail.com',
    description='Sparse representation for ndarrays',
    url='http://github.com/perimosocordiae/sparray',
    license='MIT',
    packages=['sparray'],
    install_requires=[
        'numpy >= 1.9',
        'scipy >= 0.15',
        'Cython >= 0.21',
    ],
    ext_modules=cythonize('sparray/*.pyx'),
)
