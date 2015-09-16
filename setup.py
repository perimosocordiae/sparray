#!/usr/bin/env python

from setuptools import setup

setup(
    name='sparray',
    version='0.0.1',
    author='CJ Carey',
    author_email='perimosocordiae@gmail.com',
    description='Sparse representation for ndarrays',
    url='http://github.com/perimosocordiae/sparray',
    license='MIT',
    packages=['sparray'],
    install_requires=[
        'numpy >= 1.9',
        'scipy >= 0.15',
    ],
)
