from setuptools import setup, find_packages 
from codecs import open
from os import path
from minot.__init__ import __version__

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='minot',
    version=__version__,
    description='Python code to model the intra cluster medium thermal and non-thermal components and provide predictions for associated observables',
    long_description=long_description,  #this is the readme 
    long_description_content_type='text/x-rst',
    url='https://github.com/remi-adam/minot',
    author='Remi Adam and Hazal Goksu',
    author_email='remi.adam@llr.in2p3.fr',
    license='BSD',
    # See https://pypi.org/classifiers/
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages = find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'matplotlib',
        'ebltable',
    ]
)
