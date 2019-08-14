#! /usr/bin/env python

from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='GaussianBeams',
      version='0.1',
      description='GaussianBeams module for calculating and manipulating Gauss-Laguerre beams',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
      ],
      keywords='quasioptics, Gaussian beams', 'Gaussian modes', 'Gauss-Laguerre modes',
      url='https://github.com/PaulKGrimes/GaussianBeams',
      author='Paul Grimes',
      author_email='pgrimes@cfa.harvard.edu',
      packages=find_packages(),
      install_requires=[
          'numpy', 'scipy'
      ],
      include_package_data=True,
      #test_suite='nose.collector',
      #tests_require=['nose'],
      zip_safe=True)
