#!/usr/bin/env python
import os
from setuptools import setup
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
 
setup(name='ANNrecon',
 version='1.0',
 description='Module for recognition isolated words',
 author='Alekseev Anton',
 author_email='aaalekseev0994@gmail.ru',
 install_requires=['numpy', 'neurolab', 'pybrain', 'wave', 'scipy', 'matplotlib'],
 packages= ['ANNrecon'],
 include_package_data=True
 )