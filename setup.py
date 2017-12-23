# -----------------------------------------------------------
# ecco_utils setup file
#
# (C) 2017 Jan-Erik Tesdal
# Released under MIT license
# -----------------------------------------------------------

from setuptools import setup
from sys import path

setup(
    name="ecco_utils",
    url= 'https://github.com/jetesdal/ecco_utils.git',
    py_modules= ['ecco_utils'],
    install_requires= [
	'numpy >= 1.11',
	],
    version="0.1"
)
