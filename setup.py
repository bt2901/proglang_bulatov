try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
#	from distutils.core import setup, Extension
import numpy

module = Extension('em', ['em_module.cpp', 'EM.cpp'], include_dirs=[numpy.get_include()])
setup(ext_modules=[module])
