from setuptools import setup
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options

# See https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
# for a deeper explanation of the choices here
# Cython.Compiler.Options.docstring = False
Cython.Compiler.Options.error_on_uninitialized = True
directives = {
    'language_level': '3',  # We assume Python 3 code
    'boundscheck': False,  # Do not check array access
    'wraparound': False,  # a[-1] does not work
    'embedsignature': False,  # Do not save typing / docstring
    'always_allow_keywords': False,  # Faster calling conventions
    'initializedcheck': False,  # We take care of initializing cdef classes and memory views
}

setup(
    packages=['Modules'],
    ext_modules=cythonize(["Modules/random.pyx"], quiet=False,
                          compiler_directives=directives),
    include_dirs=[numpy.get_include()]
)

setup(
    packages=['Modules'],
    ext_modules=cythonize(["Modules/mc_cython.pyx"], quiet=False,
                          compiler_directives=directives),
    include_dirs=[numpy.get_include()]
)



# setup(ext_modules=cythonize(["mc_cython.pyx", "random.pyx"]))
