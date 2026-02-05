# mamba create -n nsc2026 python=3.11
# mamba activate nsc2026
# mamba install numpy matplotlib scipy numba pytest dask

# mamba env export > mandelbrot/environment.yml

import numpy 
import matplotlib
import numba 
import scipy 
import pytest 
import dask

print("HELLO!")

print(numpy.__version__)
print(matplotlib.__version__)
print(numba.__version__)
print(scipy.__version__)
print(pytest.__version__)
print(dask.__version__)


