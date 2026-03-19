# mamba create -n nsc2026 python=3.11
# mamba activate nsc2026
# mamba install numpy matplotlib scipy numba pytest dask python-graphviz
# mamba install line_profiler

# mamba env export > mandelbrot/environment.yml

import numpy 
import matplotlib
import numba 
import scipy 
import pytest 
import dask

print("HELLO!")

print("NumPy:", numpy.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Numba:", numba.__version__)
print("Scipy:", scipy.__version__)
print("Pytest:", pytest.__version__)
print("Dask:", dask.__version__)