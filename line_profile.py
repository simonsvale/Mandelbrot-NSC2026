from mandelbrot_naive import compute_mandelbrot_set as naive_mb
from mandelbrot_vectorized import compute_mandelbrot_set as v_mb
from mandelbrot_vectorized import setup_variables
from mandelbrot_numba import compute_mandelbrot_set as nb_mb

import line_profiler # For profiling line by line


if __name__ == "__main__":
    x_interval = [-2.0, 1.0]
    y_interval = [-1.5, 1.5]
    x_res = 1024
    y_res = 1024
    max_iter = 100
    
    naive_mb(x_interval, y_interval, x_res, y_res, max_iter)

    Z, C, M, x_values, y_values = setup_variables(x_interval, y_interval, x_res, y_res)
    v_mb(Z, C, M, max_iter)

    #Warmup:
    nb_mb(x_interval, y_interval, x_res, y_res, max_iter)
    nb_mb(x_interval, y_interval, x_res, y_res, max_iter)