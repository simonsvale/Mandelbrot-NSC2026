import cProfile, pstats
from mandelbrot_naive import compute_mandelbrot_set as naive_mb
from mandelbrot_vectorized import compute_mandelbrot_set as v_mb
from mandelbrot_vectorized import setup_variables
from mandelbrot_numba import compute_mandelbrot_set as nb_mb

if __name__ == "__main__":

    x_interval = [-2.0, 1.0]
    y_interval = [-1.5, 1.5]
    x_res = 1024
    y_res = 1024
    max_iter = 100
    
    profile_names = ["naive_profile.pf", "numba_profile.pf", "numpy_profile.pf"]

    cProfile.run("naive_mb(x_interval, y_interval, x_res, y_res, max_iter)", "naive_profile.pf")

    Z, C, M, x_values, y_values = setup_variables(x_interval, y_interval, x_res, y_res)
    cProfile.run("v_mb(Z, C, M, max_iter)", "numpy_profile.pf")

    #Warmup:
    nb_mb(x_interval, y_interval, x_res, y_res, max_iter)
    cProfile.run("nb_mb(x_interval, y_interval, x_res, y_res, max_iter)", "numba_profile.pf")

    for name in (profile_names):
        stats = pstats.Stats(name)
        stats.sort_stats("cumulative")
        stats.print_stats(15)


