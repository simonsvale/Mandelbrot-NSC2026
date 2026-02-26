import time, statistics

from mandelbrot_naive import compute_mandelbrot_set as naive_mb
from mandelbrot_vectorized import compute_mandelbrot_set as v_mb
from mandelbrot_vectorized import setup_variables
from mandelbrot_numba import compute_mandelbrot_set as nb_mb

def benchmark(func, *args, n_runs=3):
    """ Time func , return median of n_runs . 
    """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
    median_t = statistics.median(times)
    print(f"Median: {median_t:.4f}s, (min = {min(times):.4f}, max = { max(times):.4f})")
    return median_t, result


if __name__ == "__main__":
    x_interval = [-2.0, 1.0]
    y_interval = [-1.5, 1.5]
    x_res = 1024
    y_res = 1024
    max_iter = 100
    
    # Numba with warmup:
    nb_mb(x_interval, y_interval, x_res, y_res, max_iter)

    t, M = benchmark(nb_mb, x_interval, y_interval, x_res, y_res, max_iter)