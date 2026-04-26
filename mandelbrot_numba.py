import matplotlib.pyplot as plt
import numpy as np
import time, statistics
from numba import njit
from typing import Any


def benchmark(func: Any, *args: Any, n_runs: int = 3) -> tuple[float, Any]:
    """
    Jimmy's benchmarking function. Finds the median time of a given input function and its arguments.

    Parameters
    -----------
    func : Any
        A function that should be benchmarked.

    *args : Any
        The input arguments of the first parameter 'func'.

    n_runs : int
        The number of runs that should be done before the median time is found.

    Returns
    --------
    median_time : float
        The median time of the n runs of the function.

    result : Any
        The return value of the input function 'func'.
    """
    result: Any = None
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
    median_t = statistics.median(times)
    print(f"Median: {median_t:.4f}s, (min = {min(times):.4f}, max = { max(times):.4f})")
    return median_t, result


@njit
def mandelbrot_point(c: complex, max_iter: int) -> int:
    """
    Calculates the number of iterations it takes a single mandelbrot point/pixel to escape.

    Parameters
    -----------
    c : complex
        A complex constant.

    max_iter : int 
        The maximum number of iterations to calculate the mandelbrot point at the complex value c for.

    Returns
    --------
    iter_count : int
        The number of iterations it takes for the complex number z to be larger than 2.
    """
    # Base case if we do not converge after the maximum amount of iterations.
    iter_count = max_iter

    # Initialize z.
    z = 0

    # Go though all iterations.
    for n in range(0, max_iter):
        z = z**2 + c
        
        if z.real * z.real + z.imag * z.imag > 4:
            iter_count = n
            break

    return iter_count


@njit
def compute_mandelbrot_set(x_interval: tuple[float, float], y_interval: tuple[float, float], x_res: int = 1024, y_res: int = 1024, max_iter: int = 100, dtype=np.float64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the Mandelbrot set given an interval on the real and imaginary axes, 
    the resolution of the Mandelbrot set in the x and y direction 
    and the max number of iterations before a point/pixel escapes.

    Parameters
    -----------
    x_interval : tuple[float, float]
        An interval on the real axis, the first index in the tuple is the lower bound and the second is the upper bound.

    y_interval : tuple[float, float]
        An interval on the imaginary axis, the first index in the tuple is the lower bound and the second is the upper bound.

    x_res : int
        The resolution of the Mandelbrot set in the horizontal direction.
    
    y_res : int
        The resolution of the Mandelbrot set in the vertical direction.

    max_iter : int 
        The maximum number of iterations before a point/pixel escapes.
    
    dtype: type
        The type that should be used. Either np.float32 or np.float64

    Returns
    --------
    mandelbrot_grid : np.ndarray
        A numpy array containign the Mandelbrot grid with size: x_res * y_res.

    x_values : np.ndarray
        A numpy array containing x_res floating point numbers within the x_interval range.

    y_values : np.ndarray
        A numpy array containing y_res floating point numbers within the y_interval range.    
    """
    # Generate x_res and y_res uniformly spaced values within the x and y intervals.
    x_values = np.linspace(x_interval[0], x_interval[1], x_res).astype(dtype)
    y_values = np.linspace(y_interval[0], y_interval[1], y_res).astype(dtype)

    # Create a grid for the Mandelbrot set.
    mandelbrot_grid = np.zeros((x_res, y_res))

    # Go through all points in the region defined by the x and y intervals.
    for i in range(x_res):
        for j in range(y_res):
            x = x_values[i]
            y = y_values[j]

            c = x + 1j * y
            iter_count = mandelbrot_point(c, max_iter)
            mandelbrot_grid[j, i] = iter_count

    return mandelbrot_grid, x_values, y_values


if __name__ == "__main__":

    # The definition of the regions in the x and y direction.
    x_interval = (-2.0, 1.0)
    y_interval = (-1.5, 1.5)

    x_res = 1024
    y_res = 1024
    max_iter = 100

    # Warmup
    # Plot the Mandelbrot set.
    mandelbrot_set, x_values, y_values = compute_mandelbrot_set(x_interval, y_interval, x_res, y_res, dtype=np.float64)
    image = plt.pcolormesh(x_values, y_values, mandelbrot_set)
    plt.title(f"Mandelbrot set {x_res}x{y_res}, {max_iter} max iterations.")
    plt.colorbar(image, orientation='vertical')
    plt.savefig("mandelbrot_set.png")
    plt.show()

    t, M = benchmark(compute_mandelbrot_set, x_interval, y_interval, x_res, y_res, max_iter, np.float32)
    t, M = benchmark(compute_mandelbrot_set, x_interval, y_interval, x_res, y_res, max_iter, np.float64)