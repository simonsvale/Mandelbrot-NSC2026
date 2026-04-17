import matplotlib.pyplot as plt
import numpy as np
import time
import statistics
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
    times: list[float] = []
    for _ in range(n_runs):
        t0: float = time.perf_counter()
        result: Any = func(*args)
        times.append(time.perf_counter() - t0)
    median_t: float = float(statistics.median(times))
    print(f"Median: {median_t:.4f}s, (min = {min(times):.4f}, max = {max(times):.4f})")
    return median_t, result


def mandelbrot_point(c: complex, max_iter: int) -> int: # Technically this should be a uint, but python does not have that...
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
    iter_count: int = max_iter

    # Initialize z.
    z: complex = 0.0

    # Go though all iterations.
    n: int
    for n in range(max_iter):
        z: complex = z**2 + c
        
        # Check if the point/pixel escapes.
        if abs(z) > 2:
            iter_count: int = n
            break

    return iter_count


def compute_mandelbrot_set(x_interval: tuple[float, float], y_interval: tuple[float, float], x_res: int = 1024, y_res: int = 1024, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        The maximum number of iterations before a point/pixel escapes..

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
    x_values: np.ndarray = np.linspace(x_interval[0], x_interval[1], x_res, dtype=np.float64)
    y_values: np.ndarray = np.linspace(y_interval[0], y_interval[1], y_res, dtype=np.float64)

    # Create a grid for the Mandelbrot set.
    mandelbrot_grid: np.ndarray = np.zeros((x_res, y_res), dtype=np.int32)

    # Go through all points in the region defined by the x and y intervals.
    i: int
    for i in range(x_res):
        j: int
        for j in range(y_res):
            x: np.float64 = x_values[i]
            y: np.float64 = y_values[j]

            c: complex = complex(x, y)
            iter_count: int = mandelbrot_point(c, max_iter)
            mandelbrot_grid[j, i] = iter_count

    return mandelbrot_grid, x_values, y_values


if __name__ == "__main__":

    # The definition of the regions in the x and y direction.
    x_interval: tuple[float, float] = (-2.0, 1.0)
    y_interval: tuple[float, float] = (-1.5, 1.5)

    x_res: int = 1024
    y_res: int = 1024
    max_iter: int = 100

    t, M = benchmark(compute_mandelbrot_set, x_interval, y_interval, x_res, y_res)
    
    # Plot the Mandelbrot set.
    mandelbrot_set, x_values, y_values = compute_mandelbrot_set(x_interval, y_interval, x_res, y_res)
    image = plt.pcolormesh(x_values, y_values, mandelbrot_set)
    plt.title(f"Mandelbrot set {x_res}x{y_res}, {max_iter} max iterations.")
    plt.colorbar(image, orientation='vertical')
    plt.savefig("mandelbrot_set.png")
    plt.show()

    # ruff check mandelbrot_naive.py