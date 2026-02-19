import matplotlib.pyplot as plt
import numpy as np
import time, statistics


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


def setup_variables(x_interval: tuple[float, float], y_interval: tuple[float, float], x_res: int = 1024, y_res: int = 1024):
    """
    Setup variables for computing the mandelbrot set.

    :param x_interval: Interval in the x direction.
    :param y_interval: Interval in the y direction.
    :param x_res: Resolution in the x direction.
    :param y_res: Resolution in the y direction.
    """
    # Generate x_res and y_res uniformly spaced values within the x and y intervals.
    x_values = np.linspace(x_interval[0], x_interval[1], x_res)
    y_values = np.linspace(y_interval[0], y_interval[1], y_res)

    # Create grid of complex numbers.
    X, Y = np.meshgrid(x_values, y_values)
    C = X + 1j * Y

    # Initialize the complex Z array to 0.
    Z = np.zeros((x_res, y_res), dtype=np.complex128)

    # Mask array containing iterations.
    M = np.zeros((x_res, y_res))
    
    return Z, C, M, x_values, y_values


def compute_mandelbrot_set(Z: np.ndarray[np.complex128], C: np.ndarray[np.complex128], M: np.ndarray[int], max_iter: int = 100):
    """
    Computes the Mandelbrot set given a x and y interval, resolution and max iterations per point.
    
    :param max_iter: The maximum iterations to calculate per Mandelbrot point.
    """

    # Go through all points in the meshgrid.
    for _ in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1

    return M


if __name__ == "__main__":

    # The definition of the regions in the x and y direction.
    x_interval = [-2.0, 1.0]
    y_interval = [-1.5, 1.5]

    x_res = 1024
    y_res = 1024
    max_iter = 100

    Z, C, M, x_values, y_values = setup_variables(x_interval, y_interval, x_res, y_res)

    t, M = benchmark(compute_mandelbrot_set, Z, C, M, max_iter)

    # Plot the Mandelbrot set.
    image = plt.pcolormesh(x_values, y_values, M)
    plt.title(f"Mandelbrot set {x_res}x{y_res}, {max_iter} max iterations.")
    plt.colorbar(image, orientation='vertical')
    plt.savefig("mandelbrot_set.png")
    plt.show()