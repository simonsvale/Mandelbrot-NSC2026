import matplotlib.pyplot as plt
import numpy as np
import time


def mandelbrot_point(c, max_iter):
    """
    Calculates the number of iterations for a single mandelbrot point.
    
    :param c: Complex number.
    :param max_iter: The maximum number of iterations to calculate the mandelbrot point at the complex value c for.
    """

    # Base case if we do not converge after the maximum amount of iterations.
    iter_count = max_iter

    # Initialize z.
    z = 0

    # Go though all iterations.
    for n in range(0, max_iter):
        z = z**2 + c
        
        if np.abs(z) > 2:
            iter_count = n
            break

    return iter_count


def compute_mandelbrot_set(x_interval, y_interval, x_res = 1024, y_res = 1024, max_iter = 100):
    """
    Computes the Mandelbrot set given a x and y interval, resolution and max iterations per point.
    
    :param x_interval: Interval in the x direction.
    :param y_interval: Interval in the y direction.
    :param x_res: Resolution in the x direction.
    :param y_res: Resolution in the y direction.
    :param max_iter: The maximum iterations to calculate per Mandelbrot point.
    """

    # Generate x_res and y_res uniformly spaced values within the x and y intervals.
    x_values = np.linspace(x_interval[0], x_interval[1], x_res)
    y_values = np.linspace(y_interval[0], y_interval[1], y_res)

    # Create a grid for the Mandelbrot set.
    mandelbrot_grid = np.zeros((1024, 1024))

    # Go through all points in the region defined by the x and y intervals.
    for i in range(x_res):
        for j in range(y_res):
            x = x_values[i]
            y = y_values[j]

            c = complex(x, y)
            iter_count = mandelbrot_point(c, max_iter)
            mandelbrot_grid[j, i] = iter_count

    return mandelbrot_grid, x_values, y_values


if __name__ == "__main__":

    # The definition of the regions in the x and y direction.
    x_interval = [-2.0, 1.0]
    y_interval = [-1.5, 1.5]

    x_res = 1024
    y_res = 1024
    max_iter = 100

    t_s = time.time()
    mandelbrot_set, x_values, y_values = compute_mandelbrot_set(x_interval, y_interval, x_res, y_res)
    t_e = time.time()
    print(f"Mandelbrot set took {t_e - t_s} seconds to compute.")

    # Plot the Mandelbrot set.
    image = plt.pcolormesh(x_values, y_values, mandelbrot_set)
    plt.title(f"Mandelbrot set {x_res}x{y_res}, {max_iter} max iterations.")
    plt.colorbar(image, orientation='vertical')
    plt.savefig("mandelbrot_set.png")
    plt.show()