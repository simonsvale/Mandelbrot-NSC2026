import matplotlib.pyplot as plt
import numpy as np


def mandelbrot_point(c, max_iter):

    # Setup base case, where max_iter is not enough.
    iter_count = max_iter

    # Initilize z
    z = 0

    # Go though all iterations.
    for n in range(0, max_iter):
        z = z**2 + c
        
        if np.abs(z) > 2:
            iter_count = n
            break

    return iter_count


def mandelbrot(x_interval, y_interval, N_x = 1024, N_y = 1024, max_iter = 100):

    # Generate uniformly spaced values.
    grid_x = np.linspace(x_interval[0], x_interval[1], N_x)
    grid_y = np.linspace(y_interval[0], y_interval[1], N_y)

    mandelbrot_grid = np.zeros((1024, 1024))

    # Go through all points in the defined region.
    for i in range(N_x):
        for j in range(N_y):
            x = grid_x[i]
            y = grid_y[j]

            c = complex(x, y)
            iter_count = mandelbrot_point(c, max_iter)
            mandelbrot_grid[j, i] = iter_count
    
    return mandelbrot_grid


if __name__ == "__main__":

    # The definition of the regions in the x and y direction.
    x_interval = [-2.0, 1.0]
    y_interval = [-1.5, 1.5]

    x_res = 100
    y_res = 100
    max_iter = 100

    mandelbrot_grid = mandelbrot(x_interval, y_interval, x_res, y_res)

    plt.imshow(mandelbrot_grid, cmap="viridis")
    plt.xlim(0, x_res)
    plt.ylim(0, y_res)
    plt.show()