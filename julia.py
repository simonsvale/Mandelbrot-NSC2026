import matplotlib.pyplot as plt
import numpy as np
import time


def julia_point(z, c, max_iter):
    """
    Calculates the number of iterations for a julia point given the complex constant c, 
    and a complex number z in 2D space.
    
    :param z: Complex number in 2D space.
    :param c: Complex constant.
    :param max_iter: The maximum number of iterations to calculate the point for.
    """

    # Base case if we do not converge after the maximum amount of iterations.
    iter_count = max_iter

    # Go though all iterations.
    for n in range(0, max_iter):
        z = z**2 + c
        
        if np.abs(z) > 2:
            iter_count = n
            break

    return iter_count


def compute_julia_set(x_interval, y_interval, c, x_res = 1024, y_res = 1024, max_iter = 100):
    """
    Computes the julia set given a x and y interval, resolution, complex constant c and max iterations per point.
    
    :param x_interval: Interval in the x direction.
    :param y_interval: Interval in the y direction.
    :param c: Complex constant determining which julia set to create.
    :param x_res: Resolution in the x direction.
    :param y_res: Resolution in the y direction.
    :param max_iter: The maximum iterations to calculate per mandelbrot point.
    """

    # Generate uniformly spaced values.
    grid_x = np.linspace(x_interval[0], x_interval[1], x_res)
    grid_y = np.linspace(y_interval[0], y_interval[1], y_res)

    julia_set_grid = np.zeros((1024, 1024))

    # Go through all points in the defined region.
    for i in range(x_res):
        for j in range(y_res):
            x = grid_x[i]
            y = grid_y[j]

            z = complex(x, y)
            iter_count = julia_point(z, c, max_iter)
            julia_set_grid[j, i] = iter_count
    
    return julia_set_grid


if __name__ == "__main__":

    # The definition of the regions in the x and y direction.
    x_interval = [-1.5, 1.5]
    y_interval = [-1.5, 1.5]

    x_res = 1024
    y_res = 1024
    max_iter = 100

    # Compute the julia set for (-0.5125 + 0.5213i)
    t_s = time.time()
    julia_set = compute_julia_set(x_interval, y_interval, complex(-0.5125, 0.5213), x_res, y_res)
    t_e = time.time()
    print(f"Julia set took {t_e - t_s} seconds to compute.")

    image = plt.imshow(julia_set, cmap="viridis")
    plt.title(f"Julia set {x_res}x{y_res}, {max_iter} max iterations.")
    plt.xlim(0, x_res)
    plt.ylim(0, y_res)
    plt.colorbar(image, orientation='vertical')
    plt.savefig("julia_set.png")
    plt.show()
