from mandelbrot_vectorized import compute_mandelbrot_set, setup_variables, benchmark

import matplotlib.pyplot as plt

if __name__ == "__main__":

    # The definition of the regions in the x and y direction.
    x_interval = [-2.0, 1.0]
    y_interval = [-1.5, 1.5]

    x_res = 1024
    y_res = 1024
    max_iter = 100

    resolutions = [256, 512, 1024, 2048, 4096]
    timings = []

    for res in resolutions:
        Z, C, M, x_values, y_values = setup_variables(x_interval, y_interval, res, res)
        t, M = benchmark(compute_mandelbrot_set, Z, C, M, max_iter)
        timings.append(t)

    plt.plot(resolutions, timings)
    plt.xlabel("Resolution [NxN Pixels]")
    plt.ylabel("Timing [s]")
    plt.show()
