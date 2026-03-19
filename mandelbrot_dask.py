import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from dask import delayed
from dask.distributed import Client, LocalCluster
import time, statistics, dask, psutil


@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real, z_imag = 0, 0

    for i in range(max_iter):
        z_sq = z_real*z_real + z_imag*z_imag
        if z_sq > 4.0: 
            return i
        z_imag_new = 2.0*z_real*z_imag + c_imag
        z_real = z_real*z_real - z_imag*z_imag + c_real
        z_imag = z_imag_new

    return max_iter


@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, N, x_interval, y_interval, max_iter):

    x_min, x_max = x_interval[0], x_interval[1]
    y_min, y_max = y_interval[0], y_interval[1]

    # Initialize output array.
    result = np.empty((row_end - row_start, N))

    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    
    for row in range(row_end - row_start):
        c_imag = y_min + (row + row_start) * dy
        for col in range(N):
            result[row, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return result


def mandelbrot_dask(N, x_interval, y_interval, max_iter=100, n_chunks=32):
    # Ensure that the chunk size is not below 1.
    chunk_size = max(1, N // n_chunks)
    dask_tasks = []

    # Start at the first row into the mandelbrot set.
    row_start = 0

    while row_start < N:
        # Ensure that the end of the row is never larger than the resolution/length N.
        row_end = min(row_start + chunk_size, N)

        # Append a task to the task graph.
        dask_tasks.append(delayed(mandelbrot_chunk)(row_start, row_end, N, x_interval, y_interval, max_iter))

        # Set the new starting position of the row.
        row_start = row_end
    
    # Compute the chunk and time it.
    t_s = time.perf_counter()
    mandelbrot_grid = dask.compute(*dask_tasks)
    t_e = time.perf_counter()
    exec_time = t_e - t_s
    return np.vstack(mandelbrot_grid), exec_time


def mandelbrot_serial(N, x_interval, y_interval, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_interval, y_interval, max_iter)


if __name__ == "__main__":

    # The definition of the regions in the x and y direction.
    x_interval = [-2.0, 1.0]
    y_interval = [-1.5, 1.5]

    N = 8192
    max_iter = 100
    
    # Get the largest amount of workers.
    workers = psutil.cpu_count(logical=False)
    n_chunks = workers
    run_count = 3

    # Create the dask cluster, set the initial amount to 1.
    cluster = LocalCluster(n_workers=workers, threads_per_worker=1, processes=True, memory_limit=None)
    client = Client(cluster)

    # JIT compile the mandelbrot chunk function.
    client.run(lambda: mandelbrot_chunk(0, 8, 8, x_interval, y_interval, 10))

    # Get the serial time.
    serial_time_list = []
    for _ in range(run_count):
        t0 = time.perf_counter()
        mandelbrot_serial(N, x_interval, y_interval, max_iter)
        t1 = time.perf_counter()
        serial_time_list.append(t1 - t0)
    T1 = statistics.median(serial_time_list)
    print(f"Serial Median: {T1:.4f}s, (min = {min(serial_time_list):.4f}, max = {max(serial_time_list):.4f})")

    # Define output variable. This is just so we can visualize it later.
    mandelbrot_grid = 0

    # Find the median time.
    time_list = []
    for _ in range(run_count):
        mandelbrot_grid, exec_time = mandelbrot_dask(N, x_interval, y_interval, max_iter, n_chunks)
        time_list.append(exec_time)
    median_time = statistics.median(time_list)
    print(f"Dask Median: {median_time:.4f}s, (min = {min(time_list):.4f}, max = {max(time_list):.4f})")

    # So we can use the dashboard.
    input("Press any key to close the Dask dashboard!")

    # Gracefully shutdown of dask.
    client.close()
    cluster.close()

    # Create x and y values to be displayed in the pcolormesh.
    """
    x_values = np.linspace(x_interval[0], x_interval[1], N)
    y_values = np.linspace(y_interval[0], y_interval[1], N)
    
    image = plt.pcolormesh(x_values, y_values, mandelbrot_grid)
    plt.title(f"Mandelbrot set {N}x{N}, {max_iter} max iterations.")
    plt.colorbar(image, orientation='vertical')
    #plt.savefig("mandelbrot_set.png")
    plt.show()
    """