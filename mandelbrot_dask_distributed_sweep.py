import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from dask import delayed
from dask.distributed import Client, LocalCluster
import time, statistics, dask, psutil, math


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


def get_n_chunk_list(N):
    n_chunks_list = [2**i for i in range(2, int(math.log2(N) + 1))]
    return n_chunks_list


def calculate_LIF(p, Tp, T1):
    LIF = p * (Tp / T1) - 1
    return LIF


def calculate_speedup(Tp, T1):
    speedup = T1 / Tp
    return speedup


def mandelbrot_serial(N, x_interval, y_interval, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_interval, y_interval, max_iter)


def sweep(N, x_interval, y_interval, n_chunks_list, T1, p, max_iter=100, run_count = 3):

    sweep_number = len(n_chunks_list)

    # n_chunks | time (s) | vs 1x | speedup | LIF
    sweep_metrics = np.zeros((sweep_number, 5), dtype=np.float64)

    # Go through all n_chunks in the list.
    for i in range(sweep_number):
        time_list = []
        for _ in range(run_count):
            _, exec_time = mandelbrot_dask(N, x_interval, y_interval, max_iter, n_chunks_list[i])
            time_list.append(exec_time)
        Tp = statistics.median(time_list)

        # Record the metrics.
        sweep_metrics[i][0] = n_chunks_list[i]
        sweep_metrics[i][1] = Tp
        sweep_metrics[i][2] = 0.0
        sweep_metrics[i][3] = calculate_speedup(Tp, T1)
        sweep_metrics[i][4] = calculate_LIF(p, Tp, T1)
        print(f"Time = {Tp}, n_chunks = {n_chunks_list[i]}")

    return sweep_metrics


def get_cluster_ip(path: str) -> str:

    cluster_ip = ""

    # Read cluster ip
    with open(path) as f:
        cluster_ip = f.read()

    return cluster_ip


if __name__ == "__main__":

    np.set_printoptions(suppress=True)

    # The definition of the regions in the x and y direction.
    x_interval = [-2.0, 1.0]
    y_interval = [-1.5, 1.5]

    N = 8192
    max_iter = 100
    
    # Get the largest amount of workers.
    workers = psutil.cpu_count(logical=True)
    n_chunks = workers
    run_count = 3
    print(f"Number of workers: {workers}")

    # Create the the distributed dask client.
    ip = get_cluster_ip("cluster_ip")
    client = Client(ip)

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

    # Initialize the list of number of chunks to sweep over.
    n_chunks_list = get_n_chunk_list(N)
        
    # Peform a sweep over the n_chunks list.
    sweep_metrics = sweep(N, x_interval, y_interval, n_chunks_list, T1, workers, max_iter)
    print(sweep_metrics)

    x_n_chunks = sweep_metrics[:,0]
    y_time = sweep_metrics[:,1]
    plt.plot(x_n_chunks, y_time)
    plt.title(f"{workers} worker n_chunk sweep for a mandelbrot set with resolution {N}x{N}")
    plt.xlabel("n_chunks")
    plt.ylabel("Time [s]")
    plt.xscale("log")
    plt.savefig("dask_chunk_sweep.png")

    # So we can use the dashboard.
    input("Press enter to close the Dask dashboard!")

    client.close()