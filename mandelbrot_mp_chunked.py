import numpy as np
from numba import njit
from multiprocessing import Pool
import time, statistics, psutil
import matplotlib.pyplot as plt
import math

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
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out


def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def _worker(args):
    return mandelbrot_chunk(*args)


def mandelbrot_parallel(N, x_interval, y_interval, max_iter=100, n_workers=4, n_chunks=32):

    x_min, x_max = x_interval[0], x_interval[1]
    y_min, y_max = y_interval[0], y_interval[1]

    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0

    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    exec_time = 0
    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks)

        t_s = time.perf_counter()
        parts = pool.map(_worker, chunks)
        parts = np.vstack(parts)
        t_e = time.perf_counter()
        exec_time = t_e - t_s

    return parts, exec_time


def get_n_chunk_list(N):
    n_chunks_list = [2**i for i in range(2, int(math.log2(N) + 1))]
    return n_chunks_list


def calculate_LIF(p, Tp, T1):
    LIF = p * (Tp / T1) - 1
    return LIF


def calculate_speedup(Tp, T1):
    speedup = T1 / Tp
    return speedup


def calculate_implied_serial(speedup, p):
    implied_serial = ((1 / speedup) - (1/p)) / (1 - (1/p))
    return implied_serial


def sweep(N, x_interval, y_interval, n_chunks_list, T1, workers, max_iter=100, run_count=3):

    sweep_number = len(n_chunks_list)

    # n_chunks | time parallel [s] | time serial [s] | speedup | LIF
    sweep_metrics = np.zeros((sweep_number, 5), dtype=np.float64)

    # Go through all n_chunks in the list.
    for i in range(sweep_number):
        time_list = []

        # Calculate the median from run_count
        for _ in range(run_count):
            M, exec_time = mandelbrot_parallel(N, x_interval, y_interval, max_iter, n_workers=workers, n_chunks=n_chunks_list[i])
            time_list.append(exec_time)
        Tp = statistics.median(time_list)

        # Record the metrics.
        sweep_metrics[i][0] = n_chunks_list[i]
        sweep_metrics[i][1] = Tp
        sweep_metrics[i][2] = T1 

        speedup = calculate_speedup(Tp, T1)
        sweep_metrics[i][3] = speedup
        sweep_metrics[i][4] = calculate_LIF(workers, Tp, T1)
        print(f"N_chunks: {n_chunks_list[i]}, Parallel Time: {Tp} for {workers} workers!")

    return sweep_metrics


if __name__ == "__main__":

    np.set_printoptions(suppress=True)

    x_interval = [-2.0, 1.0]
    y_interval = [-1.5, 1.5]

    N = 8192
    max_iter = 100

    runs = 3

    # Fixed to best L4 run worker/core count:
    workers = psutil.cpu_count(logical=True)

    s_time_list = []
    for _ in range(runs):
        t_s = time.perf_counter()
        mandelbrot_serial(
            N, 
            x_interval[0], x_interval[1], 
            y_interval[0], y_interval[1], 
            max_iter
        )
        t_e = time.perf_counter()
        s_time_list.append(t_e - t_s)
    T1 = statistics.median(s_time_list)
    print("Serial median time: ", T1)

    # Perform the sweep.
    n_chunks_list = get_n_chunk_list(N)
    sweep_metrics = sweep(N, x_interval, y_interval, n_chunks_list, T1, workers, max_iter)
    print(sweep_metrics)

