import numpy as np
from numba import njit
from multiprocessing import Pool
import time, statistics, psutil
import matplotlib.pyplot as plt

@njit
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


@njit
def mandelbrot_chunk(row_start, row_end, N,
    x_min, x_max, y_min, y_max, max_iter):
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


def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4):
    chunk_size = max(1, N // n_workers)
    chunks, row = [], 0

    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks)
        parts = pool.map(_worker, chunks)

    return np.vstack(parts)


if __name__ == "__main__":
    x_interval = [-2.0, 1.0]
    y_interval = [-1.5, 1.5]

    res = 1024
    max_iter = 100

    runs = 3

    core_count = psutil.cpu_count(logical=False)

    s_time_list = []
    for _ in range(runs):
        t_s = time.perf_counter()
        mandelbrot_serial(
            res, 
            x_interval[0], x_interval[1], 
            y_interval[0], y_interval[1], 
            max_iter
        )
        t_e = time.perf_counter()
        s_time_list.append(t_e - t_s)
    s_median = statistics.median(s_time_list)
    print("Serial median time: ", s_median)


    p_time_list = []
    for _ in range(runs):
        t_s = time.perf_counter()
        M = mandelbrot_parallel(
            res, 
            x_interval[0], x_interval[1], 
            y_interval[0], y_interval[1], 
            max_iter, n_workers=core_count
        )
        t_e = time.perf_counter()
        p_time_list.append(t_e - t_s)
    p_median = statistics.median(p_time_list)
    print("Serial median time: ", p_median)

    image = plt.imshow(M, extent=(x_interval[0], x_interval[1], y_interval[0], y_interval[1]), interpolation="nearest")
    plt.title(f"Mandelbrot set {res}x{res}, {max_iter} max iterations.")
    plt.colorbar(image, orientation='vertical')
    plt.show()

