import numpy as np
from numba import njit
from multiprocessing import Pool
import time, statistics, psutil
import matplotlib.pyplot as plt

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


def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4):
    chunk_size = max(1, N // n_workers)
    chunks, row = [], 0

    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks)

        t_s = time.perf_counter()
        parts = pool.map(_worker, chunks)
        parts = np.vstack(parts)
        t_e = time.perf_counter()
        exec_time = t_e - t_s

    return parts, exec_time


if __name__ == "__main__":
    x_interval = [-2.0, 1.0]
    y_interval = [-1.5, 1.5]

    N = 8192
    max_iter = 100

    runs = 3

    core_count = psutil.cpu_count(logical=True)

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
    s_median = statistics.median(s_time_list)
    print("Serial median time: ", s_median)

    speedup_list = []
    efficiency_list = []

    for cores in range(2, core_count + 1):
        p_time_list = []
        for _ in range(runs):
            M, exec_time = mandelbrot_parallel(
                N, 
                x_interval[0], x_interval[1], 
                y_interval[0], y_interval[1], 
                max_iter, n_workers=cores
            )
            p_time_list.append(exec_time)
        p_median = statistics.median(p_time_list)
        speedup = s_median / p_median
        efficiency = speedup / cores
        speedup_list.append(speedup)
        efficiency_list.append(efficiency)
        print(f"Parallel median time with {cores} cores: {p_median}, Speedup: {speedup}, Efficiency: {efficiency}")

    core_list = [cores for cores in range(1, core_count + 1)]
    speedup_list.insert(0, 0)
    plt.plot(core_list, speedup_list)
    plt.xlabel("Core Count")
    plt.ylabel("Speedup")
    plt.savefig("MP_speedup.png")
    plt.show()

    efficiency_list.insert(0, 1)
    plt.plot(core_list, efficiency_list)
    plt.xlabel("Core Count")
    plt.ylabel("Efficiency")
    plt.savefig("MP_efficiency.png")
    plt.show()
    
    image = plt.imshow(M, extent=(x_interval[0], x_interval[1], y_interval[0], y_interval[1]), interpolation="nearest")
    plt.title(f"Mandelbrot set {N}x{N}, {max_iter} max iterations.")
    plt.colorbar(image, orientation='vertical')
    plt.show()

