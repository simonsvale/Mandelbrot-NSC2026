from multiprocessing import Pool
import psutil, time, random

def estimate_pi_worker(N):
    # The points in the circle
    inlier_points = 0

    x, y = random.random(), random.random()

    for _ in range(N):
        x, y = random.random(), random.random()

        if x**2 + y**2 <= 1:
            inlier_points += 1

    return inlier_points


if __name__ == "__main__":
    core_count = psutil.cpu_count(logical=False)
    print("Physical Cores: ", core_count)

    # Number of points to use in monte carlo.
    N = 10_000_000
    
    est_count = [int(N / core_count) for _ in range(core_count)]
    pi_est = 0

    with Pool(processes=core_count) as pool:
        t_s = time.perf_counter()
        results = pool.map(estimate_pi_worker, est_count)
        pi_est = 4 * (sum(results) / N)
    t_e = time.perf_counter()
    print("Time took: ", t_e - t_s)
    print("Pi Estimate: ", pi_est)