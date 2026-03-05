from multiprocessing import Pool
import psutil, time, random, statistics
import matplotlib.pyplot as plt

def estimate_pi_worker(N):
    # The points in the circle
    inlier_points = 0

    for _ in range(N):
        x, y = random.random(), random.random()

        if x**2 + y**2 <= 1:
            inlier_points += 1

    return inlier_points


def serial_estimate_pi(N):

    inlier_points = 0

    for _ in range(N):
        x, y = random.random(), random.random()

        if x**2 + y**2 <= 1:
            inlier_points += 1

    pi_est = 4 * (inlier_points / N)
    return pi_est


def sweep(N, core_count, run_count=3):

    processors = [cores for cores in range(1, core_count + 1)]
    speedup_list = []
    efficiency_list = []

    # Get serial runtime.
    serial_time = []
    for _ in range(run_count):
        t_s = time.perf_counter()
        serial_estimate_pi(N)
        t_e = time.perf_counter()
        serial_time.append(t_e - t_s)
    serial_median_time = statistics.median(serial_time)

    # Run on 1 to core_count processors.
    for cores in range(1, core_count + 1):
        time_list = []
        est_count = [int(N / cores) for _ in range(cores)]
        for _ in range(run_count):
            with Pool(processes=cores) as pool:
                t_s = time.perf_counter()
                results = pool.map(estimate_pi_worker, est_count)
                pi_est = 4 * (sum(results) / N)
            t_e = time.perf_counter()
            time_list.append(t_e - t_s)
        # Find the median time for all runs for a specific processor count.
        parallel_median_time = statistics.median(time_list)
        speedup = serial_median_time / parallel_median_time
        
        speedup_list.append(speedup)
        efficiency_list.append(speedup / cores)
        if cores != 1:
            implied_serial_fraction = ((1 / speedup) - (1 / cores)) / (1 - (1 / cores))
            print(f"Serial fraction for {cores} cores: {implied_serial_fraction}")

    # Speedup plot.
    plt.plot(processors, speedup_list)
    plt.ylabel("Speedup")
    plt.xlabel("Processor Count")
    plt.show()

    # Efficiency plot.
    plt.plot(processors, efficiency_list)
    plt.ylabel("Efficiency")
    plt.xlabel("Processor Count")
    plt.show()


if __name__ == "__main__":
    core_count = psutil.cpu_count(logical=False)
    print("Physical Cores: ", core_count)

    # Number of points to use in monte carlo.
    N = 10_000_000

    sweep(N, core_count, 3)
    
    """
    t_s = time.perf_counter()
    s_pi_est = serial_estimate_pi(N)
    t_e = time.perf_counter()
    print("Serial Time took: ", t_e - t_s, "s")
    print("Pi Estimate: ", s_pi_est)
    
    est_count = [int(N / core_count) for _ in range(core_count)]
    pi_est = 0

    with Pool(processes=core_count) as pool:
        t_s = time.perf_counter()
        results = pool.map(estimate_pi_worker, est_count)
        pi_est = 4 * (sum(results) / N)
    t_e = time.perf_counter()
    print("Parallel Time took: ", t_e - t_s, "s")
    print("Pi Estimate: ", pi_est)
    """
