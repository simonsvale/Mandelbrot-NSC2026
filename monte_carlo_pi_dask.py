import dask, random, time, statistics, psutil, numba

from dask import delayed
from dask.distributed import Client, LocalCluster

def monte_carlo_chunk(n_samples):
    inside = 0
    for _ in range(n_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside += 1
    return inside


if __name__ == "__main__":

    run_count = 3
    core_count = psutil.cpu_count(logical=True)
    threads = 1

    total = 100_000_000
    n_chunks = core_count # Have as many chunks as there are cores available.
    samples = total // n_chunks

    """
    t0 = time.perf_counter()
    results = [monte_carlo_chunk(samples) for _ in range(n_chunks)]
    t_serial = time.perf_counter() - t0
    print(f"Serial:{t_serial:.3f}s pi={4*sum(results)/total:.4f}")
    """

    # Create a cluster and a client.
    cluster = LocalCluster(n_workers=core_count, threads_per_worker=threads)
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")

    # It is possible to rescale the cluster and the amount of workers the client is using.
    #cluster.scale(4)
    #client.wait_for_workers(4)

    # Generate the task graph for the scheduler.
    tasks = [delayed(monte_carlo_chunk)(samples) for _ in range(n_chunks)]

    # Visualize the generated task graph.
    dask.visualize(*tasks, filename="pi_task_graph.png")

    # Find the median time.
    time_list = []
    for _ in range(run_count):
        t0 = time.perf_counter()
        results = dask.compute(*tasks)
        t_dask = time.perf_counter() - t0
        time_list.append(t_dask)
    median_time = statistics.median(time_list)
    print(f"Dask Median: {median_time:.4f}s, (min = {min(time_list):.4f}, max = { max(time_list):.4f})")

    # Graceful shutdown.
    client.close()
    cluster.close()