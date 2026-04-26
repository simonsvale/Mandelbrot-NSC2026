import matplotlib.pyplot as plt


if __name__ == "__main__":
    implementations1024 = ["Naïve", "Vectorized", "NJIT Hybrid", "NJIT", "NJIT Parallel", "Multiprocessing", "Dask Local", "OpenCL float32", "OpenCL float64"]
    runtime1024 = [39.858, 1.2566, 1.1809, 0.1699, 0.0191, 0.0171, 0.1782, 0.0045, 0.0146]

    bar = plt.bar(implementations1024, runtime1024)
    plt.title("Performance Comparison")
    plt.xlabel("Implementations")
    plt.ylabel("Logarithmic runtime [s]")
    plt.yscale("log")
    plt.bar_label(bar, fmt='%.4f')
    plt.show()


    implementations8192 = ["NJIT Parallel", "Multiprocessing", "Dask Local", "Dask Distributed", "OpenCL float32", "OpenCL float64"]
    runtime8192 = [1.1353, 0.5531, 2.7796, 1.7715, 0.1119, 0.6745]

    bar = plt.bar(implementations8192, runtime8192)
    plt.title("Performance Comparison")
    plt.xlabel("Implementations")
    plt.ylabel("Logarithmic runtime [s]")
    plt.yscale("log")
    plt.bar_label(bar, fmt='%.4f')
    plt.show()