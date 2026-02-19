import numpy as np
import time

if __name__ == "__main__":
    N = 10000

    A = np.random.rand(N, N)
    # A = np.asfortranarray(A)

    for i in range(10):
        t_s = time.perf_counter()
        for i in range(N): 
            row_sum = np.sum(A[i, :])
        t_e = time.perf_counter()
        print(f"Row wise = {t_e - t_s}")

        t_s = time.perf_counter()
        for j in range(N): 
            column_sum = np.sum(A[:, j])
        t_e = time.perf_counter()
        print(f"Column wise = {t_e - t_s}")

