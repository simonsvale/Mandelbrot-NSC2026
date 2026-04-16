import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Define the seahorse valley region.
    x_interval = [-0.7530, -0.7490]
    y_interval = [0.0990, 0.1030]

    x_res = 1024
    y_res = 1024
    max_iter = 1000 # 1000 iterations instead of 100 because we are in the seahorse valley region.

    # Divergence metric.
    tau = 0.01

    x_values = np.linspace(x_interval[0], x_interval[1], x_res)
    y_values = np.linspace(y_interval[0], y_interval[1], y_res)

    # Create initial C values.
    C64 = (x_values[np.newaxis, :] + 1j * y_values[:, np.newaxis]).astype(np.complex128)
    C32 = C64.astype(np.complex64)

    # Create two mandelbrot grids of type: complex128 and complex64.
    grid64 = np.zeros_like(C64)
    grid32 = np.zeros_like(C32)

    # Create divergence map.
    divergence_map = np.full((x_res, y_res), max_iter, dtype=np.int64)
    active = np.ones((x_res, y_res), dtype=bool)

    for iter in range(max_iter):
        # Check if all pixels have converged.
        if not active.any():
            break

        # Calculate the mandelbrot pixel for all non diverged pixels.
        grid32[active] = grid32[active]**2 + C32[active]
        grid64[active] = grid64[active]**2 + C64[active]

        # Calculate the numerical difference between the pixels.
        diff_real = np.abs(grid32.real.astype(np.float64) - grid64.real)
        diff_imag = np.abs(grid32.imag.astype(np.float64) - grid64.imag)
        diff_total = diff_real + diff_imag

        # Check if any cells have diverged this iteration.
        # A pixel has diverged if the numerical difference is larger than the threshold tau.
        diverged = active & (diff_total > tau)
        divergence_map[diverged] = iter # Set the iteration that the pixels diverged.
        active[diverged] = False
    
    # Plot the trajectory divergence map.
    plt.imshow(divergence_map, cmap="plasma", origin="lower", extent=[x_interval[0], x_interval[1], y_interval[0], y_interval[1]])
    plt.colorbar(label="Iterations before divergence")
    plt.title(f"Trajectory divergence (tau={tau})")
    plt.show()