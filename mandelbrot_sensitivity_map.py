import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy.typing import NDArray

def escape_count(C: NDArray[np.complex128], max_iter: int) -> NDArray[np.int32]:
    """
    Calculates the escape count for the mandelbrot set.

    Parameters
    -----------
    C : NDArray[np.complex128]
        An array of complex constant with the np.complex128 type.

    max_iter : int 
        The maximum number of iterations to calculate the mandelbrot point at the complex value c for.

    Returns
    --------
    cnt : NDArray[np.int32]
        An array with the mandelbrot escape counts.
    """
    z = np.zeros_like(C)
    cnt = np.full(C.shape, max_iter, dtype=np.int32)
    esc = np.zeros(C.shape, dtype=bool)

    for iter in range(max_iter):
        z[~esc] = z[~esc]**2 + C[~esc]

        newly = ~esc & (z.real**2 + z.imag**2 > 4.0)

        cnt[newly] = iter
        esc[newly] = True
    return cnt


if __name__ == "__main__":

    # Define the seahorse valley region.
    x_interval = (-0.7530, -0.7490)
    y_interval = (0.0990, 0.1030)

    x_res = 1024
    y_res = 1024
    max_iter = 1000 # 1000 iterations instead of 100 because we are in the seahorse valley region.

    x_values = np.linspace(x_interval[0], x_interval[1], x_res)
    y_values = np.linspace(y_interval[0], y_interval[1], y_res)

    C = (x_values[np.newaxis, :] + 1j * y_values[:, np.newaxis]).astype(np.complex128)

    eps32 = float(np.finfo(np.float32).eps)
    print(eps32)
    delta = np.maximum(eps32 * np.abs(C), 1e-10)

    # Calculate n(c)
    n_base = escape_count(C, max_iter)
    n_perturb = escape_count(C + delta, max_iter)

    # Calculate ∆n.
    dn = np.abs(n_base - n_perturb)

    # Calculate κ(c).
    kappa = np.where(n_base > 0, dn / (eps32 * n_base), np.nan)
    cmap_k = plt.cm.hot.copy()
    cmap_k.set_bad("0.25")
    vmax = np.nanpercentile(kappa, 99.9)

    plt.imshow(
        kappa, 
        cmap=cmap_k, 
        origin="lower", 
        extent=(-0.7530, -0.7490, 0.0990, 0.1030), 
        norm=LogNorm(vmin=1, vmax=vmax)
    )
    plt.colorbar(label=f"Log scale (κ >= 1)")
    plt.title(f"Condition number approx κ(c) = ∆n / (ε32 * n(c))")
    plt.show()