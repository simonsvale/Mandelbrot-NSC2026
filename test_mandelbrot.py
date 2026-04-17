# pytest -v
# pytest --cov=. -v

import pytest, pytest_cov
import numpy as np

# Import functions that should be tested.
from mandelbrot_naive import compute_mandelbrot_set as compute_mandelbrot_naive
from mandelbrot_numba import mandelbrot_point, compute_mandelbrot_set as compute_mandelbrot_nb
from mandelbrot_mp import mandelbrot_chunk

from mandelbrot_vectorized import compute_mandelbrot_set as compute_mandelbrot_vec, setup_variables

# Parametrization test.
TEST_CASES = [(0.0 + 0.0j, 100), (-1.0 + 1.0j, 2), (-1.5 + 0.01j, 13)]
@pytest.mark.parametrize("c, expected", TEST_CASES)
def test_numba_mandelbrot_point(c: np.complex128, expected: int):
    result = mandelbrot_point(c=c, max_iter=100)
    assert result == expected, f"Mandelbrot point should return {expected}, but returned {result}!"


def test_mandelbrot_mp_chunk_output():
    result = mandelbrot_chunk(row_start=0, row_end=1024, N=1, x_min=-2.0, x_max=1.0, y_min=-1.5, y_max=1.5, max_iter=100)
    assert result.all() < 100, f"All pixels in the upper row in the Mandelbrot set should be escape in less than 100 iterations!"


def test_mandelbrot_mp_chunk_output_length():
    result = mandelbrot_chunk(row_start=0, row_end=1024, N=1, x_min=-2.0, x_max=1.0, y_min=-1.5, y_max=1.5, max_iter=100)
    size = result.shape[0]
    assert size == 1024, f"Mandelbrot output should have size 1024, but has {size}!"


def test_mandelbrot_vec_correctness():
    x_interval = (-2.0, 1.0)
    y_interval = (-1.5, 1.5)

    N = 1024
    max_iter = 100

    # Compute the naive Mandelbrot set.
    mandelbrot_set_naive, _, _ = compute_mandelbrot_naive(x_interval, y_interval, x_res=N, y_res=N, max_iter=max_iter)

    # Compute the vectorized Mandelbrot set.
    Z, C, M, _, _ = setup_variables(x_interval, y_interval, x_res=N, y_res=N)
    mandelbrot_set_vec = compute_mandelbrot_vec(Z, C, M, max_iter)

    # Compare outputs.
    assert np.allclose(mandelbrot_set_naive, mandelbrot_set_vec) == True, "The naively computed Mandelbrot set is not equal to the vectorized one!"


def test_mandelbrot_numba_correctness():
    x_interval = (-2.0, 1.0)
    y_interval = (-1.5, 1.5)

    N = 1024
    max_iter = 100

    # Compute the naive Mandelbrot set.
    mandelbrot_set_naive, _, _ = compute_mandelbrot_naive(x_interval, y_interval, x_res=N, y_res=N, max_iter=max_iter)

    # Compute the numba Mandelbrot set.
    mandelbrot_set_nb, _, _ = compute_mandelbrot_nb(x_interval, y_interval, x_res=N, y_res=N, max_iter=max_iter, dtype=np.float64)

    for i in range(N):
        if not np.allclose(mandelbrot_set_naive[i], mandelbrot_set_nb[i]):
            print(i, mandelbrot_set_naive[i].sum(), mandelbrot_set_nb[i].sum())

    # Compare outputs.
    assert np.allclose(mandelbrot_set_naive, mandelbrot_set_nb) == True, "The naively computed Mandelbrot set is not equal to the multiprocessed one!"


def test_numerical_differences():
    x_interval = (-2.0, 1.0)
    y_interval = (-1.5, 1.5)

    N = 1024
    max_iter = 100

    mandelbrot_64, _, _ = compute_mandelbrot_nb(x_interval, y_interval, x_res=N, y_res=N, max_iter=max_iter, dtype=np.float64)
    mandelbrot_32, _, _ = compute_mandelbrot_nb(x_interval, y_interval, x_res=N, y_res=N, max_iter=max_iter, dtype=np.float32)

    assert not np.allclose(mandelbrot_64, mandelbrot_32), f"There should be numerical differences between float32 and float64!"


