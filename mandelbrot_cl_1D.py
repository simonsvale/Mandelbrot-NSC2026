import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import time, statistics
from numpy.typing import NDArray


def get_mandelbrot_program(context: cl.Context) -> cl.Program:
    """
    Computes the Mandelbrot set using OpenCL given an interval on the real and imaginary axes, 
    the resolution of the Mandelbrot set N, the max number of iterations before a point/pixel escapes and the floating point type to use.

    Parameters
    -----------
    context : cl.Context
        An OpenCL context.

    Returns
    --------
    program : cl.Program
        An OpenCL program with the mandelbrot pixel kernel for both float32 and float64.
    """
    program = cl.Program(context, """      
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable         

    __kernel void mandelbrotPixel32(__global float *cReal, __global float *cImag, __global int *grid, const int MaximumIterations, const int N) {

        // Get id.            
        int idx = get_global_id(0);
        int cr_idx = idx % N;
        int ci_idx = idx / N;

        // Pre allocate max_iter iterations.
        grid[idx] = MaximumIterations;
        
        // Create variables.
        float zReal = 0.0;
        float zImag = 0.0;
        float zImagNew = 0.0;

        for (int iter = 0; iter <= MaximumIterations; iter++) {
                        
            zImagNew = (2.0f * zReal * zImag) + cImag[ci_idx];
            zReal = (zReal*zReal) - (zImag*zImag) + cReal[cr_idx];
            zImag = zImagNew;

            // Check if we escape.
            if (((zReal*zReal) + (zImag*zImag)) > 4.0f) {      
                grid[idx] = iter;
                return;
            }
        }
    }

    __kernel void mandelbrotPixel64(__global double *cReal, __global double *cImag, __global int *grid, const int MaximumIterations, const int N) {

        // Get id.            
        int idx = get_global_id(0);
        int cr_idx = idx % N;
        int ci_idx = idx / N;

        // Pre allocate max_iter iterations.
        grid[idx] = MaximumIterations;
        
        // Create variables.
        double zReal = 0.0;
        double zImag = 0.0;
        double zImagNew = 0.0;

        for (int iter = 0; iter <= MaximumIterations; iter++) {
                        
            zImagNew = (2.0 * zReal * zImag) + cImag[ci_idx];
            zReal = (zReal*zReal) - (zImag*zImag) + cReal[cr_idx];
            zImag = zImagNew;

            // Check if we escape.
            if (((zReal*zReal) + (zImag*zImag)) > 4.0) {      
                grid[idx] = iter;
                return;
            }
        }
    }
    """).build()

    return program


def compute_mandelbrot(
        context: cl.Context, 
        queue: cl.CommandQueue, 
        program: cl.Program, 
        x_interval: tuple[float, float], 
        y_interval: tuple[float, float], 
        N: np.int32, 
        max_iter: np.int32, 
        dtype: np.float32 | np.float64
    ) -> NDArray[np.int32]:
    """
    Computes the Mandelbrot set using OpenCL given an interval on the real and imaginary axes, 
    the resolution of the Mandelbrot set N, the max number of iterations before a point/pixel escapes and the floating point type to use.

    Parameters
    -----------
    context : cl.Context
        An OpenCL context.

    queue : cl.CommandQueue
        An OpenCL command queue.

    program : cl.Program
        An OpenCL program.

    x_interval : tuple[float, float]
        An interval on the real axis, the first index in the tuple is the lower bound and the second is the upper bound.

    y_interval : tuple[float, float]
        An interval on the imaginary axis, the first index in the tuple is the lower bound and the second is the upper bound.

    N : np.int32
        The resolution of the mandelbrot set on the x- and y-axis.

    max_iter : np.int32
        The maximum number of iterations before a point/pixel escapes..

    Returns
    --------
    grid : NDArray[np.int32]
        A numpy array containing the Mandelbrot grid with the shape: (N, N).
    """
    if dtype == np.float32:
        implementation_type = "mandelbrotPixel32"
    elif dtype == np.float64:
        implementation_type = "mandelbrotPixel64"
    else:
        raise NotImplemented("Only float32 and float64 (double) has been implemented!")
    
    # Create output grid.
    host_grid = np.zeros(N*N, dtype=np.int32)

    # Create linspace.
    host_c_real: np.ndarray = np.linspace(x_interval[0], x_interval[1], N, dtype=dtype)
    host_c_imag: np.ndarray = np.linspace(y_interval[0], y_interval[1], N, dtype=dtype)
    
    # Grid containing the iterations.
    dev_grid = cl.Buffer(context, cl.mem_flags.HOST_READ_ONLY, size=host_grid.nbytes)

    # Grid containing the real and imaginary part of the complex constant C.
    dev_c_real = cl.Buffer(context, cl.mem_flags.HOST_WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_c_real)
    dev_c_imag = cl.Buffer(context, cl.mem_flags.HOST_WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_c_imag)

    # Create the kernel from the program, this has to be done during runtime.
    kernel = cl.Kernel(program, implementation_type)
    kernel.set_arg(0, dev_c_real)
    kernel.set_arg(1, dev_c_imag)
    kernel.set_arg(2, dev_grid)
    kernel.set_arg(3, max_iter)
    kernel.set_arg(4, N)

    global_work_size = (int(N) * int(N),)
    local_work_size = (128,) # 128 is the best.

    cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size)
    queue.finish()

    # Copy from host to device.
    cl.enqueue_copy(queue, host_grid, dev_grid, is_blocking=True)

    # Reshape from N * N to (N, N) and display.
    grid = host_grid.reshape(N, N)

    return grid


if __name__ == "__main__":

    # Get the platforms
    platforms = cl.get_platforms()
    platform = platforms[0]

    # Get devices for the platform.
    devices = platform.get_devices()
    device = devices[0]

    # Create OpenCL context.
    context = cl.Context([device])

    # Create command queue.
    queue = cl.CommandQueue(context)

    # Create CL program.
    program = get_mandelbrot_program(context)

    # Setup variables.
    N = np.int32(8192)
    max_iter = np.int32(100)
    dtype = np.float32
    x_interval: tuple[float, float] = (-2.0, 1.0)
    y_interval: tuple[float, float] = (-1.5, 1.5)
    
    # Warmup using JIT, Nvidia uses NVCC. I think it is HIP if AMD is used.
    # I don't think this matters if a GPU is not available.
    grid = compute_mandelbrot(context, queue, program, x_interval, y_interval, N, max_iter, dtype)

    runs = 50
    time_list = []
    for _ in range(runs):
        t_s = time.perf_counter()
        grid = compute_mandelbrot(context, queue, program, x_interval, y_interval, N, max_iter, dtype)
        t_e = time.perf_counter()
        time_list.append(t_e - t_s)
    s_median = statistics.median(time_list)
    print("GPU median time: ", s_median)

    # Show mandelbrot set.
    #plt.imshow(grid)
    #plt.show()