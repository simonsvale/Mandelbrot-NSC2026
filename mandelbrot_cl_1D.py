import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import time, statistics


def get_mandelbrot_program(ctx: cl.Context) -> cl.Program:

    program = cl.Program(ctx, """
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
                        
            zImagNew = (2.0f * zReal * zImag) + cImag[ci_idx];
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
    
    # Setup Mandelbrot variables.
    N = np.int32(8192)
    max_iter = np.int32(100)
    host_grid = np.zeros(N*N, dtype=np.int32)

    # Set floating point precision.
    float_type = np.float32

    if float_type == np.float32:
        implementation_type = "mandelbrotPixel32"
    elif float_type == np.float64:
        implementation_type = "mandelbrotPixel64"
    else:
        raise NotImplemented("Only float32 and float64 (double) has been implemented!")

    # Create linspace.
    x_interval: tuple[float, float] = (-2.0, 1.0)
    y_interval: tuple[float, float] = (-1.5, 1.5)

    host_c_real: np.ndarray = np.linspace(x_interval[0], x_interval[1], N, dtype=float_type)
    host_c_imag: np.ndarray = np.linspace(y_interval[0], y_interval[1], N, dtype=float_type)
    
    # Grid containing the iterations.
    dev_grid = cl.Buffer(context, cl.mem_flags.HOST_READ_ONLY, size=host_grid.nbytes)
    dev_c_real = cl.Buffer(context, cl.mem_flags.HOST_WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_c_real)
    dev_c_imag = cl.Buffer(context, cl.mem_flags.HOST_WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_c_imag)

    # Create the kernel from the program.
    kernel = cl.Kernel(program, implementation_type)
    kernel.set_arg(0, dev_c_real)
    kernel.set_arg(1, dev_c_imag)
    kernel.set_arg(2, dev_grid)
    kernel.set_arg(3, max_iter)
    kernel.set_arg(4, N)

    global_work_size = (int(N) * int(N),)
    local_work_size = (128,) # 128 is the best.

    runs = 5
    time_list = []
    for _ in range(runs):
        t_s = time.perf_counter()
        cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size)
        queue.finish()
        t_e = time.perf_counter()
        time_list.append(t_e - t_s)
    s_median = statistics.median(time_list)
    print("GPU median time: ", s_median)

    # Copy from host to device.
    cl.enqueue_copy(queue, host_grid, dev_grid, is_blocking=True)

    # Reshape from N * N to (N, N) and display.
    grid = host_grid.reshape(N, N)
    plt.imshow(grid)
    plt.show()