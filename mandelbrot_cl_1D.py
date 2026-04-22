import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import time, statistics


def get_mandelbrot_program(ctx: cl.Context) -> cl.Program:

    program = cl.Program(ctx, """
    __kernel void mandelbrotPixel(__global float *cReal, __global float *cImag, __global int *grid, const int MaximumIterations, const int N) {

        // Get id.            
        int idx = get_global_id(0);
        int cr_idx = idx % N;
        int ci_idx = idx / N;

        // Pre allocate max_iter iterations.
        grid[idx] = MaximumIterations;
        
        // Create variables.
        float zReal = 0.0f;
        float zImag = 0.0f;
        float zImagNew = 0.0f;

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
    
    # Setup buffers.
    N = 8192
    work_size = (N * N)

    max_iter = np.int32(100)

    host_grid = np.zeros(work_size, dtype=np.int32)

    # Create linspace.
    x_interval: tuple[float, float] = (-2.0, 1.0)
    y_interval: tuple[float, float] = (-1.5, 1.5)

    host_c_real: np.ndarray = np.linspace(x_interval[0], x_interval[1], N, dtype=np.float32)
    host_c_imag: np.ndarray = np.linspace(y_interval[0], y_interval[1], N, dtype=np.float32)
    
    # Grid containing the iterations.
    dev_grid = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=host_grid.nbytes)
    dev_c_real = cl.Buffer(context, cl.mem_flags.HOST_WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_c_real)
    dev_c_imag = cl.Buffer(context, cl.mem_flags.HOST_WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_c_imag)

    # Create the kernel from the program.
    kernel = program.mandelbrotPixel
    #local_size = 8

    runs = 5

    time_list = []
    for _ in range(runs):
        t_s = time.perf_counter()
        kernel(queue, host_grid.shape, None, dev_c_real, dev_c_imag, dev_grid, max_iter, np.int32(N))
        queue.finish()
        t_e = time.perf_counter()
        time_list.append(t_e - t_s)
    s_median = statistics.median(time_list)
    print("GPU median time: ", s_median)

    # Copy from host to device.
    cl.enqueue_copy(queue, host_grid, dev_grid)
    queue.finish()

    # Reshape from N * N to (N, N)
    grid = host_grid.reshape(N, N)

    # Display grid.
    #plt.imshow(grid)
    #plt.show()