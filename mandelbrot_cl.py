import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import time, statistics, math


def get_mandelbrot_program(ctx: cl.Context) -> cl.Program:

    program = cl.Program(ctx, """
    __kernel void mandelbrotPixel(__global float *cReal, __global float *cImag, __global int *grid, const int MaximumIterations, const int N) {

        // Get id.            
        int idx_x = get_global_id(0);
        int idx_y = get_global_id(1);

        // Create variables.
        float zReal = 0.0f;
        float zImag = 0.0f;
        float zImagNew = 0.0f;

        for (int iter = 0; iter <= MaximumIterations; iter++) {
                        
            zImagNew = (2.0f * zReal * zImag) + cImag[idx_x * N + idx_y];
            zReal = (zReal*zReal) - (zImag*zImag) + cReal[idx_x * N + idx_y];
            zImag = zImagNew;

            // Check if we escape.
            if (((zReal*zReal) + (zImag*zImag)) > 4.0f) {      
                grid[idx_x * N + idx_y] = iter;
                return;
            }
        }    
        // Allocate max_iter iterations.
        grid[idx_x * N + idx_y] = MaximumIterations;
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
    N = np.int32(8192)

    max_iter = np.int32(100)

    host_grid = np.zeros((N, N), dtype=np.int32)

    # Create linspace.
    x_interval: tuple[float, float] = (-2.0, 1.0)
    y_interval: tuple[float, float] = (-1.5, 1.5)

    x: np.ndarray = np.linspace(x_interval[0], x_interval[1], N, dtype=np.float32)
    y: np.ndarray = np.linspace(y_interval[0], y_interval[1], N, dtype=np.float32)

    # Create arrays.
    host_c_real, host_c_imag = np.meshgrid(x, y)
    
    # Grid containing the iterations.
    dev_grid = cl.Buffer(context, cl.mem_flags.HOST_READ_ONLY, size=host_grid.nbytes)
    dev_c_real = cl.Buffer(context, cl.mem_flags.HOST_WRITE_ONLY, size=host_c_real.nbytes)
    dev_c_imag = cl.Buffer(context, cl.mem_flags.HOST_WRITE_ONLY, size=host_c_imag.nbytes)

    # Create the kernel from the program and set kernel arguments.
    kernel = cl.Kernel(program, "mandelbrotPixel")
    kernel.set_arg(0, dev_c_real)
    kernel.set_arg(1, dev_c_imag)
    kernel.set_arg(2, dev_grid)
    kernel.set_arg(3, max_iter)
    kernel.set_arg(4, N)

    # Write buffers to GPU.
    cl.enqueue_copy(queue, dest=dev_c_real, src=host_c_real)
    cl.enqueue_copy(queue, dest=dev_c_imag, src=host_c_imag)

    # Get local range.
    max_work_group_size = int(math.floor(math.sqrt(device.max_work_group_size)))
    local_range = (max_work_group_size, max_work_group_size)

    # Get global range.
    global_range_1D = int(((N + max_work_group_size - 1) // max_work_group_size) * max_work_group_size)
    global_range = (global_range_1D, global_range_1D)
    print(global_range, local_range)
    runs = 5
    time_list = []
    for _ in range(runs):
        t_s = time.perf_counter()
        cl.enqueue_nd_range_kernel(queue, kernel, global_range, local_range)
        queue.finish()
        t_e = time.perf_counter()
        time_list.append(t_e - t_s)
    s_median = statistics.median(time_list)
    print("GPU median time: ", s_median)
    cl.enqueue_nd_range_kernel(queue, kernel, global_work_size=global_range, local_work_size=local_range)
    queue.finish()

    # Copy from host to device.
    cl.enqueue_copy(queue, dest=host_grid, src=dev_grid)

    # Release memory.
    dev_grid.release()
    dev_c_imag.release()
    dev_c_real.release()

    # Display grid.
    plt.imshow(host_grid)
    plt.show()

    
