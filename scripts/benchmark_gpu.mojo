from mojo_mamba.kernels.mamba3_gpu import mamba3_siso_fwd_kernel
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from std.sys import has_accelerator
from std.math import ceildiv
import std.time

def main() raises:
    print("Mojo Mamba3 GPU Benchmark")
    
    comptime has_gpu = has_accelerator()
    if not has_gpu:
        print("No GPU detected! This script requires a GPU.")
        return

    var ctx = DeviceContext()
    
    # Configuration
    comptime batch = 1
    comptime seqlen = 128
    comptime dim = 768
    comptime d_state = 128
    comptime headdim = 64
    comptime dtype = DType.float32
    
    comptime d_inner = dim * 2
    comptime nheads = d_inner // headdim
    
    # Layouts
    comptime z_layout = Layout.row_major(batch, seqlen, d_inner)
    comptime x_layout = Layout.row_major(batch, seqlen, d_inner)
    comptime B_layout = Layout.row_major(batch, seqlen, d_state)
    comptime C_layout = Layout.row_major(batch, seqlen, d_state)
    comptime dt_layout = Layout.row_major(batch, seqlen, nheads)
    comptime A_layout = Layout.row_major(batch, seqlen, nheads)
    comptime angles_layout = Layout.row_major(batch, seqlen, d_state // 2)
    comptime dt_bias_layout = Layout.row_major(nheads)
    comptime D_layout = Layout.row_major(nheads)
    comptime out_layout = Layout.row_major(batch, seqlen, d_inner)
    
    # Create buffers
    var z_buf = ctx.enqueue_create_buffer[dtype](comptime(z_layout.size()))
    var x_buf = ctx.enqueue_create_buffer[dtype](comptime(x_layout.size()))
    var B_buf = ctx.enqueue_create_buffer[dtype](comptime(B_layout.size()))
    var C_buf = ctx.enqueue_create_buffer[dtype](comptime(C_layout.size()))
    var dt_buf = ctx.enqueue_create_buffer[dtype](comptime(dt_layout.size()))
    var A_buf = ctx.enqueue_create_buffer[dtype](comptime(A_layout.size()))
    var angles_buf = ctx.enqueue_create_buffer[dtype](comptime(angles_layout.size()))
    var dt_bias_buf = ctx.enqueue_create_buffer[dtype](comptime(dt_bias_layout.size()))
    var D_buf = ctx.enqueue_create_buffer[dtype](comptime(D_layout.size()))
    var out_buf = ctx.enqueue_create_buffer[dtype](comptime(out_layout.size()))
    
    # Tensors
    var z = LayoutTensor[dtype, z_layout](z_buf)
    var x = LayoutTensor[dtype, x_layout](x_buf)
    var B = LayoutTensor[dtype, B_layout](B_buf)
    var C = LayoutTensor[dtype, C_layout](C_buf)
    var dd_dt = LayoutTensor[dtype, dt_layout](dt_buf)
    var dd_A = LayoutTensor[dtype, A_layout](A_buf)
    var angles = LayoutTensor[dtype, angles_layout](angles_buf)
    var dt_bias = LayoutTensor[dtype, dt_bias_layout](dt_bias_buf)
    var D = LayoutTensor[dtype, D_layout](D_buf)
    var output = LayoutTensor[dtype, out_layout](out_buf)
    
    # Bound kernel with all compile-time parameters
    comptime bound_kernel = mamba3_siso_fwd_kernel[
        dtype, dim, d_state, headdim,
        z_layout, x_layout, B_layout, C_layout,
        dt_layout, A_layout, angles_layout, 
        dt_bias_layout, D_layout, out_layout
    ]

    # Warmup
    print("Warming up GPU...")
    # 32 threads per head (one warp)
    var threads_per_block = 128
    for _ in range(5):
        ctx.enqueue_function[bound_kernel, bound_kernel](
            batch, seqlen, z, x, B, C, dd_dt, dd_A, angles, dt_bias, D, output,
            grid_dim=ceildiv(batch * nheads * 32, threads_per_block),
            block_dim=threads_per_block
        )
    ctx.synchronize()
    
    # Benchmark
    print("Running benchmark (10 iterations)...")
    var start = std.time.perf_counter_ns()
    
    for _ in range(10):
        ctx.enqueue_function[bound_kernel, bound_kernel](
            batch, seqlen, z, x, B, C, dd_dt, dd_A, angles, dt_bias, D, output,
            grid_dim=ceildiv(batch * nheads * 32, threads_per_block),
            block_dim=threads_per_block
        )
    ctx.synchronize()
    
    var end = std.time.perf_counter_ns()
    var duration = Float64(end - start) / 1e6 / 10.0
    print("Mojo Mamba3 GPU Avg duration: ", duration, "ms")
    print("BENCHMARK_RESULT:", duration)
