from mojo_mamba.kernels.mamba3_gpu import mamba3_siso_fwd_kernel
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor
from std.sys import has_accelerator
from std.python import Python, PythonObject
from std.math import ceildiv
import std.time

def main() raises:
    print("Mamba3 GPU Numerical Correctness Test")
    
    comptime has_gpu = has_accelerator()
    if not has_gpu:
        print("No GPU detected!")
        return

    var ctx = DeviceContext()
    var np = Python.import_module("numpy")
    
    # Configuration (Must match what was generated in reference_mamba3.py)
    comptime batch = 1
    comptime seqlen = 16
    comptime dim = 64
    comptime d_state = 16
    comptime headdim = 8
    comptime dtype = DType.float32
    
    comptime d_inner = dim * 2
    comptime nheads = d_inner // headdim
    
    # Layouts
    comptime z_layout = Layout.row_major(batch, seqlen, nheads, headdim)
    comptime x_layout = Layout.row_major(batch, seqlen, nheads, headdim)
    comptime B_layout = Layout.row_major(batch, seqlen, d_state)
    comptime C_layout = Layout.row_major(batch, seqlen, d_state)
    comptime dt_layout = Layout.row_major(batch, seqlen, nheads)
    comptime A_layout = Layout.row_major(batch, seqlen, nheads)
    comptime angles_layout = Layout.row_major(batch, seqlen, d_state // 2)
    comptime dt_bias_layout = Layout.row_major(nheads)
    comptime D_layout = Layout.row_major(nheads)
    comptime out_layout = Layout.row_major(batch, seqlen, nheads, headdim)
    
    # Helper to load and copy to device
    def load_to_device(path: String, layout_size: Int) raises -> DeviceBuffer[dtype]:
        var npy_arr = np.load(path).flatten()
        var dev_buf = ctx.enqueue_create_buffer[dtype](layout_size)
        var host_buf = ctx.enqueue_create_host_buffer[dtype](layout_size)
        var py_len = Int(py=len(npy_arr))
        var limit = min(layout_size, py_len)
        for i in range(limit):
            host_buf[i] = Float32(py=npy_arr[i])
        ctx.enqueue_copy(dst_buf=dev_buf, src_buf=host_buf)
        return dev_buf

    # Load data
    var z_buf = load_to_device("data/split_z.npy", comptime(z_layout.size()))
    var x_buf = load_to_device("data/split_x.npy", comptime(x_layout.size()))
    var B_buf = load_to_device("data/split_B.npy", comptime(B_layout.size()))
    var C_buf = load_to_device("data/split_C.npy", comptime(C_layout.size()))
    var dt_buf = load_to_device("data/split_dt.npy", comptime(dt_layout.size()))
    var A_buf = load_to_device("data/split_A.npy", comptime(A_layout.size()))
    var angles_buf = load_to_device("data/split_angles.npy", comptime(angles_layout.size()))
    var dt_bias_buf = load_to_device("data/param_dt_bias.npy", comptime(dt_bias_layout.size()))
    var D_buf = load_to_device("data/param_D.npy", comptime(D_layout.size()))
    var out_buf = ctx.enqueue_create_buffer[dtype](comptime(out_layout.size()))
    
    # Reference output
    var gated_y_ref = np.load("data/gated_y.npy").flatten()
    
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
    
    # Launch
    comptime bound_kernel = mamba3_siso_fwd_kernel[
        dtype, dim, d_state, headdim,
        z_layout, x_layout, B_layout, C_layout,
        dt_layout, A_layout, angles_layout, 
        dt_bias_layout, D_layout, out_layout
    ]
    
    print("Launching GPU Kernel...")
    var grid_dim = ceildiv(batch * nheads, 128)
    var block_dim = 128
    ctx.enqueue_function[bound_kernel, bound_kernel](
        batch, seqlen, z, x, B, C, dd_dt, dd_A, angles, dt_bias, D, output,
        grid_dim=grid_dim,
        block_dim=block_dim
    )
    ctx.synchronize()
    
    # Verify
    print("\nResults Comparison (against gated_y):")
    var max_err: Float32 = 0.0
    var total_elements = batch * seqlen * nheads * headdim
    with out_buf.map_to_host() as host:
        for i in range(total_elements):
            var val = host[i]
            var ref_val = Float32(py=gated_y_ref[i])
            var diff = std.math.abs(val - ref_val)
            if diff > max_err: max_err = diff
            if i < 5:
                print("Index", i, "| Mojo:", val, "| Ref:", ref_val, "| Diff:", diff)
    
    print("\nMax error across all", total_elements, "elements:", max_err)
    if max_err < 1e-4:
        print("GPU Numerical check PASSED!")
    else:
        print("GPU Numerical check FAILED!")
    
    print("Verification complete.")
