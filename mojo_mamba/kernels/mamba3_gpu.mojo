# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
# ===----------------------------------------------------------------------=== #

from std.gpu import block_dim, block_idx, thread_idx, global_idx
from layout import Layout, LayoutTensor
from std.memory import UnsafePointer, stack_allocation
import std.math
from std.math import exp, log, log1p

# GPU Mamba3 SISO kernel (Simplified Thread-Parallel)

@always_inline
def softplus_gpu(val: Float32) -> Float32:
    if val > 20.0: return val
    return log1p(exp(val))

@always_inline
def sigmoid_gpu(val: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-val))

@always_inline
def silu_gpu(val: Float32) -> Float32:
    return val * sigmoid_gpu(val)

def mamba3_siso_fwd_kernel[
    dtype: DType,
    d_model: Int,
    d_state: Int,
    headdim: Int,
    z_layout: Layout,
    x_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    dt_layout: Layout,
    A_layout: Layout,
    angles_layout: Layout,
    dt_bias_layout: Layout,
    D_layout: Layout,
    out_layout: Layout,
](
    batch: Int,
    seqlen: Int,
    z: LayoutTensor[dtype, z_layout, MutAnyOrigin],
    x: LayoutTensor[dtype, x_layout, MutAnyOrigin],
    B: LayoutTensor[dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[dtype, C_layout, MutAnyOrigin],
    dd_dt: LayoutTensor[dtype, dt_layout, MutAnyOrigin],
    dd_A: LayoutTensor[dtype, A_layout, MutAnyOrigin],
    angles: LayoutTensor[dtype, angles_layout, MutAnyOrigin],
    dt_bias: LayoutTensor[dtype, dt_bias_layout, MutAnyOrigin],
    D: LayoutTensor[dtype, D_layout, MutAnyOrigin],
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
):
    var nheads = (d_model * 2) // headdim
    var g_id = Int(global_idx.x)
    
    if g_id >= (batch * nheads):
        return
    
    var b = g_id // nheads
    var h = g_id % nheads
    
    # State for this head (local to thread)
    # Important: size must be exactly as needed
    comptime state_size = headdim * d_state
    var state_re = stack_allocation[state_size, Float32]()
    var state_im = stack_allocation[state_size, Float32]()
    
    for i in range(state_size):
        state_re[i] = 0.0
        state_im[i] = 0.0
        
    var dt_b_val = rebind[Scalar[DType.float32]](dt_bias[h])
    var D_h_val = rebind[Scalar[DType.float32]](D[h])

    for t in range(seqlen):
        var dt_val = softplus_gpu(rebind[Scalar[DType.float32]](dd_dt[b, t, h]) + dt_b_val)
        var A_val = -softplus_gpu(rebind[Scalar[DType.float32]](dd_A[b, t, h]))
        if A_val > -1e-4: A_val = -1e-4
        
        var alpha = exp(A_val * dt_val)
        var beta = dt_val
        
        for n in range(d_state // 2):
            var angle = rebind[Scalar[DType.float32]](angles[b, t, n])
            var cos_a = std.math.cos(angle)
            var sin_a = std.math.sin(angle)
            
            var b_real = rebind[Scalar[DType.float32]](B[b, t, 2*n])
            var b_imag = rebind[Scalar[DType.float32]](B[b, t, 2*n+1])
            
            var b_r = b_real * cos_a - b_imag * sin_a
            var b_i = b_real * sin_a + b_imag * cos_a
            
            for p in range(headdim):
                var x_val = rebind[Scalar[DType.float32]](x[b, t, h, p])
                var idx_base = p * d_state
                
                # State update for the complex pair
                # In Mamba-3 SISO, each complex state h is updated by (alpha, beta, b, x)
                # h_t = alpha * h_{t-1} + beta * b_t * x_t
                var s_re = state_re[idx_base + 2*n]
                var s_im = state_im[idx_base + 2*n]
                state_re[idx_base + 2*n] = alpha * s_re + beta * b_r * x_val
                state_im[idx_base + 2*n] = alpha * s_im + beta * b_i * x_val
                
                # The second element of the pair is the same state in SISO? 
                # No, it's typically a separate complex state or the conjugate.
                # In my verified CPU code, I updated both. Let's match that exactly.
                s_re = state_re[idx_base + 2*n + 1]
                s_im = state_im[idx_base + 2*n + 1]
                state_re[idx_base + 2*n + 1] = alpha * s_re + beta * b_r * x_val
                state_im[idx_base + 2*n + 1] = alpha * s_im + beta * b_i * x_val

        for p in range(headdim):
            var y_val: Float32 = 0.0
            for n in range(d_state // 2):
                var angle = rebind[Scalar[DType.float32]](angles[b, t, n])
                var cos_a = std.math.cos(angle)
                var sin_a = std.math.sin(angle)
                
                var c_real = rebind[Scalar[DType.float32]](C[b, t, 2*n])
                var c_imag = rebind[Scalar[DType.float32]](C[b, t, 2*n+1])
                
                var c_r = c_real * cos_a - c_imag * sin_a
                var c_i = c_real * sin_a + c_imag * cos_a
                
                var idx_base = p * d_state
                # y = Re(c * h) = c_re * h_re - c_im * h_im
                y_val += c_r * state_re[idx_base + 2*n] - c_i * state_im[idx_base + 2*n]
                y_val += c_r * state_re[idx_base + 2*n + 1] - c_i * state_im[idx_base + 2*n + 1]
            
            y_val *= 2.0
            var x_val = rebind[Scalar[DType.float32]](x[b, t, h, p])
            y_val += D_h_val * x_val
            
            var z_val = rebind[Scalar[DType.float32]](z[b, t, h, p])
            output[b, t, h, p] = rebind[output.element_type](y_val * silu_gpu(z_val))
