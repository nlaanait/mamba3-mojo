# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
# ===----------------------------------------------------------------------=== #

from std.gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from std.utils.index import IndexList
from std.memory import UnsafePointer, alloc
import std.math
from std.math import ceildiv, exp, exp2, rsqrt, cos, sin, log

# Mamba3 helper functions

@always_inline
def sigmoid(val: Float32) -> Float32:
    if val > 20.0: return 1.0
    if val < -20.0: return 0.0
    return 1.0 / (1.0 + std.math.exp(-val))

@always_inline
def softplus(val: Float32) -> Float32:
    if val > 20.0: return val
    if val < -20.0: return std.math.exp(val)
    return std.math.log1p(std.math.exp(val))

@always_inline
def silu(val: Float32) -> Float32:
    return val * sigmoid(val)

def mamba3_siso_forward_logic[
    dtype: DType,
    d_model: Int,
    d_state: Int,
    headdim: Int,
](
    batch: Int,
    seqlen: Int,
    z: UnsafePointer[Float32, MutExternalOrigin],
    x: UnsafePointer[Float32, MutExternalOrigin],
    B: UnsafePointer[Float32, MutExternalOrigin],
    C: UnsafePointer[Float32, MutExternalOrigin],
    dd_dt: UnsafePointer[Float32, MutExternalOrigin],
    dd_A: UnsafePointer[Float32, MutExternalOrigin],
    trap: UnsafePointer[Float32, MutExternalOrigin],
    angles: UnsafePointer[Float32, MutExternalOrigin],
    dt_bias: UnsafePointer[Float32, MutExternalOrigin],
    D: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
):
    """
    Actual Mamba3 SISO forward pass logic.
    """
    var nheads = (d_model * 2) // headdim 
    var d_inner = d_model * 2
    
    for b in range(batch):
        # State per head: (headdim, d_state)
        var state = alloc[Float32](nheads * headdim * d_state)
        for i in range(nheads * headdim * d_state): state[i] = 0.0

        for t in range(seqlen):
            for h in range(nheads):
                var dt = softplus(dd_dt[b * seqlen * nheads + t * nheads + h] + dt_bias[h])
                var A = -softplus(dd_A[b * seqlen * nheads + t * nheads + h])
                if A > -1e-4: A = -1e-4
                
                var alpha = exp(A * dt)
                var beta = dt 
                
                # Load angles for RoPE (assuming num_rope_angles matches d_state or similar)
                # For SISO, angles might be per-head.
                
                for n in range(d_state // 2):
                    # RoPE on B and C
                    var angle = angles[b * seqlen * (d_state//2) + t * (d_state//2) + n]
                    var cos_a = cos(angle)
                    var sin_a = sin(angle)
                    
                    var b_real = B[b * seqlen * d_state + t * d_state + 2*n]
                    var b_imag = B[b * seqlen * d_state + t * d_state + 2*n + 1]
                    var c_real = C[b * seqlen * d_state + t * d_state + 2*n]
                    var c_imag = C[b * seqlen * d_state + t * d_state + 2*n + 1]
                    
                    # Rotated B and C
                    var b_r = b_real * cos_a - b_imag * sin_a
                    var b_i = b_real * sin_a + b_imag * cos_a
                    var c_r = c_real * cos_a - c_imag * sin_a
                    var c_i = c_real * sin_a + c_imag * cos_a
                    
                    for p in range(headdim):
                        var x_val = x[b * seqlen * d_inner + t * d_inner + h * headdim + p]
                        
                        # Real state part
                        var state_idx_r = h * (headdim * d_state) + p * d_state + 2*n
                        state[state_idx_r] = alpha * state[state_idx_r] + beta * b_r * x_val
                        
                        # Imaginary state part
                        var state_idx_i = h * (headdim * d_state) + p * d_state + 2*n + 1
                        state[state_idx_i] = alpha * state[state_idx_i] + beta * b_i * x_val
                
                # Compute output after all d_state updates
                for p in range(headdim):
                    var x_val = x[b * seqlen * d_inner + t * d_inner + h * headdim + p]
                    var y_val: Float32 = 0.0
                    for n in range(d_state // 2):
                        # Use rotated C to project state
                        # Note: In real Mamba3 this might be complex dot product
                        var angle = angles[b * seqlen * (d_state//2) + t * (d_state//2) + n]
                        var cos_a = cos(angle)
                        var sin_a = sin(angle)
                        var c_real = C[b * seqlen * d_state + t * d_state + 2*n]
                        var c_imag = C[b * seqlen * d_state + t * d_state + 2*n + 1]
                        var c_r = c_real * cos_a - c_imag * sin_a
                        var c_i = c_real * sin_a + c_imag * cos_a
                        
                        y_val += c_r * state[h * (headdim * d_state) + p * d_state + 2*n]
                        y_val += c_i * state[h * (headdim * d_state) + p * d_state + 2*n + 1]
                    
                    # SISO factor 2 for real part of complex state
                    y_val *= 2.0
                    
                    # Add skip connection D
                    y_val += D[h] * x_val
                    
                    # Gating with Z
                    var z_val = z[b * seqlen * d_inner + t * d_inner + h * headdim + p]
                    output[b * seqlen * d_inner + t * d_inner + h * headdim + p] = y_val * silu(z_val)
        
        state.free()
