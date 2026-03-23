from layout import Layout, LayoutTensor
from std.utils.index import IndexList
from std.memory import UnsafePointer, alloc
import std.math
from mojo_mamba.kernels.mamba3_kernels import mamba3_siso_forward_logic

# Mamba3 module in Mojo

struct Mamba3[
    dtype: DType,
    d_model: Int,
    d_state: Int = 128,
    headdim: Int = 64,
]:
    var out_proj_weight: UnsafePointer[Float32, MutExternalOrigin]
    var dt_bias: UnsafePointer[Float32, MutExternalOrigin]
    var D: UnsafePointer[Float32, MutExternalOrigin]

    def __init__(out self):
        self.out_proj_weight = UnsafePointer[Float32, MutExternalOrigin]()
        self.dt_bias = UnsafePointer[Float32, MutExternalOrigin]()
        self.D = UnsafePointer[Float32, MutExternalOrigin]()

    def forward(
        self, 
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
        output: UnsafePointer[Float32, MutExternalOrigin],
    ) -> None:
        var d_inner = Self.d_model * 2
        var y_kernel = alloc[Float32](batch * seqlen * d_inner)
        
        mamba3_siso_forward_logic[
            Self.dtype, Self.d_model, Self.d_state, Self.headdim
        ](
            batch, seqlen, z, x, B, C, dd_dt, dd_A, trap, angles,
            self.dt_bias, self.D, y_kernel
        )
        
        # Apply out_proj: (batch*seqlen, d_inner) @ (d_model, d_inner).T -> (batch*seqlen, d_model)
        for i in range(batch * seqlen):
            for j in range(Self.d_model):
                var val: Float32 = 0.0
                for k in range(d_inner):
                    val += y_kernel[i * d_inner + k] * self.out_proj_weight[j * d_inner + k]
                output[i * Self.d_model + j] = val
        
        y_kernel.free()
