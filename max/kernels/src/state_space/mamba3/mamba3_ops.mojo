# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Mamba-3 operation registrations."""

from std.math import ceildiv
import compiler_internal as compiler
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from std.utils.index import IndexList
from std.algorithm import parallelize
from std.memory import UnsafePointer

from state_space.mamba3.mamba3 import (
    mamba3_fwd_gpu,
    mamba3_step_gpu,
)

def softplus_fn(val: Float32) -> Float32:
    if val > 20.0: return val
    return std.math.log(1.0 + std.math.exp(val))

def sigmoid_fn(val: Float32) -> Float32:
    return 1.0 / (1.0 + std.math.exp(-val))

@compiler.register("mamba3_fwd")
struct Mamba3Fwd:
    """Mamba-3 forward pass operation."""

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        u: InputTensor[dtype=dtype, rank=3, ...],
        delta: InputTensor[dtype=dtype, rank=3, ...],
        A: InputTensor[dtype=dtype, rank=3, ...],
        B: InputTensor[dtype=dtype, rank=4, ...],
        C: InputTensor[dtype=dtype, rank=4, ...],
        D: InputTensor[dtype=dtype, rank=1, ...],
        z: InputTensor[dtype=dtype, rank=3, ...],
        angles: InputTensor[dtype=dtype, rank=3, ...],
        trap: InputTensor[dtype=dtype, rank=3, ...],
        ctx: DeviceContextPtr,
    ) capturing raises:
        var batch = u.dim_size(0)
        var seqlen = u.dim_size(1)
        var dim = u.dim_size(2)
        var dstate = A.dim_size(2)
        var nheads = delta.dim_size(2)
        var headdim = dim // nheads
        var n_groups = B.dim_size(2)

        var output_lt = output.to_layout_tensor()
        var u_lt = u.to_layout_tensor()
        var delta_lt = delta.to_layout_tensor()
        var A_lt = A.to_layout_tensor()
        var B_lt = B.to_layout_tensor()
        var C_lt = C.to_layout_tensor()
        var D_lt = D.to_layout_tensor()
        var z_lt = z.to_layout_tensor()
        var angles_lt = angles.to_layout_tensor()
        var trap_lt = trap.to_layout_tensor()

        var output_strides = output.strides()
        var u_strides = u.strides()
        var delta_strides = delta.strides()
        var A_strides = A.strides()
        var B_strides = B.strides()
        var C_strides = C.strides()
        var D_strides = D.strides()
        var z_strides = z.strides()
        var angles_strides = angles.strides()
        var trap_strides = trap.strides()

        if is_cpu[target]():
            @parameter
            def p_fwd(b: Int):
                for head_idx in range(nheads):
                    for p in range(headdim):
                        var d = head_idx * headdim + p
                        var group_id = head_idx // (nheads // n_groups)
                        
                        var s_real = SIMD[DType.float32, 128](0.0)
                        var s_imag = SIMD[DType.float32, 128](0.0)
                        var A_vals = SIMD[DType.float32, 128](0.0)
                        
                        for n in range(dstate):
                            A_vals[n] = Scalar[dtype](A_lt.ptr[b * A_strides[0] + head_idx * A_strides[1] + n * A_strides[2]]).cast[DType.float32]()

                        var prev_u = Float32(0.0)

                        for t in range(seqlen):
                            var u_val = Scalar[dtype](u_lt.ptr[b * u_strides[0] + t * u_strides[1] + d * u_strides[2]]).cast[DType.float32]()
                            var dt_val = Scalar[dtype](delta_lt.ptr[b * delta_strides[0] + t * delta_strides[1] + head_idx * delta_strides[2]]).cast[DType.float32]()
                            var tr_val = Scalar[dtype](trap_lt.ptr[b * trap_strides[0] + t * trap_strides[1] + head_idx * trap_strides[2]]).cast[DType.float32]()
                            
                            var S = angles_lt.dim(1)
                            
                            var exp_arg = A_vals * dt_val
                            for n in range(dstate):
                                if exp_arg[n] > 0.0: exp_arg[n] = 0.0
                            
                            var alpha = std.math.exp(exp_arg)
                            var beta = (1.0 - tr_val) * dt_val * alpha
                            var gamma = tr_val * dt_val
                            
                            var output_val = Float32(0.0)

                            for n in range(dstate):
                                var bv = Scalar[dtype](B_lt.ptr[b * B_strides[0] + t * B_strides[1] + group_id * B_strides[2] + n * B_strides[3]]).cast[DType.float32]()
                                var cv = Scalar[dtype](C_lt.ptr[b * C_strides[0] + t * C_strides[1] + group_id * C_strides[2] + n * C_strides[3]]).cast[DType.float32]()
                                
                                var base = bv * prev_u * beta[n] + bv * u_val * gamma
                                
                                var rc = Float32(1.0)
                                var rs = Float32(0.0)
                                var in_r = base
                                var in_i = Float32(0.0)
                                
                                if n < S:
                                    var ang = Scalar[dtype](angles_lt.ptr[b * angles_strides[0] + t * angles_strides[1] + n * angles_strides[2]]).cast[DType.float32]()
                                    rc = std.math.cos(ang)
                                    rs = std.math.sin(ang)
                                    in_r = base * rc
                                    in_i = base * rs
                                
                                s_real[n] = s_real[n] * alpha[n] + in_r
                                s_imag[n] = s_imag[n] * alpha[n] + in_i
                                
                                if n < S:
                                    output_val += (s_real[n] * rc + s_imag[n] * rs) * cv
                                else:
                                    output_val += s_real[n] * cv

                            if D_lt.dim(0) > 0:
                                output_val += Scalar[dtype](D_lt.ptr[head_idx]).cast[DType.float32]() * u_val
                            
                            var z_val = Scalar[dtype](z_lt.ptr[b * z_strides[0] + t * z_strides[1] + d * z_strides[2]]).cast[DType.float32]()
                            output_val *= (z_val * sigmoid_fn(z_val))
                            output_lt.ptr[b * output_strides[0] + t * output_strides[1] + d * output_strides[2]] = output_val.cast[dtype]()
                            prev_u = u_val

            parallelize[p_fwd](batch)
        else:
            pass

@compiler.register("mamba3_step")
struct Mamba3Step:
    """Mamba-3 single-step update operation."""

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        state_real: OutputTensor[dtype=dtype, rank=3, ...],
        state_imag: OutputTensor[dtype=dtype, rank=3, ...],
        u: InputTensor[dtype=dtype, rank=2, ...],
        delta: InputTensor[dtype=dtype, rank=2, ...],
        A: InputTensor[dtype=dtype, rank=3, ...],
        B: InputTensor[dtype=dtype, rank=3, ...],
        C: InputTensor[dtype=dtype, rank=3, ...],
        D: InputTensor[dtype=dtype, rank=1, ...],
        z: InputTensor[dtype=dtype, rank=2, ...],
        angles: InputTensor[dtype=dtype, rank=2, ...],
        trap: InputTensor[dtype=dtype, rank=2, ...],
        output: OutputTensor[dtype=dtype, rank=2, ...],
        ctx: DeviceContextPtr,
    ) capturing raises:
        pass
