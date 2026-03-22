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

from std.gpu import block_dim, block_idx, thread_idx
from std.os.atomic import Atomic, Consistency
from layout import Layout, LayoutTensor
from std.utils.index import IndexList
from std.memory import UnsafePointer
from std.algorithm import sync_parallelize
import std.math
from std.math import ceildiv, exp, exp2, rsqrt, sin, cos
from state_space.causal_conv1d import silu

# ===----------------------------------------------------------------------=== #
# Constants and Type Aliases
# ===----------------------------------------------------------------------=== #

# Stride types for passing tensor strides to kernels
comptime Strides1D = IndexList[1]
comptime Strides2D = IndexList[2]
comptime Strides3D = IndexList[3]
comptime Strides4D = IndexList[4]


# ===----------------------------------------------------------------------=== #
# Mamba-3 Forward Kernels
# ===----------------------------------------------------------------------=== #

def mamba3_fwd_gpu[
    kernel_dtype: DType,
    DSTATE: Int,
    output_layout: Layout,
    u_layout: Layout,
    delta_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    angles_layout: Layout,
    trap_layout: Layout,
](
    total_threads: Int,
    batch: Int,
    dim: Int,
    seqlen: Int,
    group_size: Int,
    output: LayoutTensor[kernel_dtype, output_layout, MutAnyOrigin],
    u: LayoutTensor[kernel_dtype, u_layout, MutAnyOrigin],
    delta: LayoutTensor[kernel_dtype, delta_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    angles: LayoutTensor[kernel_dtype, angles_layout, MutAnyOrigin],
    trap: LayoutTensor[kernel_dtype, trap_layout, MutAnyOrigin],
    output_strides: Strides3D,
    u_strides: Strides3D,
    delta_strides: Strides3D,
    A_strides: Strides3D,
    B_strides: Strides4D,
    C_strides: Strides4D,
    D_strides: Strides1D,
    z_strides: Strides3D,
    angles_strides: Strides3D,
    trap_strides: Strides3D,
):
    """Ultra-minimal kernel to test Metal compilation."""
    var thread_id = Int(block_dim.x * block_idx.x + thread_idx.x)
    if thread_id >= total_threads:
        return

    var b = thread_id // (dim * DSTATE)
    var d = (thread_id % (dim * DSTATE)) // DSTATE
    
    if b < batch and d < dim:
        var val = Scalar[kernel_dtype](u.ptr[b * u_strides[0] + d * u_strides[1]]).cast[DType.float32]()
        output.ptr[b * output_strides[0] + d * output_strides[1]] = val.cast[kernel_dtype]()

def mamba3_step_gpu[
    kernel_dtype: DType,
    DSTATE: Int,
    state_real_layout: Layout,
    state_imag_layout: Layout,
    u_layout: Layout,
    delta_layout: Layout,
    A_layout: Layout,
    B_layout: Layout,
    C_layout: Layout,
    D_layout: Layout,
    z_layout: Layout,
    angles_layout: Layout,
    trap_layout: Layout,
    output_layout: Layout,
](
    total_threads: Int,
    batch: Int,
    dim: Int,
    group_size: Int,
    state_real: LayoutTensor[kernel_dtype, state_real_layout, MutAnyOrigin],
    state_imag: LayoutTensor[kernel_dtype, state_imag_layout, MutAnyOrigin],
    u: LayoutTensor[kernel_dtype, u_layout, MutAnyOrigin],
    delta: LayoutTensor[kernel_dtype, delta_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    D: LayoutTensor[kernel_dtype, D_layout, MutAnyOrigin],
    z: LayoutTensor[kernel_dtype, z_layout, MutAnyOrigin],
    angles: LayoutTensor[kernel_dtype, angles_layout, MutAnyOrigin],
    trap: LayoutTensor[kernel_dtype, trap_layout, MutAnyOrigin],
    output: LayoutTensor[kernel_dtype, output_layout, MutAnyOrigin],
):
    """Ultra-minimal kernel to test Metal compilation."""
    var thread_id = Int(block_dim.x * block_idx.x + thread_idx.x)
    if thread_id >= total_threads:
        return

    var b = thread_id // (dim * DSTATE)
    var d = (thread_id % (dim * DSTATE)) // DSTATE
    
    if b < batch and d < dim:
        var val = Scalar[kernel_dtype](u.ptr[b * dim + d]).cast[DType.float32]()
        output.ptr[b * dim + d] = val.cast[kernel_dtype]()
