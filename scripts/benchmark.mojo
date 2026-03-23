from mojo_mamba.mamba3 import Mamba3
from std.memory import UnsafePointer, alloc
import std.time

def main() raises:
    print("Mojo Mamba3 Benchmark (Real Logic)")
    
    # Configuration
    comptime batch = 1
    comptime seqlen = 128
    comptime dim = 768
    comptime d_state = 128
    comptime headdim = 64
    comptime dtype = DType.float32
    
    var model = Mamba3[dtype, dim, d_state, headdim]()
    
    # Initialize pointers with dummy data
    var d_inner = dim * 2
    var nheads = d_inner // headdim
    
    var z = alloc[Float32](batch * seqlen * d_inner)
    var x = alloc[Float32](batch * seqlen * d_inner)
    var B = alloc[Float32](batch * seqlen * d_state)
    var C = alloc[Float32](batch * seqlen * d_state)
    var dd_dt = alloc[Float32](batch * seqlen * nheads)
    var dd_A = alloc[Float32](batch * seqlen * nheads)
    var trap = alloc[Float32](batch * seqlen * nheads)
    var angles = alloc[Float32](batch * seqlen * 4) # Simplified
    var out = alloc[Float32](batch * seqlen * dim)
    
    # Model parameters
    model.dt_bias = alloc[Float32](nheads)
    model.D = alloc[Float32](nheads)
    model.out_proj_weight = alloc[Float32](dim * d_inner)

    # Warmup
    print("Warming up...")
    for _ in range(5):
        model.forward(batch, seqlen, z, x, B, C, dd_dt, dd_A, trap, angles, out)
    
    # Benchmark
    print("Running benchmark (10 iterations)...")
    var start = std.time.perf_counter_ns()
    
    for _ in range(10):
        model.forward(batch, seqlen, z, x, B, C, dd_dt, dd_A, trap, angles, out)
    
    var end = std.time.perf_counter_ns()
    var duration = Float64(end - start) / 1e6 / 10.0
    print("Mojo Mamba3 Avg duration: ", duration, "ms")
    print("BENCHMARK_RESULT:", duration)

    # Clean up
    z.free()
    x.free()
    B.free()
    C.free()
    dd_dt.free()
    dd_A.free()
    trap.free()
    angles.free()
    out.free()
    model.dt_bias.free()
    model.D.free()
    model.out_proj_weight.free()
