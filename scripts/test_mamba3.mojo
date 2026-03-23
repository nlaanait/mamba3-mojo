from mojo_mamba.mamba3 import Mamba3
from layout import Layout, LayoutTensor
from std.memory import UnsafePointer, alloc
from std.python import Python, PythonObject

def npy_to_ptr(npy_arr: PythonObject) raises -> UnsafePointer[Float32, MutExternalOrigin]:
    var shape = npy_arr.shape
    var size: Int = 1
    for i in range(len(shape)):
        size *= Int(py=shape[i])
    
    var data = alloc[Float32](size)
    var flat = npy_arr.flatten()
    for i in range(size):
        data[i] = Float32(py=flat[i])
    
    return data

def main() raises:
    print("Mamba3 Numerical Correctness Test")
    
    var np = Python.import_module("numpy")
    
    comptime batch = 1
    comptime seqlen = 16
    comptime dim = 64
    comptime d_state = 16
    comptime headdim = 8
    comptime dtype = DType.float32
    
    # Load parameters
    var dt_bias_ptr = npy_to_ptr(np.load("data/param_dt_bias.npy"))
    var D_ptr = npy_to_ptr(np.load("data/param_D.npy"))
    var out_proj_ptr = npy_to_ptr(np.load("data/param_out_proj_weight.npy"))
    
    # Load split inputs
    var z_ptr = npy_to_ptr(np.load("data/split_z.npy"))
    var x_ptr = npy_to_ptr(np.load("data/split_x.npy"))
    var B_ptr = npy_to_ptr(np.load("data/split_B.npy"))
    var C_ptr = npy_to_ptr(np.load("data/split_C.npy"))
    var split_dt_ptr = npy_to_ptr(np.load("data/split_dt.npy"))
    var split_A_ptr = npy_to_ptr(np.load("data/split_A.npy"))
    var split_trap_ptr = npy_to_ptr(np.load("data/split_trap.npy"))
    var split_angles_ptr = npy_to_ptr(np.load("data/split_angles.npy"))
    
    # Reference output (final y)
    var y_ref_npy = np.load("data/output_y.npy")
    var y_ref_flat = y_ref_npy.flatten()
    
    print("Loaded all reference data and parameters")
    
    # Initialize Mojo model
    var model = Mamba3[dtype, dim, d_state, headdim]()
    model.dt_bias = dt_bias_ptr
    model.D = D_ptr
    model.out_proj_weight = out_proj_ptr
    
    # Output buffer
    var out_ptr = alloc[Float32](batch * seqlen * dim)
    
    print("Running Mojo Mamba3 forward pass...")
    model.forward(
        batch, seqlen, z_ptr, x_ptr, B_ptr, C_ptr,
        split_dt_ptr, split_A_ptr, split_trap_ptr, split_angles_ptr,
        out_ptr
    )
    
    # Compare results
    print("\nFull Results Comparison:")
    var max_err: Float32 = 0.0
    var total_elements = batch * seqlen * dim
    for i in range(total_elements): 
        var mojo_val = out_ptr[i]
        var ref_val = Float32(py=y_ref_flat[i])
        var diff = std.math.abs(mojo_val - ref_val)
        if diff > max_err: max_err = diff
    
    print("Max error across all", total_elements, "elements:", max_err)
    
    if max_err < 1e-5:
        print("Numerical check PASSED!")
    else:
        print("Numerical check FAILED!")
    
    print("\nVerification complete.")
    
    # Clean up
    dt_bias_ptr.free()
    D_ptr.free()
    out_proj_ptr.free()
    z_ptr.free()
    x_ptr.free()
    B_ptr.free()
    C_ptr.free()
    split_dt_ptr.free()
    split_A_ptr.free()
    split_trap_ptr.free()
    split_angles_ptr.free()
    out_ptr.free()
