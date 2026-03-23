import torch
import torch.nn.functional as F
from mamba_ssm.modules.mamba3 import Mamba3
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--d_state", type=int, default=128)
    parser.add_argument("--is_mimo", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    torch.manual_seed(42)
    model = Mamba3(
        d_model=args.dim,
        d_state=args.d_state,
        headdim=args.headdim,
        is_mimo=args.is_mimo,
        dtype=dtype,
    ).to(device)
    model.eval()

    x = torch.randn(args.batch, args.seqlen, args.dim, device=device, dtype=dtype)
    
    with torch.no_grad():
        # Manually compute in_proj splits
        zxBCdtAtrap = model.in_proj(x)
        z, x_inner, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                model.d_inner, model.d_inner, 
                model.d_state * model.num_bc_heads * model.mimo_rank,
                model.d_state * model.num_bc_heads * model.mimo_rank,
                model.nheads, model.nheads, model.nheads, 
                model.num_rope_angles
            ],
            dim=-1)
        
        # Apply RMS Norm on B and C
        B = model.B_norm(B)
        C = model.C_norm(C)

        # Save splits
        np.save("data/split_z.npy", z.cpu().float().numpy())
        np.save("data/split_x.npy", x_inner.cpu().float().numpy())
        np.save("data/split_B.npy", B.cpu().float().numpy())
        np.save("data/split_C.npy", C.cpu().float().numpy())
        np.save("data/split_dt.npy", dd_dt.cpu().float().numpy())
        np.save("data/split_A.npy", dd_A.cpu().float().numpy())
        np.save("data/split_trap.npy", trap.cpu().float().numpy())
        np.save("data/split_angles.npy", angles.cpu().float().numpy())

        # Apply Mamba-3 logic manually or via model call to get intermediate
        # In SISO mode, y is the output of the kernel
        # We can use a hook or just extract it if we are careful
        # For simplicity, let's just use the model's forward and hope we can match it
        # Since we want to compare the kernel output, we need to extract it.
        # For SISO, y before out_proj is the gated output.
        # I'll modify the model call to return it or just compute it here.
        
        # We can't easily get it from the model call without modifying the code.
        # But wait, I ALREADY MODIFIED mamba3.py!
        # I can just save it there or here.
        # Actually, let's just save it in reference_mamba3.py by using a hook or similar.
        
        # I'll modify mamba3.py to save it to a global or something for this test.
        # Better: I'll implement out_proj in Mojo too.
        
        y_final = model(x)
        # Use a hook to capture y before out_proj
        gated_y = None
        def hook(module, input, output):
            nonlocal gated_y
            gated_y = input[0] # out_proj input is the gated y
        
        handle = model.out_proj.register_forward_hook(hook)
        _ = model(x)
        handle.remove()
        
        np.save("data/gated_y.npy", gated_y.cpu().float().numpy())
        np.save("data/output_y.npy", y_final.cpu().float().numpy())
        
        # Actually, let's just save the final y and also the out_proj input if we can
        # For now, let's focus on matching the kernel logic.
        # I'll modify the Mamba3 module temporarily to save its kernel output if needed,
        # but let's try to match the final output by implementing the full path.

    # Save inputs and outputs for Mojo testing
    os.makedirs("data", exist_ok=True)
    
    # We save as float32 for easier loading in Mojo if needed, or keep as is.
    # Mojo supports bfloat16 but let's use float32 for safety during comparison.
    np.save("data/input_x.npy", x.cpu().float().numpy())
    np.save("data/output_y.npy", y_final.cpu().float().numpy())
    
    # Save parameters
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        np.save(f"data/param_{name.replace('.', '_')}.npy", param.cpu().float().numpy())

    print(f"Saved reference data to data/ folder.")

if __name__ == "__main__":
    main()
