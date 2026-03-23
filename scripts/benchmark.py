import time
import torch
import subprocess
import numpy as np
import os
import argparse
from mamba_ssm.modules.mamba3 import Mamba3

def benchmark_python(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    model = Mamba3(
        d_model=args.dim,
        d_state=args.d_state,
        headdim=args.headdim,
        is_mimo=args.is_mimo,
        dtype=dtype,
    ).to(device)
    model.eval()

    x = torch.randn(args.batch, args.seqlen, args.dim, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(args.iters):
        with torch.no_grad():
            _ = model(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / args.iters * 1000
    print(f"Python Reference Average Time: {avg_time:.2f} ms")
    return avg_time

def benchmark_mojo(args):
    # Run the mojo benchmark script
    try:
        # Use pixi run to ensure all environment variables (MODULAR_HOME, etc.) are correctly set
        cmd = ["pixi", "run", "mojo", "run", "-I", ".", "-I", "modular/max/kernels/src", "scripts/benchmark.mojo"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Mojo Error: {result.stderr}")
            return 0.0
        
        output = result.stdout
        print(output)
        for line in output.split("\n"):
            if "BENCHMARK_RESULT:" in line:
                return float(line.split(":")[1].strip())
        return 0.0
    except Exception as e:
        print(f"Error running Mojo benchmark: {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--d_state", type=int, default=128)
    parser.add_argument("--is_mimo", action="store_true")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    print(f"Starting benchmark with config: {args}")
    py_time = benchmark_python(args)
    mojo_time = benchmark_mojo(args)
    
    print("\nBenchmark Summary:")
    print(f"{'Implementation':<20} | {'Avg Time (ms)':<15}")
    print("-" * 40)
    print(f"{'Python (Ref)':<20} | {py_time:<15.4f}")
    print(f"{'Mojo (Port)':<20} | {mojo_time:<15.4f}")

if __name__ == "__main__":
    main()
