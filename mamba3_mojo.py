# Copyright (c) 2026, Modular Inc. All rights reserved.

import torch
import numpy as np
from pathlib import Path

from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

class Mamba3Mojo:
    """Python wrapper for Mamba-3 Mojo kernels using MAX."""
    
    def __init__(self, d_model: int, d_state: int, nheads: int, headdim: int, group_size: int):
        self.d_model = d_model
        self.d_state = d_state
        self.nheads = nheads
        self.headdim = headdim
        self.group_size = group_size
        
        # Path to mojo kernels
        self.mojo_kernels = Path.cwd() / "modular/max/kernels/src/state_space/mamba3"
        
        # Default to CPU for initial Mac benchmarking, Accelerator for NVIDIA
        self.device = Accelerator() if accelerator_count() > 0 else CPU()
        self.session = InferenceSession(devices=[self.device])
        self.dev_ref = DeviceRef.from_device(self.device)
        
        self.fwd_model = None
        
    def _compile_fwd(self, u_shape, delta_shape, A_shape, B_shape, C_shape, D_shape, z_shape, angles_shape, trap_shape):
        dtype = DType.float32
        with Graph(
            "mamba3_fwd_graph",
            input_types=[
                TensorType(dtype, u_shape, self.dev_ref),
                TensorType(dtype, delta_shape, self.dev_ref),
                TensorType(dtype, A_shape, self.dev_ref),
                TensorType(dtype, B_shape, self.dev_ref),
                TensorType(dtype, C_shape, self.dev_ref),
                TensorType(dtype, D_shape, self.dev_ref),
                TensorType(dtype, z_shape, self.dev_ref),
                TensorType(dtype, angles_shape, self.dev_ref),
                TensorType(dtype, trap_shape, self.dev_ref),
            ],
            custom_extensions=[self.mojo_kernels],
        ) as graph:
            results = ops.custom(
                name="mamba3_fwd",
                device=self.dev_ref,
                values=graph.inputs,
                out_types=[TensorType(dtype, u_shape, self.dev_ref)],
            )
            graph.output(*results)
        
        return self.session.load(graph)

    def forward(self, u, delta, A, B, C, D, z, angles, trap):
        if self.fwd_model is None:
            self.fwd_model = self._compile_fwd(
                u.shape, delta.shape, A.shape, B.shape, C.shape, D.shape, z.shape, angles.shape, trap.shape
            )
        
        # Convert torch to MAX Buffers
        args = [
            Buffer.from_numpy(t.detach().cpu().float().numpy()).to(self.device)
            for t in [u, delta, A, B, C, D, z, angles, trap]
        ]
        
        out_mojo = self.fwd_model.execute(*args)
        return torch.from_numpy(out_mojo[0].to_numpy()).to(u.device).to(u.dtype)
