# Copyright (c) 2026, Modular Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from mamba3_mojo import Mamba3Mojo

class Mamba3Clean(nn.Module):
    """Standalone Mamba-3 implementation for benchmarking."""
    
    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        rope_fraction=0.5,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        A_floor=1e-4,
        use_mojo=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.A_floor = A_floor
        self.use_mojo = use_mojo
        self.d_inner = int(self.expand * self.d_model)
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = ngroups
        
        # RoPE config
        self.split_tensor_size = int(d_state * rope_fraction)
        self.num_rope_angles = self.split_tensor_size // 2

        # Linear projection
        d_in_proj = 2 * self.d_inner + 2 * self.d_state * self.num_bc_heads + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        # Initialize A and dt_bias
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.nheads + 1, dtype=torch.float32, device=device).repeat(self.d_state, 1).T))
        
        _dt = torch.exp(torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        _dt = torch.clamp(_dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(_dt + torch.log(-torch.expm1(-_dt)))
        
        self.D = nn.Parameter(torch.ones(self.nheads, **factory_kwargs))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

        if self.use_mojo:
            self.mojo = Mamba3Mojo(
                d_model=d_model, d_state=d_state, nheads=self.nheads, 
                headdim=headdim, group_size=headdim
            )

    def forward(self, u):
        batch, seqlen, _ = u.shape
        zxBCdtAtrap = self.in_proj(u)
        
        # Split projections
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [self.d_inner, self.d_inner, self.d_state * self.num_bc_heads, 
             self.d_state * self.num_bc_heads, self.nheads, self.nheads, 
             self.nheads, self.num_rope_angles],
            dim=-1
        )
        
        # Discretization parameters
        _A = -F.softplus(dd_A)
        _A = torch.clamp(_A, max=-self.A_floor)
        dt = F.softplus(dd_dt + self.dt_bias)
        trap_val = torch.sigmoid(trap)
        
        # RoPE angles
        angle_delta = torch.tanh(angles.unsqueeze(-2)) * dt.unsqueeze(-1) * torch.pi
        angle_cumsum = torch.cumsum(angle_delta, dim=1) # (B, L, H, S)

        if self.use_mojo:
            # Prepare tensors for Mojo (B, L, D) layouts
            # A_mojo: (B, H, N)
            A_mojo = repeat(_A[:, 0, :], "b h -> b h n", n=self.d_state)
            
            y = self.mojo.forward(
                u=x,
                delta=dt,
                A=A_mojo,
                B=B.reshape(batch, seqlen, self.num_bc_heads, self.d_state),
                C=C.reshape(batch, seqlen, self.num_bc_heads, self.d_state),
                D=self.D,
                z=z,
                angles=angle_cumsum[:, :, 0, :], # Pass cumsum'd angles
                trap=trap_val
            )
            return self.out_proj(y)
        
        # Reference PyTorch logic (Recurrent Mamba-3)
        x_heads = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        z_heads = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        B_heads = B.reshape(batch, seqlen, self.num_bc_heads, self.d_state)
        C_heads = C.reshape(batch, seqlen, self.num_bc_heads, self.d_state)
        
        out = torch.zeros_like(x_heads)
        state_real = torch.zeros(batch, self.nheads, self.headdim, self.d_state, device=u.device)
        state_imag = torch.zeros(batch, self.nheads, self.headdim, self.d_state, device=u.device)
        
        prev_x = torch.zeros_like(x_heads[:, 0, :, :])
        
        for t in range(seqlen):
            dt_t = dt[:, t, :].unsqueeze(-1).unsqueeze(-1)
            A_t = _A[:, t, :].unsqueeze(-1).unsqueeze(-1)
            tr_t = trap_val[:, t, :].unsqueeze(-1).unsqueeze(-1)
            
            alpha = torch.exp(A_t * dt_t)
            beta = (1.0 - tr_t) * dt_t * alpha
            gamma = tr_t * dt_t
            
            theta = angle_cumsum[:, t, :, :] # (B, H, S)
            rc = torch.cos(theta).unsqueeze(-2)
            rs = torch.sin(theta).unsqueeze(-2)
            
            bt = B_heads[:, t, 0, :].unsqueeze(1).unsqueeze(1)
            ct = C_heads[:, t, 0, :].unsqueeze(1).unsqueeze(1)
            xt = x_heads[:, t, :, :].unsqueeze(-1)
            pxt = prev_x.unsqueeze(-1)
            
            # Complex input addition: (curr * gamma + prev * beta) * e^itheta
            base = bt * pxt * beta + bt * xt * gamma
            
            S = theta.shape[-1]
            in_r = base.clone()
            in_i = torch.zeros_like(in_r)
            in_r[:, :, :, :S] = base[:, :, :, :S] * rc
            in_i[:, :, :, :S] = base[:, :, :, :S] * rs
            
            state_real = state_real * alpha + in_r
            state_imag = state_imag * alpha + in_i
            
            y_t_c = (state_real[:, :, :, :S] * rc + state_imag[:, :, :, :S] * rs) * ct[:, :, :, :S]
            y_t_r = state_real[:, :, :, S:] * ct[:, :, :, S:]
            y_t = y_t_c.sum(dim=-1) + y_t_r.sum(dim=-1)
            
            y_t = y_t + self.D.unsqueeze(0).unsqueeze(-1) * x_heads[:, t, :, :]
            y_t = y_t * F.silu(z_heads[:, t, :, :])
            out[:, t, :, :] = y_t
            prev_x = x_heads[:, t, :, :]
            
        out = rearrange(out, "b l h p -> b l (h p)")
        return self.out_proj(out)
