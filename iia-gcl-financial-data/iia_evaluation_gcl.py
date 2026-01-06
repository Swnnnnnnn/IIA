"""
IIA-GCL Evaluation / Extraction on Real Financial Data

This script:
- Loads a trained IIA-GCL model
- Loads real financial data
- Extracts latent innovations h_t
- Saves them for further analysis
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn


# ==============================================================================
# Maxout Nonlinearity (same as training)
# ==============================================================================

class Maxout(nn.Module):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
    
    def forward(self, x):
        shape = x.shape
        x = x.view(shape[0], shape[1] // self.pool_size, self.pool_size)
        x, _ = torch.max(x, dim=2)
        return x


# ==============================================================================
# IIA-GCL Model (same as training)
# ==============================================================================

class NetGaussScaleMean(nn.Module):
    def __init__(self, h_sizes, num_dim, num_data, num_basis, 
                 ar_order=1, h_sizes_z=None, pool_size=2):
        super().__init__()
        
        if h_sizes_z is None:
            h_sizes_z = h_sizes.copy()
        
        self.num_dim = num_dim
        self.num_data = num_data
        self.num_basis = num_basis
        self.ar_order = ar_order
        
        # Network h
        h_sizes_aug = [num_dim * (ar_order + 1)] + h_sizes
        layers_h = []
        for k in range(1, len(h_sizes_aug) - 1):
            layers_h.append(nn.Linear(h_sizes_aug[k-1], h_sizes_aug[k] * pool_size))
        layers_h.append(nn.Linear(h_sizes_aug[-2], h_sizes_aug[-1]))
        self.layers_h = nn.ModuleList(layers_h)
        
        # Network hz
        h_sizes_z_aug = [num_dim * ar_order] + h_sizes_z
        layers_hz = []
        for k in range(1, len(h_sizes_z_aug) - 1):
            layers_hz.append(nn.Linear(h_sizes_z_aug[k-1], h_sizes_z_aug[k] * pool_size))
        layers_hz.append(nn.Linear(h_sizes_z_aug[-2], h_sizes_z_aug[-1]))
        self.layers_hz = nn.ModuleList(layers_hz)
        
        self.maxout = Maxout(pool_size)
        
        total_h_dim = h_sizes[-1] + h_sizes_z[-1]
        self.wr1 = nn.Linear(2 * num_basis, total_h_dim, bias=True)
        self.wr2 = nn.Linear(2 * num_basis, total_h_dim, bias=True)
        
        self.a = nn.Linear(1, 1, bias=False)
        self.b = nn.Linear(1, 1, bias=False)
        self.c = nn.Linear(1, 1, bias=False)
        self.d = nn.Linear(1, 1, bias=False)
        self.e = nn.Linear(1, 1, bias=False)
        self.f = nn.Linear(1, 1, bias=False)
        self.g = nn.Linear(1, 1, bias=False)
        self.m = nn.Linear(1, 1, bias=False)
    
    def forward(self, x, t):
        batch_size = x.size(0)
        xz = x[:, 1:, :]
        
        t_shfl = t[torch.randperm(batch_size, device=t.device)]
        t_cat = torch.cat([t, t_shfl], dim=0)
        
        h = x.reshape(batch_size, -1)
        for k, layer in enumerate(self.layers_h):
            h = layer(h)
            if k != len(self.layers_h) - 1:
                h = self.maxout(h)
        h = torch.cat([h, h], dim=0)
        
        hz = xz.reshape(batch_size, -1)
        for k, layer in enumerate(self.layers_hz):
            hz = layer(hz)
            if k != len(self.layers_hz) - 1:
                hz = self.maxout(hz)
        hz = torch.cat([hz, hz], dim=0)
        
        k_range = torch.arange(1, self.num_basis + 1, device=x.device, dtype=torch.float32)
        fn_basis = 2 * np.pi * k_range.view(1, -1) * t_cat.float().view(-1, 1) / self.num_data
        t_basis = torch.cat([torch.sin(fn_basis), torch.cos(fn_basis)], dim=1)
        
        t_mod_log1 = self.wr1(t_basis)
        t_mod1 = torch.exp(t_mod_log1)
        t_mod2 = self.wr2(t_basis)
        
        h_dim = h.size(1)
        h_sq = h ** 2
        hz_sq = hz ** 2
        
        h_mod = (h_sq * t_mod1[:, :h_dim] * self.a.weight + 
                 h * t_mod1[:, :h_dim] * t_mod2[:, :h_dim] * self.b.weight)
        hz_mod = (hz_sq * t_mod1[:, h_dim:] * self.c.weight + 
                  hz * t_mod1[:, h_dim:] * t_mod2[:, h_dim:] * self.d.weight)
        
        Q = torch.mean(h_mod, dim=1) + torch.mean(hz_mod, dim=1)
        Qbar = torch.mean(h_sq * self.e.weight, dim=1) + torch.mean(hz_sq * self.f.weight, dim=1)
        Z = torch.mean(t_mod_log1 * self.g.weight, dim=1)
        
        logits = -Q + Qbar + Z + self.m.weight.squeeze()
        
        return logits, h, hz, t_mod1, t_mod2
    
    def get_features(self, x):
        """Extract features h (innovations) without contrastive setup."""
        batch_size = x.size(0)
        h = x.reshape(batch_size, -1)
        for k, layer in enumerate(self.layers_h):
            h = layer(h)
            if k != len(self.layers_h) - 1:
                h = self.maxout(h)
        return h
    
    def get_features_hz(self, x):
        """Extract features hz from past observations only."""
        batch_size = x.size(0)
        xz = x[:, 1:, :]
        hz = xz.reshape(batch_size, -1)
        for k, layer in enumerate(self.layers_hz):
            hz = layer(hz)
            if k != len(self.layers_hz) - 1:
                hz = self.maxout(hz)
        return hz


# ==============================================================================
# Main Script
# ==============================================================================

if __name__ == "__main__":
    
    # =========================================================================
    # Paths
    # =========================================================================
    
    eval_dir_base = "./storage"
    
    # Find the most recent igcl model directory
    import glob
    model_dirs = sorted(glob.glob(os.path.join(eval_dir_base, "model_igcl_*")))
    
    if len(model_dirs) == 0:
        raise FileNotFoundError("No IIA-GCL model found in storage/. Run iia_training_gcl.py first.")
    
    eval_dir = model_dirs[-1]  # Most recent
    print(f"Using model directory: {eval_dir}")
    
    parm_path = os.path.join(eval_dir, "parm.pkl")
    model_path = os.path.join(eval_dir, "model.pt")
    
    # =========================================================================
    # Load training parameters
    # =========================================================================
    
    with open(parm_path, "rb") as f:
        model_parm = pickle.load(f)
    
    num_comp = model_parm["num_comp"]
    num_data = model_parm["num_data"]
    ar_order = model_parm["ar_order"]
    num_basis = model_parm["num_basis"]
    list_hidden_nodes = model_parm["list_hidden_nodes"]
    list_hidden_nodes_z = model_parm["list_hidden_nodes_z"]
    
    print(f"Model parameters:")
    print(f"  num_comp: {num_comp}")
    print(f"  num_data: {num_data}")
    print(f"  ar_order: {ar_order}")
    print(f"  num_basis: {num_basis}")
    print(f"  hidden_nodes: {list_hidden_nodes}")
    
    # =========================================================================
    # Load real financial data
    # =========================================================================
    
    x = np.load("x_finance.npy")  # shape (T, d)
    y = np.arange(x.shape[0])     # Time indices for GCL
    
    print(f"\nLoaded data:")
    print(f"  x shape: {x.shape}")
    print(f"  y shape: {y.shape}")
    
    # =========================================================================
    # Build AR windows
    # =========================================================================
    
    T = x.shape[0]
    t_idx = np.arange(T - ar_order) + ar_order
    t_idx_ar = t_idx.reshape(-1, 1) + np.arange(0, -ar_order - 1, -1).reshape(1, -1)
    
    x_ar = x[t_idx_ar.flatten(), :].reshape(-1, ar_order + 1, x.shape[1])
    y_ar = y[t_idx]  # Aligned time indices
    
    print(f"\nAR input shape: {x_ar.shape}")
    print(f"Time indices shape: {y_ar.shape}")
    
    # =========================================================================
    # Load model
    # =========================================================================
    
    device = "cpu"
    
    if list_hidden_nodes_z is None:
        list_hidden_nodes_z = list_hidden_nodes.copy()
    
    model = NetGaussScaleMean(
        h_sizes=list_hidden_nodes,
        h_sizes_z=list_hidden_nodes_z,
        num_dim=num_comp,
        num_data=num_data,
        num_basis=num_basis,
        ar_order=ar_order
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Use EMA parameters (recommended)
    model.load_state_dict(checkpoint["ema_state_dict"])
    model.to(device)
    model.eval()
    
    print("\nModel loaded (using EMA weights).")
    
    # =========================================================================
    # Forward pass - extract innovations
    # =========================================================================
    
    x_torch = torch.from_numpy(x_ar.astype(np.float32)).to(device)
    
    with torch.no_grad():
        h_val = model.get_features(x_torch)
        hz_val = model.get_features_hz(x_torch)
    
    h_val = h_val.cpu().numpy()
    hz_val = hz_val.cpu().numpy()
    
    print(f"\nExtracted innovations:")
    print(f"  h shape: {h_val.shape}")
    print(f"  hz shape: {hz_val.shape}")
    
    # =========================================================================
    # Save results
    # =========================================================================
    
    np.save(os.path.join(eval_dir, "innovations_hat.npy"), h_val)
    np.save(os.path.join(eval_dir, "innovations_hz.npy"), hz_val)
    np.save(os.path.join(eval_dir, "time_indices.npy"), y_ar)
    
    print(f"\nSaved to {eval_dir}:")
    print("  - innovations_hat.npy (h features)")
    print("  - innovations_hz.npy (hz features)")
    print("  - time_indices.npy")
    
    print("\nDone!")
