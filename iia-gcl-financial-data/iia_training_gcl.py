"""
IIA-GCL Training on Real Financial Data

This script trains an IIA-GCL model on real financial data.
Unlike IIA-TCL which uses segment indices, IIA-GCL uses the exact
time index as auxiliary variable, exploiting continuous temporal
modulation of innovation statistics.
"""

import os
import pickle
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime


# ==============================================================================
# Maxout Nonlinearity
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
# IIA-GCL Model
# ==============================================================================

class NetGaussScaleMean(nn.Module):
    """
    IIA-GCL network for Gaussian innovations with scale-mean modulation.
    
    Two feature extractors:
    - h: processes (x_t, x_{t-1}, ..., x_{t-p}) -> estimates innovations
    - hz: processes (x_{t-1}, ..., x_{t-p}) -> captures past dependency (φ term)
    """
    
    def __init__(self, h_sizes, num_dim, num_data, num_basis, 
                 ar_order=1, h_sizes_z=None, pool_size=2):
        super().__init__()
        
        if h_sizes_z is None:
            h_sizes_z = h_sizes.copy()
        
        self.num_dim = num_dim
        self.num_data = num_data
        self.num_basis = num_basis
        self.ar_order = ar_order
        
        # Network h: input is (x_t, x_{t-1}, ..., x_{t-p})
        h_sizes_aug = [num_dim * (ar_order + 1)] + h_sizes
        layers_h = []
        for k in range(1, len(h_sizes_aug) - 1):
            layers_h.append(nn.Linear(h_sizes_aug[k-1], h_sizes_aug[k] * pool_size))
        layers_h.append(nn.Linear(h_sizes_aug[-2], h_sizes_aug[-1]))
        self.layers_h = nn.ModuleList(layers_h)
        
        # Network hz: input is (x_{t-1}, ..., x_{t-p})
        h_sizes_z_aug = [num_dim * ar_order] + h_sizes_z
        layers_hz = []
        for k in range(1, len(h_sizes_z_aug) - 1):
            layers_hz.append(nn.Linear(h_sizes_z_aug[k-1], h_sizes_z_aug[k] * pool_size))
        layers_hz.append(nn.Linear(h_sizes_z_aug[-2], h_sizes_z_aug[-1]))
        self.layers_hz = nn.ModuleList(layers_hz)
        
        self.maxout = Maxout(pool_size)
        
        # Temporal modulation layers (Fourier basis -> modulation)
        total_h_dim = h_sizes[-1] + h_sizes_z[-1]
        self.wr1 = nn.Linear(2 * num_basis, total_h_dim, bias=True)
        self.wr2 = nn.Linear(2 * num_basis, total_h_dim, bias=True)
        
        # Weighting parameters (constrained positive)
        self.a = nn.Linear(1, 1, bias=False)
        self.b = nn.Linear(1, 1, bias=False)
        self.c = nn.Linear(1, 1, bias=False)
        self.d = nn.Linear(1, 1, bias=False)
        self.e = nn.Linear(1, 1, bias=False)
        self.f = nn.Linear(1, 1, bias=False)
        self.g = nn.Linear(1, 1, bias=False)
        self.m = nn.Linear(1, 1, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.layers_h:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in self.layers_hz:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.wr1.weight)
        nn.init.xavier_uniform_(self.wr2.weight)
        nn.init.zeros_(self.wr1.bias)
        nn.init.zeros_(self.wr2.bias)
        nn.init.constant_(self.a.weight, 1)
        nn.init.constant_(self.b.weight, 1)
        nn.init.constant_(self.c.weight, 1)
        nn.init.constant_(self.d.weight, 1)
        nn.init.constant_(self.e.weight, 1)
        nn.init.constant_(self.f.weight, 1)
        nn.init.constant_(self.g.weight, 1)
        nn.init.constant_(self.m.weight, 0)
    
    def apply_positivity_constraints(self):
        self.a.weight.data.clamp_(min=0)
        self.b.weight.data.clamp_(min=0)
        self.c.weight.data.clamp_(min=0)
        self.d.weight.data.clamp_(min=0)
        self.e.weight.data.clamp_(min=0)
        self.f.weight.data.clamp_(min=0)
        self.g.weight.data.clamp_(min=0)
    
    def forward(self, x, t):
        """
        Forward pass.
        
        Args:
            x: input [batch, ar_order+1, num_dim]
            t: time index [batch]
        
        Returns:
            logits, h, hz, t_mod1, t_mod2
        """
        batch_size = x.size(0)
        xz = x[:, 1:, :]  # (x_{t-1}, ..., x_{t-p})
        
        # Shuffle t for contrastive learning (create fake samples)
        t_shfl = t[torch.randperm(batch_size, device=t.device)]
        t_cat = torch.cat([t, t_shfl], dim=0)
        
        # Network h
        h = x.reshape(batch_size, -1)
        for k, layer in enumerate(self.layers_h):
            h = layer(h)
            if k != len(self.layers_h) - 1:
                h = self.maxout(h)
        h = torch.cat([h, h], dim=0)
        
        # Network hz
        hz = xz.reshape(batch_size, -1)
        for k, layer in enumerate(self.layers_hz):
            hz = layer(hz)
            if k != len(self.layers_hz) - 1:
                hz = self.maxout(hz)
        hz = torch.cat([hz, hz], dim=0)
        
        # Fourier basis for temporal modulation
        k_range = torch.arange(1, self.num_basis + 1, device=x.device, dtype=torch.float32)
        fn_basis = 2 * np.pi * k_range.view(1, -1) * t_cat.float().view(-1, 1) / self.num_data
        t_basis = torch.cat([torch.sin(fn_basis), torch.cos(fn_basis)], dim=1)
        
        # Modulation
        t_mod_log1 = self.wr1(t_basis)
        t_mod1 = torch.exp(t_mod_log1)
        t_mod2 = self.wr2(t_basis)
        
        h_dim = h.size(1)
        
        # Compute Q terms
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
        """Extract features h (estimated innovations) without contrastive setup."""
        batch_size = x.size(0)
        h = x.reshape(batch_size, -1)
        for k, layer in enumerate(self.layers_h):
            h = layer(h)
            if k != len(self.layers_h) - 1:
                h = self.maxout(h)
        return h


# ==============================================================================
# Training Function
# ==============================================================================

def train_igcl(data, label, list_hidden_nodes, list_hidden_nodes_z,
               num_data, num_basis, initial_learning_rate, momentum,
               max_steps, decay_steps, decay_factor, batch_size,
               train_dir, ar_order=1, weight_decay=1e-5,
               moving_average_decay=0.999, checkpoint_steps=10000,
               summary_steps=500, random_seed=None):
    """
    Train IIA-GCL model.
    
    Args:
        data: observations [num_data, num_dim]
        label: time indices [num_data]
        list_hidden_nodes: hidden layer sizes for h network
        list_hidden_nodes_z: hidden layer sizes for hz network
        num_data: number of data points
        num_basis: number of Fourier basis functions
        initial_learning_rate: initial LR
        momentum: SGD momentum
        max_steps: total training steps
        decay_steps: LR decay interval
        decay_factor: LR decay factor
        batch_size: batch size
        train_dir: directory to save model
        ar_order: AR model order
        weight_decay: L2 regularization
        moving_average_decay: EMA decay
        checkpoint_steps: checkpoint save interval
        summary_steps: logging interval
        random_seed: random seed
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    if list_hidden_nodes_z is None:
        list_hidden_nodes_z = list_hidden_nodes.copy()
    
    model = NetGaussScaleMean(
        h_sizes=list_hidden_nodes,
        h_sizes_z=list_hidden_nodes_z,
        num_dim=data.shape[1],
        num_data=num_data,
        num_basis=num_basis,
        ar_order=ar_order
    )
    model = model.to(device)
    model.train()
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, 
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_factor)
    
    # Initialize EMA
    state_dict_ema = {k: v.clone() for k, v in model.state_dict().items()}
    
    print(f"Training IIA-GCL for {max_steps} steps...")
    print(f"Data shape: {data.shape}")
    
    # Training loop
    for step in range(max_steps):
        # Sample batch
        t_idx = np.random.permutation(data.shape[0] - ar_order)[:batch_size] + ar_order
        t_idx_ar = t_idx.reshape(-1, 1) + np.arange(0, -ar_order - 1, -1).reshape(1, -1)
        x_batch = data[t_idx_ar.flatten(), :].reshape(batch_size, ar_order + 1, -1)
        t_batch = label[t_idx]
        
        x_torch = torch.from_numpy(x_batch.astype(np.float32)).to(device)
        t_torch = torch.from_numpy(t_batch).long().to(device)
        y_torch = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).to(device)
        
        optimizer.zero_grad()
        
        logits, h, hz, _, _ = model(x_torch, t_torch)
        loss = criterion(logits, y_torch)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Apply positivity constraints
        model.apply_positivity_constraints()
        
        # Update EMA
        with torch.no_grad():
            for k, v in model.state_dict().items():
                state_dict_ema[k] = moving_average_decay * state_dict_ema[k] + \
                                    (1 - moving_average_decay) * v
        
        # Compute accuracy
        with torch.no_grad():
            predicted = (logits > 0.5).float()
            accuracy = (predicted == y_torch).float().mean().item()
        
        # Logging
        if step % summary_steps == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"{datetime.now().strftime('%H:%M:%S')}: step {step}, "
                  f"lr={lr:.6f}, loss={loss.item():.4f}, acc={accuracy*100:.1f}%")
        
        # Checkpoint
        if step % checkpoint_steps == 0 and step > 0:
            checkpoint_path = os.path.join(train_dir, 'model.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': state_dict_ema,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, checkpoint_path)
            print(f"  Checkpoint saved at step {step}")
    
    # Save final model
    save_path = os.path.join(train_dir, 'model.pt')
    torch.save({
        'step': max_steps,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': state_dict_ema,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, save_path)
    print(f"Model saved to {save_path}")
    
    return model, state_dict_ema


# ==============================================================================
# Main Script
# ==============================================================================

if __name__ == "__main__":
    
    # =========================================================================
    # Parameters
    # =========================================================================
    
    # Model architecture
    num_layer = 2
    num_basis = 64
    ar_order = 1
    random_seed = 0
    
    # Training parameters
    initial_learning_rate = 0.01
    momentum = 0.9
    max_steps = 100_000
    decay_steps = 30_000
    decay_factor = 0.1
    batch_size = 128  # Smaller batch for smaller dataset
    moving_average_decay = 0.999
    weight_decay = 1e-5
    
    checkpoint_steps = 20_000
    summary_steps = 1_000
    
    # Directories
    train_dir_base = './storage'
    exp_id = time.strftime("%Y%m%d_%H%M%S")
    train_dir = os.path.join(train_dir_base, f"model_igcl_{exp_id}")
    
    # =========================================================================
    # Prepare save folder
    # =========================================================================
    
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir)
    print(f"Save folder: {train_dir}")
    
    # =========================================================================
    # Load real financial data
    # =========================================================================
    
    x = np.load("x_finance.npy")  # shape (T, d)
    
    # For IIA-GCL, we use the exact time index as auxiliary variable
    # (not segment indices like in IIA-TCL)
    T = x.shape[0]
    y = np.arange(T)  # Time indices: 0, 1, 2, ..., T-1
    
    num_data = T
    num_comp = x.shape[1]
    
    print(f"Loaded data: x shape = {x.shape}")
    print(f"Using time indices as auxiliary variable: y shape = {y.shape}")
    
    # =========================================================================
    # Build hidden layer sizes
    # =========================================================================
    
    list_hidden_nodes = [4 * num_comp] * (num_layer - 1) + [num_comp]
    list_hidden_nodes_z = None
    
    print(f"Hidden nodes (h): {list_hidden_nodes}")
    
    # =========================================================================
    # Train model
    # =========================================================================
    
    model, ema_state_dict = train_igcl(
        data=x,
        label=y,
        list_hidden_nodes=list_hidden_nodes,
        list_hidden_nodes_z=list_hidden_nodes_z,
        num_data=num_data,
        num_basis=num_basis,
        initial_learning_rate=initial_learning_rate,
        momentum=momentum,
        max_steps=max_steps,
        decay_steps=decay_steps,
        decay_factor=decay_factor,
        batch_size=batch_size,
        train_dir=train_dir,
        ar_order=ar_order,
        weight_decay=weight_decay,
        moving_average_decay=moving_average_decay,
        checkpoint_steps=checkpoint_steps,
        summary_steps=summary_steps,
        random_seed=random_seed
    )
    
    # =========================================================================
    # Save parameters
    # =========================================================================
    
    model_parm = {
        'random_seed': random_seed,
        'num_comp': num_comp,
        'num_data': num_data,
        'ar_order': ar_order,
        'num_basis': num_basis,
        'num_layer': num_layer,
        'list_hidden_nodes': list_hidden_nodes,
        'list_hidden_nodes_z': list_hidden_nodes_z,
        'moving_average_decay': moving_average_decay,
        'net_model': 'igcl'
    }
    
    parm_path = os.path.join(train_dir, 'parm.pkl')
    with open(parm_path, 'wb') as f:
        pickle.dump(model_parm, f, pickle.HIGHEST_PROTOCOL)
    
    print(f"Parameters saved to {parm_path}")
    print("Training complete!")
