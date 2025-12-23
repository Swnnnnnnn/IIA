"""
Evaluation / Extraction script for IIA-TCL on real data

This script:
- loads a trained IIA-TCL model
- loads real financial data
- reconstructs AR windows
- extracts latent innovations h_t
- saves them for further analysis
"""

import os
import pickle
import numpy as np
import torch

from itcl import itcl


# ============================================================
# Paths
# ============================================================

eval_dir_base = "./storage"

eval_dir = os.path.join(eval_dir_base, "model_data_yf")

parmpath = os.path.join(eval_dir, "parm.pkl")
modelpath = os.path.join(eval_dir, "model.pt")


# ============================================================
# Load training parameters
# ============================================================

with open(parmpath, "rb") as f:
    model_parm = pickle.load(f)

num_comp = model_parm["num_comp"]
ar_order = model_parm["ar_order"]
list_hidden_nodes = model_parm["list_hidden_nodes"]
list_hidden_nodes_z = model_parm["list_hidden_nodes_z"]
num_segment = model_parm["num_segment"]


# ============================================================
# Load real financial data
# ============================================================

x = np.load("x_finance.npy")      # shape (T, d)
y = np.load("u_finance.npy")      # shape (T,)

print("Loaded data:")
print("x shape:", x.shape)
print("y shape:", y.shape)


# ============================================================
# Truncate data to match AR + batch constraints (same as training)
# ============================================================

# batch_size used during training
batch_size = 256

T = x.shape[0]
T_ar = T - ar_order
T_ar_eff = (T_ar // batch_size) * batch_size

# truncate x and align labels
x = x[:T_ar_eff + ar_order]
y = y[ar_order : T_ar_eff + ar_order]

print("After truncation:")
print("x shape:", x.shape)
print("y shape:", y.shape)


# ============================================================
# Build AR windows
# ============================================================

t_idx = np.arange(x.shape[0] - ar_order) + ar_order
t_idx = (
    t_idx.reshape(-1, 1)
    + np.arange(0, -ar_order - 1, -1).reshape(1, -1)
)

x_ar = x[t_idx.reshape(-1), :].reshape(
    -1, ar_order + 1, x.shape[1]
)

print("AR input shape:", x_ar.shape)


# ============================================================
# Load model
# ============================================================

device = "cpu"

model = itcl.Net(
    h_sizes=list_hidden_nodes,
    h_sizes_z=list_hidden_nodes_z,
    ar_order=ar_order,
    num_dim=x_ar.shape[-1],
    num_class=num_segment
)

checkpoint = torch.load(modelpath, map_location=device)

# use EMA parameters (recommended)
model.load_state_dict(checkpoint["ema_state_dict"])

model.to(device)
model.eval()

print("Model loaded.")


# ============================================================
# Forward pass – extract innovations
# ============================================================

x_torch = torch.from_numpy(x_ar.astype(np.float32)).to(device)

with torch.no_grad():
    logits, h, hz = model(x_torch)

h_val = h.cpu().numpy()
hz_val = hz.cpu().numpy()

print("Extracted innovations:")
print("h shape:", h_val.shape)
print("hz shape:", hz_val.shape)


# ============================================================
# Save results
# ============================================================

np.save(os.path.join(eval_dir, "innovations_hat.npy"), h_val)
np.save(os.path.join(eval_dir, "segments.npy"), y)

print("Saved:")
print(" - innovations_hat.npy")
print(" - segments.npy")

print("Done.")
