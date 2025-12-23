""" Training
    Main script for training the model
"""


import os
import pickle
import shutil
import tarfile

from igcl.igcl_train import train as igcl_train
from itcl.itcl_train import train as itcl_train
from subfunc.showdata import *


# Parameters ==================================================
# =============================================================

# Data generation ---------------------------------------------
num_layer = 2  # number of layers of mixing-MLP (changed from 3 to 2)
num_comp = 10  # number of components (dimension) (changed from 20 to 10)
num_data = 2**15  # number of data points (changed from 2**18 to 2**15)
num_basis = 64  # number of frequencies of fourier bases
modulate_range = [-2, 2]
modulate_range2 = [-2, 2]
ar_order = 1
random_seed = 0  # random seed

# select learning framework (igcl or itcl)
# net_model = 'igcl'  # learn by IIA-GCL
net_model = 'itcl'


# MLP ---------------------------------------------------------
list_hidden_nodes = [4 * num_comp] * (num_layer - 1) + [num_comp]
list_hidden_nodes_z = None
# list of the number of nodes of each hidden layer of feature-MLP
# [layer1, layer2, ..., layer(num_layer)]


# Training ----------------------------------------------------
initial_learning_rate = 0.01  # initial learning rate (default:0.1)
momentum = 0.9  # momentum parameter of SGD
max_steps = int(3e6)  # number of iterations (mini-batches)
decay_steps = int(1e6)  # decay steps (tf.train.exponential_decay)
decay_factor = 0.1  # decay factor (tf.train.exponential_decay)
batch_size = 512  # mini-batch size
moving_average_decay = 0.999  # moving average decay of variables to be saved
checkpoint_steps = int(1e7)  # interval to save checkpoint
summary_steps = int(1e4)  # interval to save summary
apply_pca = True  # apply PCA for preprocessing or not
weight_decay = 1e-5  # weight decay

# For quick test ----------------------------------------------
max_steps = 80_000
summary_steps = 2_000
checkpoint_steps = 20_000
batch_size = 256




# Other -------------------------------------------------------
# # Note: save folder must be under ./storage
train_dir_base = './storage'

import time
exp_id = time.strftime("%Y%m%d_%H%M%S")
train_dir = os.path.join(train_dir_base, f"model_{exp_id}")


saveparmpath = os.path.join(train_dir, 'parm.pkl')  # file name to save parameters


# =============================================================
# =============================================================

# Prepare save folder -----------------------------------------
if os.path.normpath(train_dir).startswith(os.path.normpath(train_dir_base)):
    if os.path.exists(train_dir):
        print('delete savefolder: %s...' % train_dir)
        shutil.rmtree(train_dir)
    print('make savefolder: %s...' % train_dir)
    os.makedirs(train_dir)
else:
    raise RuntimeError('savefolder looks wrong')


# ----------------------------------------------------------
# Load real financial data
# ----------------------------------------------------------
x = np.load("x_finance.npy")      # shape (T, d)
y = np.load("u_finance.npy")      # shape (T,)
num_segment = len(np.unique(y))

x_te, y_te = None, None

num_data = x.shape[0]
num_comp = x.shape[1]

# ----------------------------------------------------------
# Ensure AR windows are compatible with batch_size
# ----------------------------------------------------------
T = x.shape[0]

# Number of usable AR samples
T_ar = T - ar_order

# Truncate so that T_ar is divisible by batch_size
T_ar_eff = (T_ar // batch_size) * batch_size

# Final truncation
x = x[:T_ar_eff + ar_order]
# labels must align with AR windows
y = y[: T_ar_eff + ar_order]
    
num_data = T_ar_eff


# Train model  ------------------------------------------------
if net_model == 'igcl':
    igcl_train(x,
               y,
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
               checkpoint_steps=checkpoint_steps,
               moving_average_decay=moving_average_decay,
               summary_steps=summary_steps,
               random_seed=random_seed)
elif net_model == 'itcl':
    itcl_train(x,
               y,
               list_hidden_nodes=list_hidden_nodes,
               list_hidden_nodes_z=list_hidden_nodes_z,
               num_segment=num_segment,
               initial_learning_rate=initial_learning_rate,
               momentum=momentum,
               max_steps=max_steps,
               decay_steps=decay_steps,
               decay_factor=decay_factor,
               batch_size=batch_size,
               train_dir=train_dir,
               ar_order=ar_order,
               weight_decay=weight_decay,
               checkpoint_steps=checkpoint_steps,
               moving_average_decay=moving_average_decay,
               summary_steps=summary_steps,
               random_seed=random_seed)

# Save parameters necessary for evaluation --------------------
model_parm = {'random_seed': random_seed,
              'num_comp': num_comp,
              'num_data': num_data,
              'ar_order': ar_order,
              'num_basis': num_basis,
              'modulate_range': modulate_range,
              'modulate_range2': modulate_range2,
              'num_layer': num_layer,
              'list_hidden_nodes': list_hidden_nodes,
              'list_hidden_nodes_z': list_hidden_nodes_z,
              'moving_average_decay': moving_average_decay,
              'num_segment': num_segment if 'num_segment' in locals() else None,
              'net_model': net_model}

print('Save parameters...')
with open(saveparmpath, 'wb') as f:
    pickle.dump(model_parm, f, pickle.HIGHEST_PROTOCOL)

# Save as tarfile
tarname = train_dir + ".tar.gz"
archive = tarfile.open(tarname, mode="w:gz")
archive.add(train_dir, arcname="./")
archive.close()

print('done.')
