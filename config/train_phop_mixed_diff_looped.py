# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
init_from = 'scratch_loop' # 'scratch' or 'resume' or 'gpt2*' or 'scratch_loop'
out_dir = 'out-phop-16-looped'
model_output_name = 'ckpt_2_0_6_random.pt' # ckpt_<base_block_size>_<loop_start>_<num_loops>.pt
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100 # don't print too too often

# data
train_file_paths = [
    "data/phop/p_hop_sequences_16_256_4_4m.txt",
    "data/phop/p_hop_sequences_32_512_8_4m.txt",
    "data/phop/p_hop_sequences_64_1024_16_4m.txt",
]
test_file_paths = [
    "data/phop/p_hop_sequences_100_256_4_test.txt",
    "data/phop/p_hop_sequences_100_512_8_test.txt",
    "data/phop/p_hop_sequences_100_1024_16_test.txt",
]

# system
device = 'cuda'
compile = False # do not torch compile the model
# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'phop-mix'
wandb_run_name = 'phop-mix-looped_2_0_6_random' # <base_block_size>_<loop_start>_<num_loops>.pt

dataset = 'phop'
gradient_accumulation_steps = 16
batch_size = 16
block_size = 1152 # context of up to 278 previous characters

# baby GPT model :)
n_layer = 2
n_head = 6
n_embd = 768
dropout = 0.0

# Loop config
num_loops = 6
loop_start = 0
loop_func = 'z=f(x+z)'

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 60000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
warmup_iters = 100 # not super necessary potentially

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# define dataloader here
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, GPTLooped
from torch.utils.data import Dataset, DataLoader
# Add project root to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent  # Adjust as needed
sys.path.append(str(project_root))
from data.dataLoader import LargeTextDataset
from data.phop.phop_generation import phop_collate_batch

# Create a dataset and a dataloader for phop mixed difficulty learning
train_dataset = LargeTextDataset(train_file_paths)
test_dataset   = LargeTextDataset(test_file_paths)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    collate_fn=phop_collate_batch,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    drop_last=True,
)
test_loader   = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    collate_fn=phop_collate_batch,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    drop_last=True,
)