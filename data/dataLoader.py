import torch
from torch.utils.data import Dataset, DataLoader
import os
from phop.phop_generation import phop_collate_batch 
from phop.constant import *

# Constants for p-hop training data generation

class LargeTextDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        self.line_offsets = []
        self.files = []

        # Precompute line offsets for random access
        for path in file_paths:
            offsets = []
            with open(path, "rb") as f:
                offset = 0
                for line in f:
                    offsets.append(offset)
                    offset += len(line)
            self.line_offsets.append((path, offsets))

        # Flatten indexing (file_index, line_index)
        self.index_map = []
        for f_idx, (_, offs) in enumerate(self.line_offsets):
            for l_idx in range(len(offs)):
                self.index_map.append((f_idx, l_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        f_idx, l_idx = self.index_map[idx]
        file_path, offsets = self.line_offsets[f_idx]

        with open(file_path, "rb") as f:
            f.seek(offsets[l_idx])
            line = f.readline().decode("utf-8").strip()

        # Example: parse space-separated integers
        sample = list(map(int, line.split()))
        sample = torch.tensor(sample, dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)

        return sample


# Usage
file_paths = [
    "data/phop/p_hop_sequences_16_256_4.txt",
    "data/phop/p_hop_sequences_32_512_8.txt",
    "data/phop/p_hop_sequences_64_1024_16.txt",
]

dataset = LargeTextDataset(file_paths)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=phop_collate_batch)

for batch in loader:
    print(f"Batch shape: {batch.shape}")  # (batch_size, 2, BLOCK_SIZE)
    print(f"X shape: {batch[:, 0, :].shape}")  # Input sequences
    print(f"Y shape: {batch[:, 1, :].shape}")  # Target sequences
    break