import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import os
from .phop.phop_generation import phop_collate_batch 
from .phop.constant import *
import random

class FixedRatioBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, total_iters, file_ratios, drop_last=False):
        """
        dataset: LargeTextDataset
        batch_size: int
        total_iters: total number of batches to output
        file_ratios: list of percentages summing to 1.0, e.g. [1/3, 1/3, 1/3]
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.total_iters = total_iters
        self.file_ratios = file_ratios
        self.drop_last = drop_last

        # Group indices by file
        self.file_to_indices = {}
        for idx, (f_idx, _) in enumerate(dataset.index_map):
            self.file_to_indices.setdefault(f_idx, []).append(idx)

        # Shuffle indices inside each file pool
        for f_idx in self.file_to_indices:
            random.shuffle(self.file_to_indices[f_idx])

        # Compute batch quotas per file
        self.file_batch_quota = [
            int(round(r * total_iters)) for r in file_ratios
        ]

    def __iter__(self):
        # Iterator yields batches from files according to quota
        file_iters = {f_idx: iter(self.file_to_indices[f_idx]) for f_idx in self.file_to_indices}
        
        for f_idx, quota in enumerate(self.file_batch_quota):
            for _ in range(quota):
                batch = []
                for _ in range(self.batch_size):
                    try:
                        batch.append(next(file_iters[f_idx]))
                    except StopIteration:
                        # restart if we run out of samples in that file
                        random.shuffle(self.file_to_indices[f_idx])
                        file_iters[f_idx] = iter(self.file_to_indices[f_idx])
                        batch.append(next(file_iters[f_idx]))
                yield batch

    def __len__(self):
        return self.total_iters
    
    
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


if __name__ == "__main__":
    from collections import Counter
    # Usage example
    file_paths = [
        "data/phop/p_hop_sequences_16_256_4.txt",
        "data/phop/p_hop_sequences_32_512_8.txt",
        "data/phop/p_hop_sequences_64_1024_16.txt",
    ]

    # dataset = LargeTextDataset(file_paths)
    # loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=phop_collate_batch)

    # for batch in loader:
    #     print(f"Batch shape: {batch.shape}")  # (batch_size, 2, BLOCK_SIZE)
    #     print(f"X shape: {batch[:, 0, :].shape}")  # Input sequences
    #     print(f"Y shape: {batch[:, 1, :].shape}")  # Target sequences
    #     break
    
    dataset = LargeTextDataset(file_paths)
    batch_size = 16
    total_iters = 6000
    ratios = [1/3, 1/3, 1/3]  # N, M, K percentages

    sampler = FixedRatioBatchSampler(dataset, batch_size, total_iters, ratios)

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=phop_collate_batch
    )

    # Write list of p values to variable
    p_values = []
    for step, batch in enumerate(loader):
        # Find the number of non -1 values in the Y tensor and subtract 2 for the 3 output start tokens and first 
        # initial position of hop.
        print(f"Step {step}: {batch.shape} | p = {torch.where(batch[0][1]!=-1)[0].shape[0] - 2}")
        p_values.append(torch.where(batch[0][1]!=-1)[0].shape[0] - 2)
    
    
    # Compute distinct count of each p value
    p_values_count = Counter(p_values)
    print(p_values_count)
    