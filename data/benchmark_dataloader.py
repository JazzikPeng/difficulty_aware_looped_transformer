import time
import torch
from torch.utils.data import DataLoader
from dataLoader import LargeTextDataset  # assume dataset code is saved in dataset.py
from phop.phop_generation import phop_collate_batch


def benchmark_dataloader(file_paths, batch_size=32, num_workers_list=[0, 2, 4, 8], num_batches=200):
    """
    Benchmark dataloader performance.
    - file_paths: list of txt files
    - batch_size: how many samples per batch
    - num_workers_list: test different worker counts
    - num_batches: how many batches to iterate for benchmark
    """

    for num_workers in num_workers_list:
        print(f"\n=== Benchmark with num_workers={num_workers} ===")

        dataset = LargeTextDataset(file_paths)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=phop_collate_batch,
            pin_memory=True
        )

        start = time.time()
        n_samples = 0

        for i, batch in enumerate(dataloader):
            n_samples += batch.size(0)
            if i + 1 >= num_batches:  # limit for fair timing
                break

        elapsed = time.time() - start
        print(f"Processed {n_samples} samples in {elapsed:.2f}s")
        print(f"  → {n_samples/elapsed:.2f} samples/sec")
        print(f"  → {(i+1)/elapsed:.2f} batches/sec")


if __name__ == "__main__":
    file_paths = [
        "data/phop/p_hop_sequences_16_256_4.txt",
        "data/phop/p_hop_sequences_32_512_8.txt",
        "data/phop/p_hop_sequences_64_1024_16.txt",
    ]
    benchmark_dataloader(file_paths, batch_size=64, num_workers_list=[0, 2, 4, 8], num_batches=500)