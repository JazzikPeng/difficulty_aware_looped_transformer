# A streaming data loader for efficient data loading
import os
import io
import random
from typing import Callable, Iterable, List, Dict, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

class phop_dataset(IterableDataset):
    """
    Lazily reads very large text files with byte-range sharding per worker.
    Each yielded sample is {'ids': List[int]} produced by encode_fn(line).
    """
    def __init__(self,
                 file_paths: List[str],
                 encode_fn: Callable[[str], List[int]],
                 *,
                 shuffle_files: bool = True,
                 encoding: str = "utf-8",
                 errors: str = "ignore"
    ) -> None:
        super().__init__()
        self.encode_fn = encode_fn
        self.shuffle_files = shuffle_files
        self.encoding = encoding
        self.errors = errors
    
    def _iter_file_shard(self, path: str, start: int, end: int) -> Iterable[Dict[str, List[int]]]:
        with open(path, "r", encoding=self.encoding, errors=self.errors, buffering=io.DEFAULT_BUFFER_SIZE) as f:
            f.seek(start)
            if start != 0:
                # Align to next full line
                _ = f.readline()
            pos = f.tell()
            while pos < end:
                line = f.readline()
                if not line:
                    break
                pos = f.tell()
                line = line.rstrip("\n")
                if not line:
                    continue
                ids = self.encode_fn(line)
                if ids:
                    yield {"ids": ids}
            
    
    