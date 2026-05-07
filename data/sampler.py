import random
import math
import torch
from torch.utils.data import Sampler

class StratifiedBatchSampler(Sampler):
    """
    STRATEGY FOR IMBALANCED KTO:
    1. Majority Class (Unsafe, label=0): Iterated through exactly once per epoch.
    2. Minority Class (Safe, label=1): Guaranteed a minimum count per batch, 
       cycled/repeated to fill slots.
    """

    def __init__(self, labels, batch_size, min_safe=8, shuffle=True):
        # Identify indices by actual meaning
        self.safe_idx = [i for i, l in enumerate(labels) if int(l) == 0]   # Minority (20k)
        self.unsafe_idx = [i for i, l in enumerate(labels) if int(l) == 1] # Majority (61k)
        
        self.batch_size = batch_size
        self.min_safe = min_safe
        self.shuffle = shuffle

        assert len(self.safe_idx) >= min_safe, (
            f"Dataset has only {len(self.safe_idx)} safe samples "
            f"but min_safe={min_safe} required"
        )

        # How many Unsafe samples can we fit per batch?
        self.unsafe_per_batch = batch_size - min_safe
        
        # We define the epoch length by the MAJORITY set (Unsafe)
        # to ensure every unsafe sample is seen once.
        self._len = math.ceil(len(self.unsafe_idx) / self.unsafe_per_batch)

        print(
            f"[StratifiedBatchSampler] "
            f"Safe(Min)={len(self.safe_idx)}  Unsafe(Maj)={len(self.unsafe_idx)}  "
            f"BatchSize={batch_size}  MinSafe/Batch={min_safe}  "
            f"TotalBatches/Epoch={self._len}"
        )

    def __iter__(self):
        safe = self.safe_idx.copy()
        unsafe = self.unsafe_idx.copy()
        
        if self.shuffle:
            random.shuffle(safe)
            random.shuffle(unsafe)

        # Repeat the Safe (minority) list so we never run out while 
        # iterating through the long Unsafe list.
        safe_cycle = safe * ( (self._len * self.min_safe) // len(safe) + 2 )
        if self.shuffle:
            random.shuffle(safe_cycle)

        safe_ptr = 0
        for i in range(0, len(unsafe), self.unsafe_per_batch):
            # 1. Grab the Unsafe samples (Majority)
            unsafe_batch = unsafe[i : i + self.unsafe_per_batch]
            
            # 2. Grab the Safe samples (Guaranteed Minority count)
            safe_batch = safe_cycle[safe_ptr : safe_ptr + self.min_safe]
            safe_ptr += self.min_safe
            
            # 3. Combine and shuffle within the batch
            indices = unsafe_batch + safe_batch
            
            # Handle potential partial last batch
            if len(indices) < self.batch_size:
                # If last batch is small, top it off with more random safe samples
                extra = self.batch_size - len(indices)
                indices += random.sample(self.safe_idx, extra)

            if self.shuffle:
                random.shuffle(indices)
                
            yield indices

    def __len__(self):
        return self._len