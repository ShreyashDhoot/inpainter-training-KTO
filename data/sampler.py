import random
import math
from torch.utils.data import Sampler


class StratifiedBatchSampler(Sampler):
    """
    Guarantees at least `min_neg` unsafe (label=0) samples per batch.
    Remaining slots filled with safe (label=1) samples.
    Iterates through the full positive set once per epoch.
    """

    def __init__(self, labels, batch_size, min_neg=8, shuffle=True):
        self.pos_idx = [i for i, l in enumerate(labels) if int(l) == 1]
        self.neg_idx = [i for i, l in enumerate(labels) if int(l) == 0]
        self.batch_size = batch_size
        self.min_neg = min_neg
        self.shuffle = shuffle

        assert len(self.neg_idx) >= min_neg, (
            f"Dataset has only {len(self.neg_idx)} unsafe samples "
            f"but min_neg={min_neg} required"
        )
        self.pos_per_batch = batch_size - min_neg
        # number of full batches we can make
        self._len = math.floor(len(self.pos_idx) / self.pos_per_batch)

        print(
            f"[StratifiedBatchSampler] "
            f"pos={len(self.pos_idx)}  neg={len(self.neg_idx)}  "
            f"batch_size={batch_size}  min_neg={min_neg}  "
            f"batches/epoch≈{self._len}"
        )

    def __iter__(self):
        pos = self.pos_idx.copy()
        neg = self.neg_idx.copy()
        if self.shuffle:
            random.shuffle(pos)
            random.shuffle(neg)

        neg_cycle = neg * (                         # repeat neg list so we never run out
            (self._len * self.min_neg) // len(neg) + 2
        )
        random.shuffle(neg_cycle)

        neg_ptr = 0
        for i in range(0, self._len * self.pos_per_batch, self.pos_per_batch):
            pos_batch = pos[i : i + self.pos_per_batch]
            neg_batch = neg_cycle[neg_ptr : neg_ptr + self.min_neg]
            neg_ptr += self.min_neg
            indices = pos_batch + neg_batch
            if self.shuffle:
                random.shuffle(indices)     # shuffle within the batch
            yield indices

    def __len__(self):
        return self._len