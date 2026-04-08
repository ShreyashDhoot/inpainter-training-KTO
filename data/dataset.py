import os
import glob
import bisect
from pathlib import Path

import torch
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


class LatentInpaintDataset(torch.utils.data.Dataset):
    def __init__(self, repo_id=None, local_dir=None, split="train", cache_dir="./hf_cache"):
        self.repo_id = repo_id
        self.split = split
        self.cache_dir = cache_dir

        if local_dir is None:
            if repo_id is None:
                raise ValueError("Provide either repo_id or local_dir")
            local_dir = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=cache_dir,
                local_dir_use_symlinks=False,
            )

        self.root = os.path.join(local_dir, split) if os.path.isdir(os.path.join(local_dir, split)) else local_dir
        self.files = sorted(glob.glob(os.path.join(self.root, "*.parquet")))
        if not self.files:
            raise FileNotFoundError(f"No parquet files found in {self.root}")

        self.cum_rows = [0]
        self.row_counts = []

        for f in self.files:
            pf = pq.ParquetFile(f)
            n = pf.metadata.num_rows
            self.row_counts.append(n)
            self.cum_rows.append(self.cum_rows[-1] + n)

    def __len__(self):
        return self.cum_rows[-1]

    def _locate(self, idx):
        file_idx = bisect.bisect_right(self.cum_rows, idx) - 1
        local_idx = idx - self.cum_rows[file_idx]
        return file_idx, local_idx

    def _read_row(self, f, local_idx):
        table = pq.read_table(f)
        row = table.slice(local_idx, 1).to_pylist()[0]
        return row

    def __getitem__(self, idx):
        file_idx, local_idx = self._locate(idx)
        row = self._read_row(self.files[file_idx], local_idx)

        z0 = torch.tensor(row["z0"], dtype=torch.float32)
        masked_latent = torch.tensor(row["masked_latent"], dtype=torch.float32)
        mask_latent = torch.tensor(row["mask_latent"], dtype=torch.float32)
        if mask_latent.ndim == 2:
            mask_latent = mask_latent.unsqueeze(0)
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        label = torch.tensor(row["label"], dtype=torch.float32)

        return {
            "z0": z0,
            "masked_latent": masked_latent,
            "mask_latent": mask_latent,
            "input_ids": input_ids,
            "label": label,
        }