import numpy as np
import torch
from torch.utils.data import Dataset
from src.entities.config_entity import DatasetConfig
from src.entities.artifact_entity import DatasetArtifact

class ShallowSeekDataset(Dataset):
    def __init__(self, config: DatasetConfig):
        self.encoded_path = config.encoded_path
        self.block_size = config.block_size
        self.split = config.split
        self.val_split = config.val_split

        data = np.memmap(str(self.encoded_path), dtype=np.uint16, mode="r")
        split_idx = int(len(data) * (1 - self.val_split))

        if self.split == "train":
            self.data = data[:split_idx]
        else:
            self.data = data[split_idx:]

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        chunk = torch.from_numpy(self.data[idx : idx + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]

    def get_artifact(self) -> DatasetArtifact:
        return DatasetArtifact(num_samples=len(self), split=self.split, block_size=self.block_size)


if __name__ == "__main__":
    train_ds = ShallowSeekDataset(DatasetConfig(split="train"))
    val_ds   = ShallowSeekDataset(DatasetConfig(split="val"))

    print(f"Train samples : {len(train_ds):,}")
    print(f"Val   samples : {len(val_ds):,}")

    x, y = val_ds[0]
    print(f"\n=== Sample 0 ===")
    print(f"  input  ids : {x.tolist()}")
    print(f"  target ids : {y.tolist()}")
    print(f"\n  (target is input shifted by 1 — each token predicts the next)")
