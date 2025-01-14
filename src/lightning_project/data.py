import pytorch_lightning as pl
from torch.utils.data import DataLoader

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))) # Otherwise it can't read the src directory
from src.my_project.data import corrupt_mnist

class CorruptMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # Ensure data files exist and are preprocessed
        pass

    def setup(self, stage: str | None = None):
        # Load data and split into train/test
        self.train_set, self.test_set = corrupt_mnist()

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )

    def val_dataloader(self):
        # Optionally use part of the train set for validation
        return DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False
        )
