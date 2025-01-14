import torch
from src.my_project.data import corrupt_mnist
import pytest
import os.path
@pytest.mark.skipif(not os.path.exists("data"), reason="Data files not found")
def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000, "len(train) is not 30.000"
    assert len(test) == 5000, "len(train) is not 5.000"
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), "x.shape is not (1, 28, 28)"
            assert y in range(10), "y is not in range (10)"
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()