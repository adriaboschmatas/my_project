from src.my_project.model import MyAwesomeModel
import torch
import pytest

def test_model():
    model = MyAwesomeModel([32, 64, 128])
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10), "Output shape is not (1, 10)"


def test_error_on_wrong_shape():
    model = MyAwesomeModel([32, 64, 128])
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    with pytest.raises(ValueError, match='Expected each sample to have shape \\[1, 28, 28\\]'):
        model(torch.randn(1,1,28,29))

@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = MyAwesomeModel([32, 64, 128])
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)