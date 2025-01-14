import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self, filters, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, filters[0], 3, 1)
        self.conv2 = nn.Conv2d(filters[0], filters[1], 3, 1)
        self.conv3 = nn.Conv2d(filters[1], filters[2], 3, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(filters[2], 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)



if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")