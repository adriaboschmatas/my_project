import torch
from torch import nn, optim
from src.my_project.model import MyAwesomeModel

def test_optimizer_step():
    # Initialize model, optimizer, and loss function
    model = MyAwesomeModel([32, 64, 128])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Dummy data for testing
    img = torch.randn(4, 1, 28, 28)  # Batch of 4 images
    target = torch.randint(0, 10, (4,))  # Random target labels for 4 images

    # Forward pass
    preds = model(img)

    # Compute initial weights
    initial_weights = [param.clone() for param in model.parameters()]

    # Compute loss and perform backward pass
    loss = loss_fn(preds, target)
    loss.backward()

    # Optimizer step
    optimizer.step()

    # Validate that weights have updated
    for param, initial_param in zip(model.parameters(), initial_weights):
        assert not torch.equal(param, initial_param), "Parameters did not update after optimizer.step()"

