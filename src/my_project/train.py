import os
import sys
import matplotlib.pyplot as plt
import torch
from data import corrupt_mnist

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
import logging
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(config_path="../../conf", config_name="conf.yaml")
def train(cfg: DictConfig) -> None:
    """Train a model on MNIST."""

    # Log configuration
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Save the final configuration to the output directory
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # log loaded hyperparameters (for debugging)
    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Hydra's Output directory: {hydra_output_dir}")

    # Set seed
    torch.manual_seed(cfg.training.seed)

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from src.my_project.model import MyAwesomeModel
    print(MyAwesomeModel)

    # Instantiate model, optimizer, and training params
    model = instantiate(cfg.model).to(DEVICE)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs

    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                log.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    log.info("Training complete")
    # save weights
    output_path = os.path.join(hydra_output_dir, "trained_model.pt")
    torch.save(model, output_path)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    train()
