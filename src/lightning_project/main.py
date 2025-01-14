import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import wandb
wandb.login()  # Ensure you are logged in to Weights & Biases


from lightning import MyAwesomeModel  # Import your model
from data import CorruptMNISTDataModule

if __name__ == "__main__":
    # Create data module and model
    data_module = CorruptMNISTDataModule(batch_size=64)
    model = MyAwesomeModel()

    early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=3, verbose=True, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )

    # Initialize Lightning Trainer
    trainer = Trainer(
        max_epochs=10,
        limit_train_batches=0.2,  # Limit the amount of training data to 20%
        callbacks=[early_stopping_callback, checkpoint_callback],
        logger=pl.loggers.WandbLogger(project="dtu_mlops"),
        accelerator="gpu",  # Use GPU acceleration
        devices=1,  # Use 1 GPU
        precision="16-mixed",  # Enable mixed-precision training
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    #trainer.test(model, data_module)
