from typing import Callable, Type

import pytorch_lightning as pl
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from clearml import Task


from src.constants import *
from src.datasets import SegmentationDataset
from src.unet import UNet2d
from src.metrics import pixel_accuracy


class UNetSegmentationModule(pl.LightningModule):
    def __init__(
        self,
        model_kwargs: dict,
        learning_rate: float,
        loss_fn: Callable,
        optimizer_cls: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.unet_kwargs = model_kwargs

        self.model = UNet2d(**model_kwargs)
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, masks = batch

        logits = self(images)
        loss = self.loss_fn(logits, masks)
        acc = pixel_accuracy(logits, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_pixel_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        acc = pixel_accuracy(logits, masks)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_pixel_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer_params = {"lr": self.learning_rate, **self.optimizer_kwargs}
        return self.optimizer_cls(self.parameters(), **optimizer_params)


if __name__ == "__main__":
    # --- ClearML Task initialization ---
    task = Task.init(project_name="UNet Segmentation", task_name="Test")

    # Example usage
    input_dim = 1
    output_dim = 1
    learning_rate = 1e-3
    unet_kwargs = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dims": [64, 128, 256, 512, 1024],
        "kernel_size": 3,
        "padding_mode": "same",
        "skip_mode": "concat",
        "upsampling_mode": "transpose",
        "dropout": 0,
    }

    loss_fn = nn.BCEWithLogitsLoss()

    model_lt = UNetSegmentationModule(unet_kwargs, learning_rate, loss_fn=loss_fn)

    dataset_fluo = SegmentationDataset(
        root_path=FLUO_PATH,
        img_folder="01",
        target_folder="01_ST/SEG",
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((512, 512))],
        ),
        target_transform=transforms.Resize((512, 512)),
    )
    dataset_fluo_val = SegmentationDataset(
        root_path=FLUO_PATH,
        img_folder="02",
        target_folder="02_ST/SEG",
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((512, 512))],
        ),
        target_transform=transforms.Resize((512, 512)),
    )

    dataloader = DataLoader(dataset=dataset_fluo, batch_size=4)
    val_dataloader = DataLoader(dataset=dataset_fluo_val, batch_size=4)
    trainer = pl.Trainer(max_epochs=3)
    trainer.fit(model_lt, train_dataloaders=dataloader, val_dataloaders=val_dataloader)
