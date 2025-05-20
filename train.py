import pytorch_lightning as pl
import torch
from clearml import Task
from torch.utils.data import DataLoader
from torchvision import transforms

from src.constants import FLUO_PATH, LossWeights
from src.datasets import SegmentationDataset
from src.unet_lightning import UNetSegmentationModule

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

    loss_weights = LossWeights(
        cross_entropy=1.0,  # weight for Cross Entropy loss
        dice=1.0,  # weight for Dice loss
    )

    model_lt = UNetSegmentationModule(
        unet_kwargs, learning_rate, loss_weights=loss_weights
    )

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
