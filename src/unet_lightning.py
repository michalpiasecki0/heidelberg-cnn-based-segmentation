from typing import Type

import pytorch_lightning as pl
import torch.optim as optim
from segmentation_models_pytorch.losses import DiceLoss
from torch.nn import BCEWithLogitsLoss

from src.constants import LossWeights
from src.metrics import intersection_over_union, pixel_accuracy
from src.unet import UNet2d


class UNetSegmentationModule(pl.LightningModule):
    def __init__(
        self,
        model_kwargs: dict,
        learning_rate: float,
        loss_weights: LossWeights,
        optimizer_cls: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: dict = None,
    ):
        """
        Args:
            model_kwargs (dict): kwargs to pass to UNet
            learning_rate (float):
            loss_weights (list): weight for Cross Entropy and DiceLoss (we are supposed to use these two losses)
            optimizer_cls (Type[optim.Optimizer], optional)
            optimizer_kwargs (dict, optional)
        """
        super().__init__()
        self.save_hyperparameters()

        # define model
        self.unet_kwargs = model_kwargs
        self.model = UNet2d(**model_kwargs)

        # define losses
        self.loss_weights: LossWeights = loss_weights
        self.bce_loss = BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(mode="binary")

        # training parameters
        self.learning_rate = learning_rate
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        # inference
        images, masks = batch
        logits = self(images)

        # calculate loss  set weight to 0 if you want to train on single loss 
        loss_cross = self.bce_loss(logits, masks)
        loss_dice = self.dice_loss(logits, masks)
        loss = (
            self.loss_weights.cross_entropy * loss_cross
            + self.loss_weights.dice * loss_dice
        )

        # get metrics
        acc = pixel_accuracy(logits, masks)
        iou = intersection_over_union(logits, masks)

        # log
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_pixel_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_iou", iou, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        
        # inference
        images, masks = batch
        logits = self(images)
        
        # calculate loss  set weight to 0 if you want to train on single loss 
        loss_cross = self.bce_loss(logits, masks)
        loss_dice = self.dice_loss(logits, masks)
        loss = (
            self.loss_weights.cross_entropy * loss_cross
            + self.loss_weights.dice * loss_dice
        )
        
        # get metrics
        acc = pixel_accuracy(logits, masks)
        iou = intersection_over_union(logits, masks)   

        # log
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_pixel_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_iou", iou, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer_params = {"lr": self.learning_rate, **self.optimizer_kwargs}
        return self.optimizer_cls(self.parameters(), **optimizer_params)


