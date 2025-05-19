import tifffile as tiff
import numpy as np
import torch

from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from typing import Callable


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root_path: Path,
        img_folder: str,
        target_folder: str,
        transform: Callable = None,
        target_transform: Callable = None
    ):
        self.path: Path = root_path
        self.img_path = self.path / img_folder
        self.target_path = self.path / target_folder
        self.transform = transform
        self.target_transform = target_transform
        self.images = self._get_sorted_paths(self.img_path)
        self.targets = self._get_sorted_paths(self.target_path)

        assert len(self.images) == len(self.targets), (
            "Img and target datsets don\t match"
        )

    def _get_sorted_paths(self, path):
        return sorted(path.iterdir(), key=lambda x: x.name)

    def __getitem__(self, index):
        image = tiff.imread(self.images[index])
        target = tiff.imread(self.targets[index])

        if image.dtype == np.uint16:
            # take cake of fluo dataset which is in uint16
            image = image.astype(np.float32)
        # transform input
        if self.transform:
            image = self.transform(image)

        # transform target
        target = transforms.ToTensor()(target).to(torch.long)
        target = (target > 0).to(torch.float)  # map instance segmentation to binary segmentation (foreground / background)

        if self.target_transform:
            target = self.target_transform(target)
            
        return image, target

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
   
    from constants import (
        FLUO_PATH, FLUO_MEAN, FLUO_STD
    )
    
    dataset = SegmentationDataset(
        root_path=FLUO_PATH,
        img_folder="01",
        target_folder="01_ST/SEG",
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[FLUO_MEAN], std=[FLUO_STD]),  # Normalize to [-1, 1]
            ]
        )
    )
    print(dataset[0][0].shape)