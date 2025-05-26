import tifffile as tiff
import numpy as np
import torch

from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from typing import Callable


def get_slices(
    dataset_shape: tuple[int, int], patch_size: tuple[int, int], stride: tuple[int, int]
) -> list[tuple[slice, slice]]:
    x, y = dataset_shape
    p_x, p_y = patch_size
    s_x, s_y = stride

    # Generate indices in x-direction
    x_indices = np.arange(0, x - p_x + 1, s_x)
    if x_indices[-1] + p_x < x:
        x_indices = np.append(x_indices, x - p_x)

    # Generate indices in y-direction
    y_indices = np.arange(0, y - p_y + 1, s_y)
    if y_indices[-1] + p_y < y:
        y_indices = np.append(y_indices, y - p_y)

    slices = []
    for idx_x in x_indices:
        for idx_y in y_indices:
            slices.append((slice(idx_x, idx_x + p_x), slice(idx_y, idx_y + p_y)))
    return slices


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root_path: Path,
        img_folder: str,
        target_folder: str,
        transform: Callable = None,
        target_transform: Callable = None,
        patch_size: tuple | None = None,
        stride: tuple | None = None,
    ):
        self.path: Path = root_path
        self.img_path = self.path / img_folder
        self.target_path = self.path / target_folder
        self.transform = transform
        self.target_transform = target_transform
        self.images = self._get_sorted_paths(self.img_path)
        self.targets = self._get_sorted_paths(self.target_path)
        self.patch_size = patch_size
        self.stride = stride
        if self.patch_size and self.stride:
            img_slices = []
            for img_path in self.images:
                img = tiff.imread(img_path)
                img_slices.append(
                    get_slices(
                        dataset_shape=img.shape,
                        patch_size=self.patch_size,
                        stride=self.stride,
                    )
                )
            # self.slices is a list of tuples that contain two elements: (img_index, slice)
            # with its length equal to the total number of patches
            # self.slices[i] returns the idx of the image and the slice for that image
            self.slices = [
                (img_idx, sl)
                for img_idx, slices in enumerate(img_slices)
                for sl in slices
            ]
        else:
            self.slices = None

    def _get_sorted_paths(self, path):
        return sorted(path.iterdir(), key=lambda x: x.name)

    def __getitem__(self, index):
        if self.slices:
            img_idx, sl = self.slices[index]
            # if slices are defined, img_idx != dataset index
            index = img_idx

        image = tiff.imread(self.images[index])
        target = tiff.imread(self.targets[index])

        if image.dtype == np.uint16:
            # take care of fluo dataset which is in uint16
            image = image.astype(np.float32)

        # transform target
        target = transforms.ToTensor()(target).to(torch.long)
        target = (target > 0).to(
            torch.float
        )  # map instance segmentation to binary segmentation (foreground / background)

        if self.slices:
            # apply slicing to the last two dimensions (width, height)
            image = image[..., *sl]
            target = target[..., *sl]

        # transform input
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        if self.slices:
            return len(self.slices)
        return len(self.images)


if __name__ == "__main__":

    from constants import FLUO_PATH, FLUO_MEAN, FLUO_STD

    dataset = SegmentationDataset(
        root_path=FLUO_PATH,
        img_folder="01",
        target_folder="01_ST/SEG",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[FLUO_MEAN], std=[FLUO_STD]
                ),  # Normalize to [-1, 1]
            ]
        ),
    )
    print(dataset[0][0].shape)
