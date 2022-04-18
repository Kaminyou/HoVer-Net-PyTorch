import os

import cv2
from torch.utils.data import Dataset

from hover_net.dataloader.preprocessing import cropping_center

from .hover_dataset import HoVerDatasetBase


class FolderInferenceDataset(HoVerDatasetBase):
    """Data Loader. Loads images from a file list and
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are
    generated.
    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'

    """

    # TODO: doc string

    def __init__(
        self,
        data_path,
        input_shape=None,
    ):
        assert input_shape is not None
        self.data_path = data_path
        self.load_all_data()

        self.input_shape = input_shape

    def load_all_data(self):
        data_list = os.listdir(self.data_path)
        self.data = []
        for data_name in data_list:
            if not data_name.rsplit(".", 1)[1] in ["png", "jpg", "tiff"]:
                continue
            self.data.append(os.path.join(self.data_path, data_name))

    def load_data(self, idx):
        image = cv2.imread(self.data[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("uint8")
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.load_data(idx)
        img = cropping_center(img, self.input_shape)
        return img


class SingleInferenceDataset(Dataset):
    """Data Loader. Loads images from a file list and
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are
    generated.
    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'

    """

    # TODO: doc string

    def __init__(
        self,
        data_path_list,
        input_shape=None,
    ):
        assert input_shape is not None
        self.data_path_list = data_path_list
        self.input_shape = input_shape

    def load_data(self, idx):
        image = cv2.imread(self.data_path_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("uint8")
        return image

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        img = self.load_data(idx)
        img = cropping_center(img, self.input_shape)
        return img
