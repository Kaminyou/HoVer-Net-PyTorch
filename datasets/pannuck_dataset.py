import abc
import os

import cv2
import numpy as np
from imgaug import augmenters as iaa

from dataloader.augmentation import (add_to_brightness, add_to_contrast,
                                     add_to_hue, add_to_saturation,
                                     gaussian_blur, median_blur)
from dataloader.preprocessing import cropping_center, gen_targets
from datasets.hover_dataset import HoVerDatasetBase


class PanNuckDataset(HoVerDatasetBase):
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
        with_type=False,
        input_shape=None,
        mask_shape=None,
        run_mode="train",
        setup_augmentor=True,
    ):
        assert input_shape is not None and mask_shape is not None
        self.run_mode = run_mode
        self.data_path = data_path
        self.load_all_data()

        self.with_type = with_type
        self.mask_shape = mask_shape
        self.input_shape = input_shape
        self.id = 0
        if setup_augmentor:
            self.setup_augmentor(0, 0)
        return

    def setup_augmentor(self, worker_id, seed):
        self.augmentor = self.__get_augmentation(self.run_mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id
        return

    def load_all_data(self):
        data_list = os.listdir(self.data_path)
        self.data = []
        for data_name in data_list:
            if data_name.rsplit('.', 1)[1] != "npy": continue
            self.data.append(os.path.join(self.data_path, data_name))

    def load_data(self, idx):
        data = np.load(self.data[idx])
        img = data[:, :, :3][:, :, ::-1].astype("uint8") # BGR -> RGB
        ann = data[:, :, 3:].astype("int32")
        return img, ann
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, ann = self.load_data(idx)
        
        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            ann = shape_augs.augment_image(ann)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

        img = cropping_center(img, self.input_shape)
        feed_dict = {"img": img}

        inst_map = ann[..., 0]  # HW1 -> HW
        if self.with_type:
            type_map = (ann[..., 1]).copy()
            type_map = cropping_center(type_map, self.mask_shape)
            feed_dict["tp_map"] = type_map

        target_dict = gen_targets(
            inst_map, self.mask_shape
        )
        feed_dict.update(target_dict)

        return feed_dict

    def __get_augmentation(self, mode, rng):
        if mode == "train":
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                # ! for pannuke v0, no rotation or translation,
                # ! just flip to avoid mirror padding
                # iaa.Affine(
                #     # scale images to 80-120% of their size,
                #     # individually per axis
                #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                #     # translate by -A to +A percent (per axis)
                #     translate_percent={
                #         "x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                #     shear=(-5, 5),  # shear by -5 to +5 degrees
                #     rotate=(-179, 179),  # rotate by -179 to +179 degrees
                #     order=0,  # use nearest neighbour
                #     backend="cv2",  # opencv for fast processing
                #     seed=rng,
                # ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0],
                    self.input_shape[1],
                    position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]

            input_augs = [
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: median_blur(*args, max_ksize=3),
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                    ]
                ),
                iaa.Sequential(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        else:
            shape_augs = [
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs

class PanNuckInferenceDataset(HoVerDatasetBase):
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
        self.id = 0

    def load_all_data(self):
        data_list = os.listdir(self.data_path)
        self.data = []
        for data_name in data_list:
            if not data_name.rsplit('.', 1)[1] in ["png", "jpg", "tiff"]: continue
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
