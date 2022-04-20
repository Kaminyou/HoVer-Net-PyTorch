import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np
from hover_net.dataloader.augmentation import (add_to_brightness,
                                               add_to_contrast, add_to_hue,
                                               add_to_saturation,
                                               gaussian_blur, median_blur)
from hover_net.dataloader.preprocessing import cropping_center, gen_targets
from imgaug import augmenters as iaa
from PIL import Image
from pycocotools.coco import COCO as _COCO
from torch.utils.data import Dataset


class COCO(_COCO):
    def __init__(self, annotation_file=None, class_mapping=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            if class_mapping is not None:
                print('Mapping to new classes')
                classes = sorted(list(set(class_mapping.values())))
                new_dataset = []
                for i, _class in enumerate(classes):
                    new_dataset.append({
                        'supercategory': _class,
                        'name': _class,
                        'id': i + 1
                    })
                class_id_mapping = {}
                for category in dataset['categories']:
                    raw_name = category['name']
                    raw_id = category['id']
                    new_name = class_mapping[raw_name]
                    class_id_mapping[raw_id] = classes.index(new_name) + 1
                new_annotations = []
                for annotation in dataset['annotations']:
                    new_annotation = annotation
                    raw_category_id = new_annotation['category_id']
                    new_annotation['category_id'] = class_id_mapping[raw_category_id]
                    new_annotations.append(new_annotation)
                dataset = {
                    'images': dataset['images'],
                    'annotations': new_annotations,
                    'categories': new_dataset
                }
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

class COCODataset(Dataset):
    """`MS Coco Detection
    <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in an
            PIL image and returns a transformed version. E.g,
            ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
    """

    def __init__(
        self,
        ann_file,
        classes,
        class_mapping=None,
        input_shape=(512, 512),
        mask_shape=(512, 512),
        setup_augmentor=True,
        test_mode=False,
        filter_empty_gt=True,
    ):

        self.ann_file = ann_file
        self.classes = classes
        if class_mapping is not None:
            with open(class_mapping) as f:
                self.class_mapping = json.load(f)
        else:
            self.class_mapping = None
        self.input_shape = input_shape
        self.mask_shape = mask_shape
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt

        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        self.id = 0
        if setup_augmentor:
            self.setup_augmentor(0, 0)

        # processing pipeline
        # self.transform = transform
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def __len__(self):
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCO(ann_file, self.class_mapping)
        self.cat_ids = self.coco.getCatIds(catNms=self.classes)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info["filename"] = info["file_name"]
            data_infos.append(info)
            ann_ids = self.coco.getAnnIds(imgIds=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids
        ), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_["image_id"] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.catToImgs[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info["width"], img_info["height"]) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)

        self.img_ids = valid_img_ids

        return valid_inds

    def setup_augmentor(self, worker_id, seed):
        self.augmentor = self.__get_augmentation(seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id
        return

    def get_annotation(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]["id"]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def _parse_ann_info(self, img_info, ann_info):

        classes_dict = {}
        instance_mask = []
        mask_idx = 1

        for ann in ann_info:
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            
            gt_label = self.cat2label[ann['category_id']]
            classes_dict[mask_idx] = gt_label
            mask_idx += 1
            instance_mask.append(self.coco.annToMask(ann))

        instance_mask = np.array(instance_mask)
        arange_array = np.arange(1, mask_idx).reshape(-1, 1, 1)
        instance_mask = np.max(instance_mask * arange_array, axis=0)

        # add background mapping into classes_dict
        classes_dict[0] = 0

        # map to the category mask
        category_mask = np.vectorize(classes_dict.get)(instance_mask)

        return instance_mask.astype("int32"), category_mask.astype("int32")

    def prepare_train_img(self, idx):

        img_info = self.data_infos[idx]
        instance_mask, category_mask = self.get_annotation(idx)
        img_id = img_info["id"]
        image = cv2.imread(img_info['filename'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # augmentation
        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            image = shape_augs.augment_image(image)
            instance_mask = shape_augs.augment_image(instance_mask)
            category_mask = shape_augs.augment_image(category_mask)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            image = input_augs.augment_image(image)

        image = cropping_center(image, self.input_shape)
        feed_dict = {"img": image.copy(), "img_id": img_id}

        category_mask = cropping_center(category_mask, self.mask_shape)
        feed_dict["tp_map"] = category_mask.copy()

        if not np.any(category_mask):
            return None

        target_dict = gen_targets(instance_mask.copy(), self.mask_shape, img_id)
        feed_dict.update(target_dict)

        return feed_dict
    
    def prepare_test_img(self, idx):

        img_info = self.data_infos[idx]
        img_id = img_info['image_id']
        image = cv2.imread(img_info['filename'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        feed_dict = {"img": image.copy(), "img_id": img_id}

        return feed_dict

    def __get_augmentation(self, rng):
        if not self.test_mode:
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
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]

            input_augs = [
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: gaussian_blur(
                                *args,
                                max_ksize=3
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: median_blur(
                                *args,
                                max_ksize=3
                            ),
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
                            func_images=lambda *args: add_to_hue(
                                *args, range=(-8, 8)
                            ),
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
            ]
            input_augs = []

        return shape_augs, input_augs
