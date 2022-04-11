import time

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from stardist import edt_prob, star_dist
from stardist.utils import mask_to_categorical
from torch.utils.data import Dataset
class COCODataset(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, 
                 ann_file,
                 classes,
                 n_rays,
                 transform,
                 target_transform,
                 test_mode=False,
                 filter_empty_gt=True):
        
        self.ann_file = ann_file
        self.classes = classes
        self.n_rays = n_rays
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)
        
        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]

        # processing pipeline
        self.transform = transform
        self.target_transform = target_transform
    
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
        data = self.prepare_train_img(idx)
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
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds(catNms=self.classes)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.getAnnIds(imgIds=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos
    
    def get_annotation(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    
    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths.
        """
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
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
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
                
        self.img_ids = valid_img_ids
        
        return valid_inds
    
    def _parse_ann_info(self, img_info, ann_info):

        classes_dict = {}
        gt_masks = []
        mask_idx = 1
        
        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            
            category_id = ann['category_id']
            classes_dict[mask_idx] = category_id
            mask_idx += 1
            gt_masks.append(self.coco.annToMask(ann))
        
        gt_masks = np.array(gt_masks)
        arange_array = np.arange(1, mask_idx).reshape(-1, 1, 1)
        gt_masks = np.max(gt_masks * arange_array, axis=0)
        
        return gt_masks, classes_dict
    
    
    def prepare_train_img(self, idx):

        img_info = self.data_infos[idx]
        gt_masks, classes_dict = self.get_annotation(idx)
        
        image = Image.open(img_info['filename']).convert('RGB')
        image = self.transform(image)
        
        distances = star_dist(gt_masks, self.n_rays, mode="cpp")
        obj_probabilities = edt_prob(gt_masks)
        obj_probabilities = obj_probabilities[..., None]
        
        semantic_mask = mask_to_categorical(gt_masks, len(self.classes), classes_dict)

        distances = self.target_transform(distances)
        obj_probabilities = self.target_transform(obj_probabilities)
        semantic_mask = self.target_transform(semantic_mask)
        
        return image, obj_probabilities, distances, semantic_mask

    def prepare_test_img(self, idx):
        
        img_info = self.data_infos[idx]
        image = Image.open(img_info['filename']).convert('RGB')
        image = self.transform(image)
        
        return image