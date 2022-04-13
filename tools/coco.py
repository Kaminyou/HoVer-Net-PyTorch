import cv2
import numpy as np
from pycocotools.mask import encode


def parse_single_instance(image_id, single_inst_info):
    # bbox
    x = single_inst_info['bbox'][0][1]
    y = single_inst_info['bbox'][0][0]
    width = single_inst_info['bbox'][1][1] - x
    height = single_inst_info['bbox'][1][0] - y
    
    # category_id
    category_id = single_inst_info['type']
    
    # score
    score = single_inst_info['type_prob']
    
    # rle
    mask = np.zeros((512, 512), dtype="uint8")
    mask = cv2.drawContours(mask, [single_inst_info['contour']], -1, 255, -1)
    mask = mask.astype(bool)
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = encode(mask)
    
    detection_dict = {"image_id":image_id, "category_id":category_id, "bbox":[x,y,width,height], "score":score}
    segmentation_dict = {"image_id":image_id, "category_id":category_id, "segmentation": rle, "score":score}
    return detection_dict, segmentation_dict