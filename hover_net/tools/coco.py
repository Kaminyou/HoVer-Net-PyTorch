import contextlib
import copy
import io
import itertools
import json
import os
import random
import string
from collections import OrderedDict

import cv2
import numpy as np
from hover_net.postprocess import process
from hover_net.process import infer_step
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import encode
from terminaltables import AsciiTable
from tidecv import TIDE, datasets

tide_mode_mapping = {"bbox": TIDE.BOX, "segm": TIDE.MASK}


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bytes):
            return obj.decode("utf-8")
        return super(NpEncoder, self).default(obj)


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

    detection_dict = {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, y, width, height],
        "score": score
    }

    segmentation_dict = {
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": rle,
        "score": score
    }
    return detection_dict, segmentation_dict


def infer_for_coco_evaluation(dataloader, model, device, nr_types):
    detection_list = []
    segmentation_list = []
    for step_idx, data in enumerate(dataloader):

        test_result_output = infer_step(
            batch_data=data["img"],
            model=model,
            device=device
        )
        image_ids = data["img_id"]

        for curr_image_idx in range(len(test_result_output)):
            pred_inst, inst_info_dict = process(
                test_result_output[curr_image_idx],
                nr_types=nr_types,
                return_centroids=True
            )

            for single_inst_info in inst_info_dict.values():
                detection_dict, segmentation_dict = parse_single_instance(
                    image_ids[curr_image_idx].item(),
                    single_inst_info
                )
                detection_list.append(detection_dict)
                segmentation_list.append(segmentation_dict)

    return detection_list, segmentation_list


def coco_evaluation_pipeline(
    dataloader,
    model,
    device,
    nr_types,
    cat_ids,
    tide_evaluation=False
):
    """Pipeline for coco dataset evaluation.

    Args:
        dataloader (torch.utils.data.DataLoader): dataloader with COCO dataset.
        model (torch.nn): The HoVer-Net model.
        device (str): cpu or cuda.
        nr_types (int): # of types should be output.
        cat_ids (tuple of int): which category should be evaluated.

    Returns:
        eval_results (dict): The evaluation results.
    """

    metrics = ["bbox", "segm"]
    coco_metric_names = {
        'mAP': 0,
        'mAP_50': 1,
        'mAP_75': 2,
        'mAP_s': 3,
        'mAP_m': 4,
        'mAP_l': 5,
        'AR@100': 6,
        'AR@300': 7,
        'AR@1000': 8,
        'AR_s@1000': 9,
        'AR_m@1000': 10,
        'AR_l@1000': 11
    }
    classwise = True
    metric_items = None
    eval_results = OrderedDict()

    detection_list, segmentation_list = infer_for_coco_evaluation(
        dataloader=dataloader,
        model=model,
        device=device,
        nr_types=nr_types
    )

    predictions = {
        "bbox": copy.deepcopy(detection_list),
        "segm": copy.deepcopy(segmentation_list)
    }

    for metric in metrics:
        if tide_evaluation:
            temp_file = os.path.join("/tmp", f"{id_generator()}.json")
            with open(temp_file, "w") as f:
                json.dump(predictions[metric], f, indent=4, cls=NpEncoder)

            tide = TIDE()
            # Use TIDE.MASK for masks
            tide.evaluate(
                datasets.COCO(dataloader.dataset.ann_file),
                datasets.COCOResult(temp_file), mode=tide_mode_mapping[metric])
            # Summarize the results as tables in the console
            tide.summarize()
            # Show a summary figure. Specify a folder
            # and it'll output a png to that folder.
            tide.plot()
            os.remove(temp_file)

        coco_gt = dataloader.dataset.coco
        coco_det = dataloader.dataset.coco.loadRes(predictions[metric])

        cocoEval = COCOeval(dataloader.dataset.coco, coco_det, metric)
        if cat_ids is None:
            cocoEval.params.catIds = coco_gt.getCatIds()
        else:
            cocoEval.params.catIds = cat_ids
        # cocoEval.params.imgIds = self.img_ids
        # cocoEval.params.maxDets = list(proposal_nums)
        cocoEval.params.iouThrs = [0.1]

        cocoEval.evaluate()
        cocoEval.accumulate()

        # Save coco summarize print information to logger
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        print('\n' + redirect_string.getvalue())

        if classwise:  # Compute per-category AP
            # Compute per-category AP
            # from https://github.com/facebookresearch/detectron2/
            precisions = cocoEval.eval['precision']
            # precision: (iou, recall, cls, area range, max dets)
            assert len(cat_ids) == precisions.shape[2]

            results_per_category = []
            for idx, catId in enumerate(cat_ids):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = coco_gt.loadCats(catId)[0]
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                if precision.size:
                    ap = np.mean(precision)
                else:
                    ap = float('nan')
                results_per_category.append(
                    (f'{nm["name"]}', f'{float(ap):0.3f}'))

            num_columns = min(6, len(results_per_category) * 2)
            results_flatten = list(
                itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (num_columns // 2)
            results_2d = itertools.zip_longest(*[
                results_flatten[i::num_columns]
                for i in range(num_columns)
            ])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            print('\n' + table.table)

        if metric_items is None:
            metric_items = [
                'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
            ]

        for metric_item in metric_items:
            key = f'{metric}_{metric_item}'
            val = float(
                f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
            )
            eval_results[key] = val
        ap = cocoEval.stats[:6]
        eval_results[f'{metric}_mAP_copypaste'] = (
            f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
            f'{ap[4]:.3f} {ap[5]:.3f}')
    return eval_results
