from torch.utils.data import DataLoader

from hover_net.datasets.coco_dataset import COCODataset
from hover_net.datasets.consep_dataset import CoNSePDataset
from hover_net.datasets.inference_dataset import (FolderInferenceDataset,
                                                  SingleInferenceDataset)


def get_dataloader(
    dataset_type=None,
    data_path=None,
    with_type=True,
    input_shape=None,
    mask_shape=None,
    batch_size=1,
    run_mode="train",
    ann_file=None,
    classes=None,
):
    if run_mode == "inference_folder":
        dataset = FolderInferenceDataset(
            data_path=data_path, input_shape=input_shape
        )
    elif run_mode == "inference_single":
        dataset = SingleInferenceDataset(
            data_path_list=data_path, input_shape=input_shape
        )
    elif dataset_type.lower() == "consep":
        dataset = CoNSePDataset(
            data_path=data_path,
            with_type=with_type,
            input_shape=input_shape,
            mask_shape=mask_shape,
            run_mode=run_mode,
            setup_augmentor=True,
        )
    elif dataset_type.lower() == "coco":
        assert ann_file is not None
        assert classes is not None
        test_mode = False if run_mode == "train" else True
        dataset = COCODataset(
            ann_file=ann_file,
            classes=classes,
            input_shape=input_shape,
            mask_shape=mask_shape,
            test_mode=test_mode,
        )
    else:
        raise NotImplementedError

    sulffle = True if run_mode == "train" else False
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sulffle,
        num_workers=8,
        pin_memory=True
    )
    return dataloader
