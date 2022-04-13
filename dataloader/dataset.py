from torch.utils.data import DataLoader
from datasets.coco_dataset import COCODataset
from datasets.pannuck_dataset import PanNuckDataset, PanNuckInferenceDataset


def get_dataloader(
    dataset_type, 
    data_path=None, 
    with_type=True, 
    input_shape=None, 
    mask_shape=None, 
    batch_size=1, 
    run_mode="train",
    ann_file=None,
    classes=None
):
    if dataset_type.lower() == "pannuck":
        if run_mode == "inference":
            dataset = PanNuckInferenceDataset(
                data_path=data_path, 
                input_shape=input_shape
            )
        else:
            dataset = PanNuckDataset(
                data_path=data_path,
                with_type=with_type,
                input_shape=input_shape,
                mask_shape=mask_shape,
                run_mode=run_mode,
                setup_augmentor=True
            )
    elif dataset_type.lower() == "coco":
        if run_mode == "inference":
            raise NotImplementedError
        else:
            assert ann_file is not None
            assert classes is not None
            test_mode = False if run_mode == "train" else True
            dataset = COCODataset(
                ann_file=ann_file,
                classes=classes,
                input_shape=input_shape,
                mask_shape=mask_shape,
                test_mode=test_mode
            )
    else:
        raise NotImplementedError

    sulffle = True if run_mode == "train" else False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=sulffle, num_workers=8, pin_memory=True)
    return dataloader