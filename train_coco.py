import argparse
import os

import torch
import torch.optim as optim

from hover_net.dataloader import get_dataloader
from hover_net.models import HoVerNetExt
from hover_net.process import proc_valid_step_output, train_step, valid_step
from hover_net.tools.coco import coco_evaluation_pipeline
from hover_net.tools.utils import (dump_yaml, read_yaml,
                                   update_accumulated_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train model with dataset in COCO format")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="yaml config file path"
    )
    args = parser.parse_args()

    config = read_yaml(args.config)

    # deal with coco evalution cat ids
    if config["EVAL"]["COCO_EVAL_CAT_IDS"] is not None:
        config["EVAL"]["COCO_EVAL_CAT_IDS"] = tuple(
            config["EVAL"]["COCO_EVAL_CAT_IDS"]
        )

    train_dataloader = get_dataloader(
        dataset_type="coco",
        ann_file=config["DATA"]["TRAIN_COCO_JSON"],
        classes=config["DATA"]["CLASSES"],
        class_mapping=config["DATA"]["CLASS_MAPPING"],
        input_shape=(
            config["DATA"]["PATCH_SIZE"]['HEIGHT'],
            config["DATA"]["PATCH_SIZE"]['WIDTH']
        ),
        mask_shape=(
            config["DATA"]["PATCH_SIZE"]['HEIGHT'],
            config["DATA"]["PATCH_SIZE"]['WIDTH']
        ),
        batch_size=config["TRAIN"]["BATCH_SIZE"],
        run_mode="train",
    )
    val_dataloader = get_dataloader(
        dataset_type="coco",
        ann_file=config["DATA"]["VALID_COCO_JSON"],
        classes=config["DATA"]["CLASSES"],
        class_mapping=config["DATA"]["CLASS_MAPPING"],
        input_shape=(
            config["DATA"]["PATCH_SIZE"]['HEIGHT'],
            config["DATA"]["PATCH_SIZE"]['WIDTH']
        ),
        mask_shape=(
            config["DATA"]["PATCH_SIZE"]['HEIGHT'],
            config["DATA"]["PATCH_SIZE"]['WIDTH']
        ),
        batch_size=config["TRAIN"]["BATCH_SIZE"],
        run_mode="val",
    )

    model = HoVerNetExt(
        backbone_name=config["MODEL"]["BACKBONE"],
        pretrained_backbone=config["MODEL"]["PRETRAINED"],
        num_types=config["MODEL"]["NUM_TYPES"],
    )
    optimizer = optim.Adam(model.parameters(), lr=1.0e-4, betas=(0.9, 0.999))

    model.to(config["TRAIN"]["DEVICE"])

    os.makedirs(config["LOGGING"]["SAVE_PATH"], exist_ok=True)
    dump_yaml(
        os.path.join(
            config["LOGGING"]["SAVE_PATH"],
            "config.yaml"
        ),
        config
    )

    for epoch in range(config["TRAIN"]["EPOCHS"]):
        accumulated_output = {}
        for step_idx, data in enumerate(train_dataloader):
            train_result_dict = train_step(
                epoch,
                step_idx,
                batch_data=data,
                model=model,
                optimizer=optimizer,
                device=config["TRAIN"]["DEVICE"],
                show_step=1,
                verbose=config["LOGGING"]["VERBOSE"],
            )

        for step_idx, data in enumerate(val_dataloader):
            valid_result_dict = valid_step(
                epoch, step_idx,
                batch_data=data,
                model=model,
                device=config["TRAIN"]["DEVICE"]
            )
            update_accumulated_output(accumulated_output, valid_result_dict)

        out_dict = proc_valid_step_output(accumulated_output)

        print(
            f"[Epoch {epoch + 1} / {config['TRAIN']['EPOCHS']}] Val || "
            f"ACC={out_dict['scalar']['np_acc']:.3f} || "
            f"DICE={out_dict['scalar']['np_dice']:.3f} || "
            f"MSE={out_dict['scalar']['hv_mse']:.3f}"
        )

        if (epoch + 1) % config["LOGGING"]["SAVE_STEP"] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    config["LOGGING"]["SAVE_PATH"],
                    f"epoch_{epoch + 1}.pth"
                )
            )

        if (epoch + 1) % config["EVAL"]["COCO_EVAL_STEP"] == 0:
            coco_evaluation_pipeline(
                dataloader=val_dataloader,
                model=model,
                device=config["TRAIN"]["DEVICE"],
                nr_types=config["DATA"]["NUM_TYPES"],
                cat_ids=config["EVAL"]["COCO_EVAL_CAT_IDS"]
            )

    torch.save(
        model.state_dict(),
        os.path.join(config["LOGGING"]["SAVE_PATH"], "latest.pth")
    )
