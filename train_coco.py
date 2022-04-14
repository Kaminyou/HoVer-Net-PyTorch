import argparse
import os

import torch
import torch.optim as optim

from dataloader.dataset import get_dataloader
from models.hovernet import HoVerNetExt
from process.train import train_step
from process.utils import proc_valid_step_output
from process.validate import valid_step
from tools.utils import update_accumulated_output
from tools.coco import coco_evaluation_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train model with dataset in COCO format")
    parser.add_argument(
        "--train_coco",
        type=str,
        required=True,
        help="Path to the config file."
    )
    parser.add_argument(
        "--valid_coco",
        type=str,
        required=True,
        help="Path to the config file."
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        help="Patch size",
        default=512
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )
    parser.add_argument(
        "--num_types",
        type=int,
        required=True
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="./pretrained/resnet50-0676ba61.pth"
    )
    parser.add_argument(
        "--save_step",
        type=int,
        default=5,
        help="Save model for N steps"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./experiments/initial/",
        help="Path to save models"
    )
    parser.add_argument(
        "--coco_eval_step",
        type=int,
        default=5
    )
    parser.add_argument(
        "--coco_eval_cat_ids",
        type=tuple,
        default=(1, 2)
    )
    args = parser.parse_args()

    train_dataloader = get_dataloader(
        dataset_type="coco",
        ann_file=args.train_coco,
        classes=["negative", "positive"],
        input_shape=(512, 512),
        mask_shape=(512, 512),
        batch_size=2,
        run_mode="train",
    )
    val_dataloader = get_dataloader(
        dataset_type="coco",
        ann_file=args.valid_coco,
        classes=["negative", "positive"],
        input_shape=(512, 512),
        mask_shape=(512, 512),
        batch_size=2,
        run_mode="val",
    )

    model = HoVerNetExt(
        pretrained_backbone=args.pretrained,
        num_types=args.num_types,
    )
    optimizer = optim.Adam(model.parameters(), lr=1.0e-4, betas=(0.9, 0.999))

    model.to(args.device)

    os.makedirs(args.save_path, exist_ok=True)

    for epoch in range(args.epochs):
        accumulated_output = {}
        for step_idx, data in enumerate(train_dataloader):
            train_result_dict = train_step(
                epoch,
                step_idx,
                batch_data=data,
                model=model,
                optimizer=optimizer,
                device=args.device,
                show_step=1,
                verbose=False,
            )

        for step_idx, data in enumerate(val_dataloader):
            valid_result_dict = valid_step(
                epoch, step_idx,
                batch_data=data,
                model=model,
                device=args.device
            )
            update_accumulated_output(accumulated_output, valid_result_dict)

        out_dict = proc_valid_step_output(accumulated_output)

        print(
            f"[Epoch {epoch + 1} / {args.epochs}] Val || "
            f"ACC={out_dict['scalar']['np_acc']:.3f} || "
            f"DICE={out_dict['scalar']['np_dice']:.3f} || "
            f"MSE={out_dict['scalar']['hv_mse']:.3f}"
        )

        if (epoch + 1) % args.save_step == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.save_path, f"epoch_{epoch + 1}.pth")
            )

        if (epoch + 1) % args.coco_eval_step == 0:
            coco_evaluation_pipeline(
                dataloader=val_dataloader,
                model=model,
                device=args.device,
                nr_types=args.nr_types,
                cat_ids=args.coco_eval_cat_ids
            )

    torch.save(
        model.state_dict(),
        os.path.join(args.save_path, "latest.pth")
    )
