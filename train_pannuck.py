import argparse

import torch
import torch.optim as optim

from dataloader.dataset import get_dataloader
from models.hovernet import HoVerNetExt
from process.train import train_step
from process.utils import proc_valid_step_output
from process.validate import valid_step
from tools.utils import update_accumulated_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train model with PanNuck dataset")
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="Path to the config file."
    )
    parser.add_argument(
        "--valid_data_path",
        type=str,
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
    args = parser.parse_args()

    train_dataloader = get_dataloader(
        dataset_type="pannuck",
        data_path=args.train_data_path,
        with_type=True,
        input_shape=(512, 512),
        mask_shape=(512, 512),
        batch_size=2,
        run_mode="train",
    )
    val_dataloader = get_dataloader(
        dataset_type="pannuck",
        data_path=args.valid_data_path,
        with_type=True,
        input_shape=(512, 512),
        mask_shape=(512, 512),
        batch_size=2,
        run_mode="val",
    )
    pretrained_backbone = "/home/kaminyouho/hover_net/" + \
                          "conic/resnet50-0676ba61.pth"
    model = HoVerNetExt(
        pretrained_backbone=pretrained_backbone,
        num_types=5,
    )
    optimizer = optim.Adam(model.parameters(), lr=1.0e-4, betas=(0.9, 0.999))

    model.to(args.device)

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
                verbose=True,
            )

        for step_idx, data in enumerate(val_dataloader):
            valid_result_dict = valid_step(
                epoch,
                step_idx,
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

    torch.save(model.state_dict(), "initial.pth")
