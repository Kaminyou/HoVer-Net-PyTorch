from collections import OrderedDict

import torch
import torch.nn.functional as F
from hover_net.models.loss import dice_loss, mse_loss, msge_loss, xentropy_loss

loss_opts = {
    "np": {"bce": 1, "dice": 1},
    "hv": {"mse": 1, "msge": 1},
    "tp": {"bce": 1, "dice": 1},
}


def train_step(
    epoch,
    step,
    batch_data,
    model,
    optimizer,
    device="cuda",
    show_step=250,
    verbose=True,
):
    # TODO: synchronize the attach protocol
    loss_func_dict = {
        "bce": xentropy_loss,
        "dice": dice_loss,
        "mse": mse_loss,
        "msge": msge_loss,
    }
    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {"EMA": {}}

    def track_value(name, value):
        result_dict["EMA"].update({name: value})

    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]

    imgs = imgs.to(device).type(torch.float32)
    
    # HWC
    true_np = true_np.to(device).type(torch.int64)
    true_hv = true_hv.to(device).type(torch.float32)

    true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
    true_dict = {
        "np": true_np_onehot,
        "hv": true_hv,
    }

    if model.num_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = torch.squeeze(true_tp).to(device).type(torch.int64)
        true_tp_onehot = F.one_hot(true_tp, num_classes=model.num_types)
        true_tp_onehot = true_tp_onehot.type(torch.float32)
        true_dict["tp"] = true_tp_onehot

    ####
    model.train()
    model.zero_grad()  # not rnn so not accumulate

    pred_dict = model(imgs)
    pred_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
    )
    pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
    if model.num_types is not None:
        pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)

    ####
    loss = 0
    for branch_name in pred_dict.keys():
        for loss_name, loss_weight in loss_opts[branch_name].items():
            loss_func = loss_func_dict[loss_name]
            loss_args = [true_dict[branch_name], pred_dict[branch_name]]
            if loss_name == "msge":
                loss_args.append(true_np_onehot[..., 1])
                loss_args.append(device)
            term_loss = loss_func(*loss_args)
            track_value(
                "loss_%s_%s" % (branch_name, loss_name),
                term_loss.cpu().item()
            )
            loss += loss_weight * term_loss

    track_value("overall_loss", loss.cpu().item())
    # * gradient update

    # torch.set_printoptions(precision=10)
    loss.backward()
    optimizer.step()
    ####
    if verbose:
        out = f"[Epoch {epoch + 1:3d}] {step + 1:4d} || "
        out += f"overall_loss {result_dict['EMA']['overall_loss']:.4f}"
        print(
            out,
            end="\r",
        )

    if ((step + 1) % show_step) == 0:
        # pick 2 random sample from the batch for visualization
        sample_indices = torch.randint(0, true_np.shape[0], (2,))

        imgs = (imgs[sample_indices]).byte()  # to uint8
        imgs = imgs.permute(0, 2, 3, 1).contiguous().cpu().numpy()

        pred_dict["np"] = pred_dict["np"][..., 1]  # return pos only
        pred_dict_detach = {}
        for k, v in pred_dict.items():
            pred_dict_detach[k] = v[sample_indices].detach().cpu().numpy()
        pred_dict = pred_dict_detach

        true_dict["np"] = true_np
        true_dict_detach = {}
        for k, v in true_dict.items():
            true_dict_detach[k] = v[sample_indices].detach().cpu().numpy()
        true_dict = true_dict_detach

        # * Its up to user to define the protocol to
        # process the raw output per step!
        result_dict["raw"] = {  # protocol for contents exchange within `raw`
            "img": imgs,
            "np": (true_dict["np"], pred_dict["np"]),
            "hv": (true_dict["hv"], pred_dict["hv"]),
        }
    return result_dict
