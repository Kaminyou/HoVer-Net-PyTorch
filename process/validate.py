from collections import OrderedDict

import torch
import torch.nn.functional as F


def valid_step(epoch, step, batch_data, model, device="cuda"):
    model.eval()  # infer mode

    ####
    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]

    imgs_gpu = imgs.to(device).type(torch.float32)  # to NCHW
    imgs_gpu = imgs_gpu.permute(0, 3, 1, 2).contiguous()

    # HWC
    # true_np = torch.squeeze(true_np).to(device).type(torch.int64)
    # true_hv = torch.squeeze(true_hv).to(device).type(torch.float32)

    true_dict = {
        "np": true_np,
        "hv": true_hv,
    }

    if model.num_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = torch.squeeze(true_tp).to(device).type(torch.int64)
        true_dict["tp"] = true_tp

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        pred_dict = model(imgs_gpu)
        pred_list = []
        for k, v in pred_dict.items():
            pred_list.append([k, v.permute(0, 2, 3, 1).contiguous()])
        pred_dict = OrderedDict(pred_list)
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1]
        if model.num_types is not None:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=False)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map

    # * Its up to user to define the protocol to
    # process the raw output per step!
    result_dict = {  # protocol for contents exchange within `raw`
        "raw": {
            "imgs": imgs.numpy(),
            "true_np": true_dict["np"].cpu().numpy(),
            "true_hv": true_dict["hv"].cpu().numpy(),
            "prob_np": pred_dict["np"].cpu().numpy(),
            "pred_hv": pred_dict["hv"].cpu().numpy(),
        }
    }
    if model.num_types is not None:
        result_dict["raw"]["true_tp"] = true_dict["tp"].cpu().numpy()
        result_dict["raw"]["pred_tp"] = pred_dict["tp"].cpu().numpy()
    return result_dict
