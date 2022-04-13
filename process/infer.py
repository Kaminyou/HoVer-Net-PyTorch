from collections import OrderedDict

import torch
import torch.nn.functional as F


def infer_step(batch_data, model, device="cuda"):
    patch_imgs = batch_data

    patch_imgs_gpu = patch_imgs.to(device).type(torch.float32)  # to NCHW
    patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

    ####
    model.eval()  # infer mode

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        pred_dict = model(patch_imgs_gpu)
        pred_list = []
        for k, v in pred_dict.items():
            pred_list.append([k, v.permute(0, 2, 3, 1).contiguous()])
        pred_dict = OrderedDict(pred_list)
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
        if "tp" in pred_dict:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map
        pred_output = torch.cat(list(pred_dict.values()), -1)

    # * Its up to user to define the protocol '
    # to process the raw output per step!
    return pred_output.cpu().numpy()
