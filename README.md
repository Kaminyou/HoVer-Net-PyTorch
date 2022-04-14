# HoVer-Net PyTorch
Unofficial PyTorch implementation of [HoVer-Net](https://arxiv.org/abs/1812.06499), which further allows arbitary input and output size and supports **COCO format**. You can check the official implementation [here](https://github.com/vqdang/hover_net). *This repo simplifies the codes (from **Continuation-passing style** to **Direct style**) from the [CoNiC](https://github.com/vqdang/hover_net/tree/conic) branch in the offiical repository.*

- [ ] Codes with direct style (higher readability)
- [ ] Support arbitary input size
- [ ] Support COCO format input
- [ ] Support training with only CPU

## Quick start
Please install packages specified in the *requirements.txt*.

### COCO format input
If you already have datasets in the COCO format, `train_coco.py` is provided for you to train a HoVer-Net model without any tedious data preprocessing.
```script
$ python3 train_coco.py \
    --train_coco [Path to the train.json] \
    --valid_coco [Path to the valid.json] \
    --epochs [# of epochs] \
    --device [Device] \
    --num_types [# of categories] \
    --save_step [# of steps to save model weights] \
    --save_path [Path to save model weights] \
    --coco_eval_step [# of steps to evalate] \
    --coco_eval_cat_ids [Categories require evaluation]
```
