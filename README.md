# HoVer-Net PyTorch
Unofficial PyTorch implementation of [HoVer-Net](https://arxiv.org/abs/1812.06499), which further allows arbitary input and output size and supports **COCO format**. You can check the official implementation [here](https://github.com/vqdang/hover_net). *This repo simplifies the codes (from **Continuation-passing style** to **Direct style**) from the [CoNiC](https://github.com/vqdang/hover_net/tree/conic) branch in the offiical repository.*

- [x] Codes with direct style (higher readability)
- [x] Support arbitary input image size
- [x] Support COCO format input
- [x] Support training with only CPU

## Quick start
1. Please install packages specified in the *requirements.txt*.
2. Please download [Pytorch ImageNet ResNet50  pretrained weights](https://download.pytorch.org/models/resnet50-0676ba61.pth) and put it under `./pretrained/`.

### COCO format input
If you already have datasets in the COCO format, `train_coco.py` is provided for you to train a HoVer-Net model without any tedious data preprocessing.
Please create a config file in the yaml format first. You can put it in the `./configs/` folder.
```yaml
DATA:
  TRAIN_COCO_JSON: "PATH-TO-THE-TRAIN-JSON"
  VALID_COCO_JSON: "PATH-TO-THE-VALID-JSON"
  NUM_TYPES: 3
  PATCH_SIZE: 512
TRAIN:
  DEVICE: "cuda"
  EPOCHS: 50
  BATCH_SIZE: 2
  PRETRAINED: "./pretrained/resnet50-0676ba61.pth"
EVAL:
  COCO_EVAL_STEP: 5
  COCO_EVAL_CAT_IDS: [1, 2]
MODEL:
  NUM_TYPES: 3
LOGGING:
  SAVE_STEP: 5
  SAVE_PATH: "./experiments/initial/"
  VERBOSE: TRUE
```
Then
```script
$ python3 train_coco.py --config [PATH TO THE YAML CONFIG]
```
### Option
- **`LOGGING::VERBOSE`**: to show fluctuation of loss at each step.
