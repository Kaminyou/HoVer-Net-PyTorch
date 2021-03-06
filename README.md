# HoVer-Net PyTorch
Unofficial PyTorch implementation of [HoVer-Net](https://arxiv.org/abs/1812.06499), which further allows arbitary input and output size and supports **COCO format**. You can check the official implementation [here](https://github.com/vqdang/hover_net). *This repo simplifies the codes (from **Continuation-passing style** to **Direct style**) from the [CoNiC](https://github.com/vqdang/hover_net/tree/conic) branch in the offiical repository.*

- [x] Codes with direct style (higher readability)
- [x] Support arbitary input image size
- [x] Support COCO format input
- [x] Support training with only CPU
- [x] Support [TIDE metric](https://github.com/dbolya/tide) evaluation

## Quick start
1. Please install packages specified in the *requirements.txt*.
2. Please download [Pytorch ImageNet ResNet50  pretrained weights](https://download.pytorch.org/models/resnet50-0676ba61.pth) and put it under `./pretrained/`.

### CoNSeP dataset
Please download the [CoNSeP dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/consep.zip) (or visit its [website](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/)) and follow the [official preprocessing](https://github.com/vqdang/hover_net/blob/master/extract_patches.py) steps. Then, prepare a config file in the yaml format as the following. An example is provided in the `./configs/consep_config.yaml`.
```yaml
DATA:
  TRAIN_DATA_PATH: "./training_data/consep/consep/train/540x540_164x164/"
  VALID_DATA_PATH: "./training_data/consep/consep/valid/540x540_164x164/"
  NUM_TYPES: 5
  PATCH_SIZE: 540
TRAIN:
  DEVICE: "cuda"
  EPOCHS: 50
  BATCH_SIZE: 2
MODEL:
  BACKBONE: "resnet"
  PRETRAINED: "./pretrained/resnet50-0676ba61.pth"
  NUM_TYPES: 5
LOGGING:
  SAVE_STEP: 5
  SAVE_PATH: "./experiments/consep/"
  VERBOSE: TRUE
```
Finally, execute `python3 train_coco.py --config [PATH TO THE YAML CONFIG]`
### COCO format input
If you already have datasets in the COCO format, `train_coco.py` is provided for you to train a HoVer-Net model without any tedious data preprocessing.
Please create a config file in the yaml format first. You can put it in the `./configs/` folder.
```yaml
DATA:
  TRAIN_COCO_JSON: "PATH-TO-THE-TRAIN-JSON"
  VALID_COCO_JSON: "PATH-TO-THE-VALID-JSON"
  CLASSES:
    - "CLS_1"
    - "CLS_2"
    - "CLS_3"
  NUM_TYPES: 3
  PATCH_SIZE: 512
TRAIN:
  DEVICE: "cuda"
  EPOCHS: 50
  BATCH_SIZE: 2
EVAL:
  COCO_EVAL_STEP: 5
  COCO_EVAL_CAT_IDS: [1, 2]
MODEL:
  BACKBONE: "resnet"
  PRETRAINED: "./pretrained/resnet50-0676ba61.pth"
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
- **`DATA::CLASSES`**: a list specifies classes.
- **`DATA::NUM_TYPES`**: this should be `(# of types) + 1` (the `background`).
- **`LOGGING::VERBOSE`**: to show fluctuation of loss at each step.
- **`EVAL::COCO_EVAL_CAT_IDS`**: the classes required to be evaluated. If it is set to `NULL`, all the classes would be evaluated.
- **`MODEL::BACKBONE`**: currently, `resnet` and `resnext` are provided.
- **`MODEL::PRETRAINED`**: might be `str` to specify a path or `bool` to load from PyTorch offical one.

### Model options
| Model | Pretrained  |
| :---:   | :-: |
| resnet | `str` |
| resnext | `bool` |

## Useful APIs
### Infer for one image
- **`tools.api.infer_one_image(image_path, model, device, show)`**
```python
from hover_net.tools.api import infer_one_image

result = infer_one_image(
    image_path="IMAGE-PATH",
    model=model,
    device="cuda",
    show=True,
)
```
### TIDE metric
Besides the standard COCO mAP metric, TIDE is supported.
- **`tools.coco.coco_evaluation_pipeline(dataloader, model, device, nr_types, cat_ids, tide_evaluation=False`**
```python
from hover_net.tools.coco import coco_evaluation_pipeline

coco_evaluation_pipeline(
    dataloader=dataloader,
    model=model,
    device=device,
    nr_types=3,
    cat_ids=(1, 2),
    tide_evaluation=True, # Please modify this flag
)
```