from collections import OrderedDict

import torch.nn as nn
from torch.nn import Upsample

from .dense import DenseBlock
from .resnet import ResNetExt
from .resnext import ResNextExt

# from .upsample import UpSample2x


class HoVerNetExt(nn.Module):
    """Initialise HoVer-Net."""

    def __init__(
        self,
        num_types=None,
        freeze=False,
        backbone_name="resnet",
        pretrained_backbone=None,
    ):
        super().__init__()
        self.freeze = freeze
        self.num_types = num_types
        self.output_ch = 3 if num_types is None else 4

        self.backbone = self.get_backbone(
            backbone_name, pretrained=pretrained_backbone
        )

        self.conv_bot = nn.Conv2d(
            2048, 1024, 1, stride=1, padding=0, bias=False
        )

        def create_decoder_branch(out_ch=2, ksize=5):
            pad = ksize // 2
            module_list = [
                nn.Conv2d(1024, 256, ksize, stride=1, padding=pad, bias=False),
                DenseBlock(256, [1, ksize], [128, 32], 8, split=4),
                nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
            ]
            u3 = nn.Sequential(*module_list)

            module_list = [
                nn.Conv2d(512, 128, ksize, stride=1, padding=pad, bias=False),
                DenseBlock(128, [1, ksize], [128, 32], 4, split=4),
                nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            ]
            u2 = nn.Sequential(*module_list)

            module_list = [
                nn.Conv2d(256, 64, ksize, stride=1, padding=pad, bias=False),
            ]
            u1 = nn.Sequential(*module_list)

            module_list = [
                nn.BatchNorm2d(64, eps=1e-5),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),
            ]
            u0 = nn.Sequential(*module_list)

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)])
            )
            return decoder

        ksize = 3
        if num_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(
                            ksize=ksize, out_ch=num_types
                        )),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        # self.upsample2x = UpSample2x()
        self.upsample2x = Upsample(scale_factor=2)

    def get_backbone(self, backbone_name, pretrained):
        if backbone_name == "resnet":
            return ResNetExt.resnet50(3, pretrained=pretrained)
        elif backbone_name == "resnext":
            return ResNextExt(3, pretrained=pretrained)
        else:
            raise NotImplementedError

    def forward(self, imgs):
        imgs = imgs / 255.0  # to 0-1 range to match XY

        d0, d1, d2, d3 = self.backbone(imgs, self.freeze)
        d3 = self.conv_bot(d3)
        d = [d0, d1, d2, d3]

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict
