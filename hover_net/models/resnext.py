import torch
import torch.nn as nn
import torchvision.models as models


class ResNextExt(nn.Module):
    def __init__(self, num_input_channels, pretrained=None):
        super(ResNextExt, self).__init__()
        self.backbone = models.resnext50_32x4d(pretrained=pretrained)
        self.backbone.conv1 = nn.Conv2d(
            num_input_channels, 64, 7, stride=1, padding=3
        )

    def forward(self, x, freeze=False):
        if self.training:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            with torch.set_grad_enabled(not freeze):
                x1 = x = self.backbone.layer1(x)
                x2 = x = self.backbone.layer2(x)
                x3 = x = self.backbone.layer3(x)
                x4 = x = self.backbone.layer4(x)
        else:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x1 = x = self.backbone.layer1(x)
            x2 = x = self.backbone.layer2(x)
            x3 = x = self.backbone.layer3(x)
            x4 = x = self.backbone.layer4(x)
        return x1, x2, x3, x4
