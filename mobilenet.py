from torch import nn
from torchvision.models import MobileNetV2
import torch


class MobileNetV2Encoder(MobileNetV2):
    """
    MobileNetV2Encoder inherits from torchvision's official MobileNetV2. It is modified to
    use dilation on the last block to maintain output stride 16, and deleted the
    classifier block that was originally used for classification. The forward method
    additionally returns the feature maps at all resolutions for decoder's use.
    """

    def __init__(self, in_channels, norm_layer=None):
        super().__init__()

        # Replace first conv layer if in_channels doesn't match.
        if in_channels != 3:
            self.features[0][0] = nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False)

        # Remove last block
        self.features = self.features[:-1]

        # Change to use dilation to maintain output stride = 16
        self.features[14].conv[1][0].stride = (1, 1)
        for feature in self.features[15:]:
            feature.conv[1][0].dilation = (2, 2)
            feature.conv[1][0].padding = (2, 2)

        # Delete classifier
        del self.classifier

        self.layer1 = nn.Sequential(self.features[0], self.features[1])
        self.layer2 = nn.Sequential(self.features[2], self.features[3])
        self.layer3 = nn.Sequential(self.features[4], self.features[5], self.features[6])
        self.layer4 = nn.Sequential(self.features[7], self.features[8], self.features[9], self.features[10],
                                    self.features[11], self.features[12], self.features[13])
        self.layer5 = nn.Sequential(self.features[14], self.features[15], self.features[16], self.features[17])
    def forward(self, x):
        x0 = x  # 1/1
        x = self.features[0](x)
        x = self.features[1](x)
        x = x 
        x1 = x  # 1/2
        x = self.features[2](x)
        x = self.features[3](x)
        x2 = x  # 1/4
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        x3 = x  # 1/8
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        x4 = x # 1/16
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        x = self.features[17](x)
        x5 = x # 1/16
        return x1,x2,x3,x4,x5

