import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import time
import timm
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from depth import DepthBranch
from mobilenet import MobileNetV2Encoder


def upsample(x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=True)

class DFMNet(nn.Module):
    def __init__(self, **kwargs):
        super(DFMNet, self).__init__()
        self.rgb = RGBBranch()
        self.depth = DepthBranch()

    def forward(self, r, d):
        size = r.shape[2:]
        outputs = []

        sal_d,feat = self.depth(d)
        sal_final= self.rgb(r,feat)

        sal_final = upsample(sal_final, size)
        sal_d = upsample(sal_d, size)

        outputs.append(sal_final)
        outputs.append(sal_d)

        return outputs

class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,dilation=1, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation=dilation ,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class _ConvBNSig(nn.Module):
    """Conv-BN-Sigmoid"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,dilation=1, **kwargs):
        super(_ConvBNSig, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation=dilation ,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

def _make_layer( block, inplanes, planes, blocks, t=6, stride=1):
    layers = []
    layers.append(block(inplanes, planes, t, stride))
    for i in range(1, blocks):
        layers.append(block(planes, planes, t, 1))
    return nn.Sequential(*layers)

class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out



class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x



class RGBBranch(nn.Module):
    """RGBBranch for low-level RGB feature extract"""

    def __init__(self, c1=16, c2=24, c3=32, c4=96,c5=320,k=32 ,**kwargs):
        super(RGBBranch, self).__init__()
        self.base = MobileNetV2Encoder(3)
        initialize_weights(self.base)

        self.conv_cp1 = _DSConv(c1,k)
        self.conv_cp2 = _DSConv(c2, k)
        self.conv_cp3 = _DSConv(c3, k)
        self.conv_cp4 = _DSConv(c4, k)
        self.conv_cp5 = _DSConv(c5, k)
        self.conv_s_f = nn.Sequential(_DSConv(2 * k,  k),
                                          _DSConv( k, k),
                                          nn.Conv2d(k, 1, 1), )

        # self.focus = focus()
        self.ca1 = nn.Sequential(_ConvBNReLU(k, k, 1, 1), nn.Conv2d(k, k, 1, 1), nn.Sigmoid())
        self.ca2 = nn.Sequential(_ConvBNReLU(k, k, 1, 1), nn.Conv2d(k, k, 1, 1), nn.Sigmoid())
        self.ca3 = nn.Sequential(_ConvBNReLU(k, k, 1, 1), nn.Conv2d(k, k, 1, 1), nn.Sigmoid())
        self.ca4 = nn.Sequential(_ConvBNReLU(k, k, 1, 1), nn.Conv2d(k, k, 1, 1), nn.Sigmoid())
        self.ca5 = nn.Sequential(_ConvBNReLU(k, k, 1, 1), nn.Conv2d(k, k, 1, 1), nn.Sigmoid())
        
        self.conv_r1_tran = _ConvBNReLU(16, 16, 1, 1)
        self.conv_d1_tran = _ConvBNReLU(16, 16, 1, 1)
        self.mlp = nn.Sequential(_ConvBNReLU(48, 24, 1, 1),_ConvBNSig(24,5,1,1))

        self.conv_r1_tran2 = _ConvBNReLU(16, 16, 1, 1)
        self.conv_d1_tran2 = _ConvBNReLU(16, 16, 1, 1)
        self.conv_sgate1 = _ConvBNReLU(16, 16, 3, 1,2,2)
        self.conv_sgate2 = _ConvBNReLU(16, 16, 3, 1,2,2)
        self.conv_sgate3 = _ConvBNSig(16,5,3,1,1)

        self.ppm = PyramidPooling(320, 32)

        self.conv_guide = _ConvBNReLU(320, 16, 1, 1)



    def forward(self, x,feat):

        d1, d2, d3, d4, d5 = feat

        d5_guide = upsample(self.conv_guide(d5),d1.shape[2:])

        r1 = self.base.layer1(x)

        r1t = self.conv_r1_tran(r1)
        d1t = self.conv_d1_tran(d1)
        r1t2 = self.conv_r1_tran2(r1)
        d1t2 = self.conv_d1_tran2(d1)

        # QDW
        iou = F.adaptive_avg_pool2d(r1t * d1t, 1) / \
              (F.adaptive_avg_pool2d(r1t + d1t, 1))

        e_rp = F.max_pool2d(r1t, 2, 2)
        e_dp = F.max_pool2d(d1t, 2, 2)

        e_rp2 = F.max_pool2d(e_rp, 2, 2)
        e_dp2 = F.max_pool2d(e_dp, 2, 2)

        iou_p1 = F.adaptive_avg_pool2d(e_rp * e_dp, 1) / \
                 (F.adaptive_avg_pool2d(e_rp + e_dp, 1))

        iou_p2 = F.adaptive_avg_pool2d(e_rp2 * e_dp2, 1) / \
                 (F.adaptive_avg_pool2d(e_rp2 + e_dp2, 1))

        gate = self.mlp(torch.cat((iou, iou_p1, iou_p2), dim=1))


        # DHA
        mc = r1t2 * d1t2

        sgate = self.conv_sgate1(upsample(mc + d5_guide, d2.shape[2:]))
        d5_guide1 = mc + upsample(sgate, d1.shape[2:])

        sgate = self.conv_sgate1(upsample(mc + d5_guide1, d2.shape[2:]))
        d5_guide2 = mc + upsample(sgate, d1.shape[2:])

        sgate = self.conv_sgate3(d5_guide1 + d5_guide2 + mc)

        dqw1 = gate[:,0:1,...]
        dha1 = upsample(sgate[:, 0:1, ...], d1.shape[2:])
        dqw2 = gate[:, 1:2, ...]
        dha2 = upsample(sgate[:, 1:2, ...], d2.shape[2:])
        dqw3 = gate[:, 2:3, ...]
        dha3 = upsample(sgate[:, 2:3, ...], d3.shape[2:])
        dqw4 = gate[:, 3:4, ...]
        dha4 = upsample(sgate[:, 3:4, ...], d4.shape[2:])
        dqw5 = gate[:, 4:5, ...]
        dha5 = upsample(sgate[:, 4:5, ...], d5.shape[2:])

        r1 = r1 + d1 * dqw1 * dha1 
        r2 = self.base.layer2(r1) + d2 * dqw2 * dha2
        r3 = self.base.layer3(r2) + d3 * dqw3 * dha3 
        r4 = self.base.layer4(r3) + d4 * dqw4 * dha4
        r5 = self.base.layer5(r4) + d5 * dqw5 * dha5
        r6 = self.ppm(r5)

        # Two stage decoder
        ## pre-fusion
        r5 = self.conv_cp5(r5)
        r4 = self.conv_cp4(r4)
        r3 = self.conv_cp3(r3)
        r2 = self.conv_cp2(r2)
        r1 = self.conv_cp1(r1)

        r5 = self.ca5(F.adaptive_avg_pool2d(r5, 1)) * r5
        r4 = self.ca4(F.adaptive_avg_pool2d(r4, 1)) * r4
        r3 = self.ca3(F.adaptive_avg_pool2d(r3, 1)) * r3
        r2 = self.ca2(F.adaptive_avg_pool2d(r2, 1)) * r2
        r1 = self.ca1(F.adaptive_avg_pool2d(r1, 1)) * r1

        r3 = upsample(r3, r1.shape[2:])
        r2 = upsample(r2, r1.shape[2:])
        rh = r4 + r5 + r6
        rl = r1 + r2 + r3

        ## full-fusion
        rh = upsample(rh, rl.shape[2:])
        sal = self.conv_s_f (torch.cat((rh,rl),dim=1))

        return sal

def initialize_weights(model):
    m = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    pretrained_dict = m.state_dict()
    all_params = {}
    for k, v in model.state_dict().items():
        if k in pretrained_dict.keys():
            v = pretrained_dict[k]
            all_params[k] = v
    model.load_state_dict(all_params,strict = False)

if __name__ == '__main__':
    img = torch.randn(1, 3, 256, 256).cuda()
    depth = torch.randn(1, 1, 256, 256).cuda()
    model = DFMNet().cuda()
    model.eval()
    time1= time.time()
    outputs = model(img,depth)
    time2 = time.time()
    torch.cuda.synchronize()
    print(1000/(time2-time1))
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(num_params)
