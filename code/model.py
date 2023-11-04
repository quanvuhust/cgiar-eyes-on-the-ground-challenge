import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np

class Net(nn.Module):
    def __init__(self, back_bone, device_id):
        super().__init__()
        self.model = timm.create_model(back_bone, num_classes=1, pretrained=True, in_chans=3)

        self.IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device_id)
        self.IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225]).to(device_id)

    def forward(self, x):
        x = x.transpose(1, 2).transpose(1, 3).contiguous()
        x = x/255.0
        x = (x - self.IMAGENET_DEFAULT_MEAN[None,:, None, None])/self.IMAGENET_DEFAULT_STD[None,:, None, None]
        x = self.model(x)

        return x


if __name__ == '__main__':
    m = Net("coat_lite_medium_384.in1k", 3, "cuda:0")
    m.cuda()
    x = torch.rand((1, 448, 448, 3)).cuda()
    m(x)
