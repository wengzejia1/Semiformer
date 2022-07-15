import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_
import timm

class myResnet50(nn.Module):

    def __init__(self, num_classes=1000,):
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)

    def forward(self, x, ret_feat=False):
        output = self.model(x)
        if ret_feat:
            return {'output':output, 'feat':None}
        else:
            return output



class myResnet101(nn.Module):
    def __init__(self, num_classes=1000, **kwargs):
        super().__init__()
        self.model = timm.create_model('resnet101', pretrained=False, num_classes=num_classes)

    def forward(self, x, ret_feat=False):
        output = self.model(x)
        if ret_feat:
            return {'output':output, 'feat':None}
        else:
            return output


class myResnet152(nn.Module):
    def __init__(self, num_classes=1000, **kwargs):
        super().__init__()
        self.model = timm.create_model('resnet152', pretrained=False, num_classes=num_classes)

    def forward(self, x, ret_feat=False):
        output = self.model(x)
        if ret_feat:
            return {'output':output, 'feat':None}
        else:
            return output






