import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict

import os
import sys
sys.path.append('../../')
import graphs.models.resnet as resnet
from functools import partial

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, model):
        return getattr(self.module, self.prefix + model)


class BaseModel(nn.Module):
    def __init__(self, configs):
        super(BaseModel, self).__init__()

        self.configs = configs

        try:
            Backbone = getattr(resnet, 'resnet' + str(self.configs.model_depth))
        except:
            raise Exception('No this model settings! Please Check!')

        small = Backbone(shortcut_type='B', 
            num_classes=self.configs.num_classes)
        if self.configs.finetune:
            small.load_pretrain(self.configs.pretrain)
        small = nn.Sequential(*(list(small.children())[:-1]))
        self.add_module('fext_small', small)

        middle = Backbone(shortcut_type='B', 
            num_classes=self.configs.num_classes)
        if self.configs.finetune:
            middle.load_pretrain(self.configs.pretrain)
        middle = nn.Sequential(*(list(middle.children())[:-1]))
        self.add_module('fext_middle', middle)

        large = Backbone(shortcut_type='B', 
            num_classes=self.configs.num_classes)
        if self.configs.finetune:
            large.load_pretrain(self.configs.pretrain)
        large = nn.Sequential(*(list(large.children())[:-1]))
        self.add_module('fext_large', large)

        self.backbone = AttrProxy(self, 'fext_')

    def load_pretrain(self, model):
        model_path = self.configs.pretrain
        curmodel_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if 'layer' not in k:
                continue
            name = k[7:] # remove 'module'
            new_state_dict[name] = v
        new_state_dict = {k: v for k, v in new_state_dict.items() if k in curmodel_dict}
        curmodel_dict.update(new_state_dict)
        model.load_state_dict(curmodel_dict)
        return model


    def base_forward(self, x):
        s = x[:, 0:1, ...]
        s = self.backbone['small'](s)

        m = x[:, 1:2, ...]
        m = self.backbone['middle'](m)

        l = x[:, 2:3, ...]
        l = self.backbone['large'](l)

        return s, m, l



class MultiScaleParallelNet(BaseModel):
    def __init__(self, configs):
        super(MultiScaleParallelNet, self).__init__(configs)
        if self.configs.model_depth < 50:
            expansion = 1
        else:
            expansion = 4
        final = 512

        self.fus_conv = nn.Sequential(
            nn.Linear(final*expansion*3, final*expansion), 
            nn.BatchNorm1d(final*expansion), 
            nn.ReLU(inplace=True) 
        )

        self.fc = nn.Linear(final * expansion, self.configs.num_classes)


    def forward(self, x):
        s, m, l = self.base_forward(x)

        out = torch.cat((s, m, l), dim=1).view(x.size(0), -1)
        out = self.fus_conv(out)

        out = self.fc(out)

        return out

