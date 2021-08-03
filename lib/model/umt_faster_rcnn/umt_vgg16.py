# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.umt_faster_rcnn.umt_faster_rcnn import _fasterRCNN

# from model.faster_rcnn.faster_rcnn_imgandpixellevel_gradcam  import _fasterRCNN
from model.utils.config import cfg


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
    )


class netD_confidence(nn.Module):
    def __init__(self, feat_d):
        super(netD_confidence, self).__init__()
        self.fc1 = nn.Linear(feat_d, 128)
        self.bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc2(F.dropout(self.bn(self.fc1(x))))
        return x


class vgg16(_fasterRCNN):
    def __init__(
        self,
        classes,
        pretrained=False,
        class_agnostic=False,
        conf=None,
    ):
        self.model_path = cfg.VGG_PATH
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.conf = conf

        _fasterRCNN.__init__(self, classes, class_agnostic, conf)

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict(
                {k: v for k, v in state_dict.items() if k in vgg.state_dict()}
            )

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        # print(vgg.features)
        self.RCNN_base1 = nn.Sequential(*list(vgg.features._modules.values())[:14])

        self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[14:-1])
        # print(self.RCNN_base1)
        # print(self.RCNN_base2)
        feat_d = 4096
        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.RCNN_base1[layer].parameters():
                p.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

        self.RCNN_top = vgg.classifier

        self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)
        self.netD_confidence = netD_confidence(feat_d)

        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7
