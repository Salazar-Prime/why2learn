"""
Homework 5: Create custom skipblock and train/test
            modified from DLStudio

Author: Varun Aggarwal
Last Modified: 07 Mar 2022
"""

import torch
import torch.nn as nn
import sys

sys.path.append("/home/varun/work/courses/why2learn/hw/DLStudio-2.1.6/")
from DLStudio import *


class SkipBlock(nn.Module):
    """
    Implementation of SkipBlock
    Inspired from Inception and  resnet
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # self.downsample = downsample
        # self.skip_connections = skip_connections
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.convo1x1 = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=1)
        self.convo3x3 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.maxPool3x3 = nn.MaxPool2d(3, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.reLU = nn.functional.relu

    def forward(self, x):
        identity = x
        # first
        out1 = self.convo3x3(x)
        out1 = self.convo3x3(out1)
        out1 = self.convo3x3(out1)
        # second
        out2 = self.maxPool3x3(x)
        out2 = self.convo1x1(out2)
        # add first and second
        out = out1 + out2
        out = self.bn(out)
        out = self.reLU(out)
        # skip connection
        out += identity
        return out


class pneumaNet(nn.Module):
    def __init__(self, depth=8):
        super().__init__()
        self.depth = depth
        # for classification
        self.sB128x128 = SkipBlock(128, 128)
        self.sB64x64 = SkipBlock(64, 64)

        self.convInx128 = nn.Conv2d(3, 128, 3, padding=1)
        self.conv128x64 = nn.Conv2d(128, 64, 3, padding=1)

        self.pool2x2 = nn.MaxPool2d(2, 2)
        self.reLU = nn.functional.relu
        self.fc1 = nn.Linear(65536, 1000)
        self.fc2 = nn.Linear(1000, 5)

        # for regression
        self.conv_seqn = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc_seqn = nn.Sequential(
            nn.Linear(65536, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(
                512, 4
            ),  ## output for the 4 coords (x_min,y_min,x_max,y_max) of BBox
        )

    def forward(self, x):
        # the neck
        x = self.pool2x2(self.reLU(self.convInx128(x)))
        x1 = x.clone()

        ## network
        # four blocks
        for i in range(self.depth // 2):
            x1 = self.sB128x128(x1)
        # downsample
        x1 = self.conv128x64(x1)
        # four more blocks
        for i in range(self.depth // 2):
            x1 = self.sB64x64(x1)
        # head
        x1 = x1.view(-1, 65536)
        x1 = self.reLU(self.fc1(x1))
        x1 = self.fc2(x1)

        ## The Bounding Box regression
        x2 = self.conv_seqn(x)
        # flatten
        x2 = x2.view(x.size(0), -1)
        x2 = self.fc_seqn(x2)
        return x1, x2
