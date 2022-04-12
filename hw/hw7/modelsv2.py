"""
Homework 7: Create GAN network  
Author: Varun Aggarwal
Last Modified: 11 Apr 2022
Inspired from FCC_GAN
"""

from tkinter import X
import torch
import torch.nn as nn
import torch.nn.functional as F

class discrimination(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32,  kernel_size=4, stride=2, padding=1) 
        self.conv2 = nn.Conv2d(32, 64,  kernel_size=4, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) 
        
        self.bn32 = nn.InstanceNorm2d(32, affine=True)
        self.bn64 = nn.InstanceNorm2d(64, affine=True)
        self.bn128 = nn.InstanceNorm2d(128, affine=True)
        self.bn256 = nn.InstanceNorm2d(256, affine=True)

        self.pool = nn.AvgPool2d(2, 2)
        self.sig = nn.Sigmoid()

        self.fc1 = nn.Linear(4*4*256, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn32(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn64(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn128(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn256(self.conv4(x)), 0.2)
        x = x.view(-1, 4*4*256)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.sig(x)
        # x = x.mean(0)       
        # x = x.view(1)
        return x


class generation(nn.Module):
    def __init__(self):
        super().__init__()
        #self.task = task
        self.convTrans1 = nn.ConvTranspose2d( 256, 128, kernel_size=4, stride=2, padding=1)
        self.convTrans2 = nn.ConvTranspose2d( 128, 64, kernel_size=4, stride=2, padding=1)
        self.convTrans3 = nn.ConvTranspose2d( 64, 32, kernel_size=4, stride=2, padding=1)
        self.convTrans4 = nn.ConvTranspose2d( 32, 3, kernel_size=4, stride=2, padding=1)

        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)

        self.tan = nn.Tanh()

        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 512)
        self.fc3 = nn.Linear(512, 4096)

        
    def forward(self, x):
        # print(x.shape)
        x = x.view(-1, 100)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.reshape(x, (x.shape[0],256,4,4))
        x = F.relu(self.bn128(self.convTrans1(x)))
        x = F.relu(self.bn64(self.convTrans2(x)))
        x = F.relu(self.bn32(self.convTrans3(x)))
        x = self.convTrans4(x)
        x = self.tan(x)

        return x
