"""
Homework 4: Create Convolution Neural Network (CNN)  
Author: Varun Aggarwal
Last Modified: 10 Feb 2022
"""

from tkinter import X
import torch.nn as nn
import torch.nn.functional as F

class discrimination(nn.Module):
    def __init__(self):
        super().__init__()
        # self.task = task
        self.conv1 = nn.Conv2d(3, 128, 3) 
        self.conv2 = nn.Conv2d(128, 256, 3) 
        self.conv3 = nn.Conv2d(256, 512, 3) 
        self.conv4 = nn.Conv2d(512, 256, 3) 

        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)
        self.bn128 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256*2*2, 500)
        self.fc2 = nn.Linear(500, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.bn128(self.conv1(x))))
        x = self.pool(F.relu(self.bn256(self.conv2(x))))
        x = self.pool(F.relu(self.bn512(self.conv3(x))))
        x = self.pool(F.relu(self.bn256(self.conv4(x))))
        x = x.view(-1, 256*2*2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x
        ## Uncomment the next statement and see what happens to the
        ## performance of your classifier with and without padding.
        ## Note that you will have to change the first arg in the
        ## call to Linear in line (C) above and in the line (E)
        ## shown below. After you have done this experiment, see
        ## if the statement show n below can be invoked twice with
        ## and without padding. How about three times?
        
        # Changing line (E)
        # if self.task==1:
        #     x = x.view(-1, 128*31*31)
        # elif self.task==2:
        #     x = self.pool(F.relu(self.conv2(x))) ## (D)
        #     x = x.view(-1, 128*14*14)
        # else:
        #     x = self.pool(F.relu(self.conv2(x)))
        #     x = x.view(-1, 128*15*15)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # return x


class generation(nn.Module):
    def __init__(self):
        super().__init__()
        #self.task = task
        self.convTrans1 = nn.ConvTranspose2d( 100, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.convTrans2 = nn.ConvTranspose2d( 256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.convTrans3 = nn.ConvTranspose2d( 512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.convTrans4 = nn.ConvTranspose2d( 256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.convTrans5 = nn.ConvTranspose2d( 256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.convTrans6 = nn.ConvTranspose2d( 128, 3, kernel_size=4, stride=2, padding=1, bias=False)

        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)
        self.bn128 = nn.BatchNorm2d(128)

        self.tan = nn.Tanh()
        
    def forward(self, x):
        x = F.relu(self.bn256(self.convTrans1(x)))
        x = F.relu(self.bn512(self.convTrans2(x)))
        x = F.relu(self.bn256(self.convTrans3(x)))
        x = F.relu(self.bn256(self.convTrans4(x)))
        x = F.relu(self.bn128(self.convTrans5(x)))
        x =  self.tan(F.relu(self.convTrans6(x)))
        return x