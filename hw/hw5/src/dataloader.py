"""
Homework 5: Create custom skipblock and train/test
            modified from DLStudio

Author: Varun Aggarwal
Last Modified: 07 Mar 2022
"""

from pycocotools.coco import COCO
import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2

import torch
from torch.utils.data import Dataset
from PIL import Image
import os, glob, sys
import numpy as np
import utils, pickle

class CocoDetection(Dataset):
    def __init__(self,transform,dataPath,classes,size,coco,loadDict=False, saveDict=False,mode="test"):
        self.dataPath = dataPath
        self.class_labels = classes
        self.transform = transform

        # load pre-existing dictionary 
        if loadDict and mode=="train":
            with open(os.path.join(dataPath,'dictTrain.pkl'), 'rb') as file:  
                self.imgDict = pickle.load(file)
        elif loadDict and mode=="test":
            with open(os.path.join(dataPath,'dictTest.pkl'), 'rb') as file:  
                self.imgDict = pickle.load(file)
        elif mode=="train":
            self.imgDict = utils.downloadCOCO(dataPath,classes,size,coco,saveDict)
        elif mode=="test":
            self.imgDict = utils.downloadCOCO(dataPath,classes,size,coco,saveDict)
        else:
            print("Something is wrong here !!!")
            sys.exit()
        self.imgDict = list(self.imgDict.values())
        
    def __len__(self):
        return len(self.imgDict)
    
    def __getitem__(self,idx):
        img = self.imgDict[idx]
        label = torch.tensor(img['classID'], dtype=torch.long)
        bbox = torch.tensor(img['bbox'], dtype=torch.float)
        if self.transform:
            image = self.transform(img['imgActual'])
        
        return {'image':image, 'label':label, 'bbox':bbox}