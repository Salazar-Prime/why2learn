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
    def __init__(self,transform,dataPath,classes,size,coco,loadDict=False,mode="test"):
        self.dataPath = dataPath
        self.classes = classes
        self.transform = transform

        # load pre-existing dictionary 
        if loadDict and mode=="train":
            with open(os.path.join(dataPath,'dictTrain.pkl'), 'rb') as file:  
                self.imgDict = pickle.load(file)
        elif loadDict and mode=="test":
            with open(os.path.join(dataPath,'dictTest.pkl'), 'rb') as file:  
                self.imgDict = pickle.load(file)
        elif mode=="train":
            self.imgDict = utils.downloadCOCO(dataPath,classes,size,coco,True)
        elif mode=="test":
            self.imgDict = utils.downloadCOCO(dataPath,classes,size,coco,True)
        else:
            print("Something is wrong here !!!")
            sys.exit()
        
    def __len__(self):
        return len(self.imgDict)
    
    def __getitem__(self,idx):
        img = self.imgDict[idx]
        label = torch.tensor(img['classID'], dtype=torch.uint8)
        bbox = torch.tensor(img['bbox'], dtype=float)
        if self.transform:
            image = self.transform(img['imgActual'])
        
        return image, label, bbox