import torch
from torch.utils.data import Dataset
from PIL import Image
import os, glob
import numpy as np

class dataLoader(Dataset):
    def __init__(self,dataPath,classes,transform):
        self.dataPath = dataPath
        self.classes = classes
        self.transform = transform
        # image list and corresponding label
        self.imgList = []
        self.labelForImg()
        
    def __len__(self):
        length = 0
        for cls in self.classes:
            length += len(glob.glob(os.path.join(self.dataPath,cls,"*")))
        return length
    
    def __getitem__(self,idx):
        img_path = self.imgList[idx,1]
        label = torch.tensor(int(self.imgList[idx,0]), dtype=torch.uint8)
        label = int(self.imgList[idx,0])
        
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        
        return image, label
        
    def labelForImg(self):
        # get list of classes form datapath 
        self.classes = os.listdir(self.dataPath)
        # connect label with classes 
        for i, cls in enumerate(self.classes):
            imgs = glob.glob(os.path.join(self.dataPath,cls,'*'))
            if i==0:
                self.imgList = np.vstack((i*np.ones(len(imgs),dtype=int),imgs))
            else:
                self.imgList = np.hstack((self.imgList,np.vstack((i*np.ones(len(imgs),dtype=int),imgs))))
        self.imgList = self.imgList.T