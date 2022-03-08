"""
Homework 5: Create custom skipblock and train/test
            modified from DLStudio

Author: Varun Aggarwal
Last Modified: 07 Mar 2022
"""

from pycocotools.coco import COCO
import os, sys
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2

from PIL import Image
import requests
from io import BytesIO
import pickle

def normalize(data,min=0.0,max=63.0):
    return list((np.array(data) - min)/(max-min))

def unnormalize(data,min=0.0,max=63.0):
    return list(np.array(data)*(max-min)+min)

# funciton for checking if img follows the spec 
def checkImgAnn(path_cls, classes, cls, img, catId, anns, size):
    filePath = os.path.join(path_cls, img['file_name'])
    if os.path.exists(filePath):
        imgActual = Image.open(filePath)
    else:
        response = requests.get(img['coco_url'])
        imgActual = Image.open(BytesIO(response.content))
        imgActual = imgActual.convert(mode='RGB')
        imgActual.save(filePath)

    ## Sepcs for a valid image
    # Spec 1: make sure class of largest object is equal to catID
    boxAreaMax = 0
    catIDmax = -1
    boxMax = []
    for ann in anns:
        area = ann['bbox'][2] * ann['bbox'][3]
        if area > boxAreaMax and ann['category_id'] in coco.getCatIds(catNms=classes):
            boxAreaMax = area
            catIDmax = ann['category_id']
            boxMax = ann['bbox']
    if catIDmax != catId[0]:
        return False

    # Spec 2: check if widht and height of largest object are at least 1/3 of image width and height
    #         and bbox size is less than the image size
    w,h = imgActual.size
    if boxMax[2] < w/3 or boxMax[3] < h/3 or boxMax[2] > w or boxMax[3] > h:
        return False

    wScale, hScale = size[0]/w, size[1]/h
    bbox = [wScale*(boxMax[0])-1,hScale*(boxMax[1])-1,wScale*(boxMax[0]+boxMax[2])-1,hScale*(boxMax[1]+boxMax[3])-1]
    # can improve the logic here
    for i,coor in enumerate(bbox):
        if coor > 63:
            bbox[i] = 63
        elif coor < 0:
            bbox[i] = 0

    ## For valid images, return image dictionary
    imgDict = {}
    imgDict['classID'] = classes.index(cls)
    imgDict['catID'] = catId
    imgDict['bbox'] = normalize(bbox)
    imgDict['imgActual'] = imgActual.resize(size)
    return {filePath:imgDict}


# coco - instance of COCO class -> coco = COCO(jsonPath)
def downloadCOCO(root_path, classes, size=(128,128), coco=None, saveDict=False, mode="test"):
    # create a dictionary of images which can be used for training/validation
    dictImgs = {}
    # check if image folder already exists if not then create one
    for cls in classes:
        path_cls = os.path.join(root_path, cls)
        if not os.path.exists(path_cls):
            os.makedirs(path_cls)
        
        # load images 
        catId  = coco.getCatIds(catNms=cls)
        imgIds = coco.getImgIds(catIds=catId)
        imgs = coco.loadImgs(imgIds)

        # Start Downloading
        print("Downloading %d images for Class %s"%(len(imgIds),cls))
        for img in imgs:
            # get all annotations for the img
            annIds = coco.getAnnIds(imgIds=img['id'])
            anns = coco.loadAnns(annIds)

            # download only if annotation are to the spec
            imgDict = checkImgAnn(path_cls, classes, cls, img, catId, anns, size)
            
            if imgDict != False:
                dictImgs.update(imgDict)
    
    # save the dictionary for faster access
    if saveDict==True and mode=="train":
        with open (os.path.join(root_path, 'dictTrain.pkl'),'wb') as file:
            pickle.dump(dictImgs, file)
    elif saveDict==True and mode=="test":
        with open (os.path.join(root_path, 'dictTest.pkl'),'wb') as file:
            pickle.dump(dictImgs, file)
    return dictImgs
           
# coco  = COCO('/home/varun/work/courses/why2learn/hw/annotations/instances_train2017.json')
# dictImgs = downloadCOCO('/home/varun/work/courses/why2learn/hw/hw5/data',['cat','train','airplane'],(64,64),coco, True, "train")
# coco  = COCO('/home/varun/work/courses/why2learn/hw/annotations/instances_val2017.json')
# dictImgs = downloadCOCO('/home/varun/work/courses/why2learn/hw/hw5/data',['cat','train','airplane'],(64,64),coco, True)
# print("The size of the dictionary is {} bytes".format(sys.getsizeof(dictImgs)))
