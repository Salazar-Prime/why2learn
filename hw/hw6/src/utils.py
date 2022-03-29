"""
Homework 6: Implement part of YOLO logic 

Author: Varun Aggarwal
Last Modified: 28 Mar 2022
"""

from pycocotools.coco import COCO
import os, sys
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import itertools

from PIL import Image
import requests
from io import BytesIO
import pickle
import torch

def normalize(data,min=0.0,max=127.0):
    return list((np.array(data) - min)/(max-min))

def unnormalize(data,min=0.0,max=127.0):
    return list(np.array(data)*(max-min)+min)

# funciton for checking if img follows the spec 
def checkImgAnn(root_path, classes, img, catIds, anns, size, coco):
    filePath = os.path.join(root_path, img['file_name'])
    if os.path.exists(filePath):
        imgActual = Image.open(filePath)
    else:
        response = requests.get(img['coco_url'])
        imgActual = Image.open(BytesIO(response.content))
        imgActual = imgActual.convert(mode='RGB')
        imgActual.save(filePath)

    # bbox scaling parameters
    w,h = imgActual.size
    wScale, hScale = size[0]/w, size[1]/h

    objPresenceCount = len(anns)
    # choose first five annotations it total is over 5
    if objPresenceCount > 5:
        anns = anns[0:5]
        objPresenceCount = 5

    # prepare bbox and label tensor
    bbox_tensor = torch.zeros(5,4, dtype=torch.uint8)
    bbox_label_tensor = torch.zeros(5, dtype=torch.uint8) + 13  # for empty object
    for i in range(objPresenceCount):
        ## normalize bbox 
        wScale, hScale = size[0]/w, size[1]/h
        bbox = anns[i]['bbox']
        bbox = [wScale*(bbox[0]),hScale*(bbox[1]),wScale*(bbox[0]+bbox[2]-1),hScale*(bbox[1]+bbox[3]-1)]
        # can improve the logic here for seperate width and height
        for j,coor in enumerate(bbox):
            if coor > size[0]-1:
                bbox[j] = size[0]-1
            elif coor < 0:
                bbox[j] = 0
        # bbox = normalize(bbox,0,123)
        # save bbox and label in tensor
        bbox_tensor[i] = torch.LongTensor(bbox)
        bbox_label_tensor[i] = catIds.index(anns[i]['category_id'])
    
    ## For valid images, return image dictionary
    imgDict = {}
    imgDict['bbox_label'] = bbox_label_tensor
    imgDict['no_of_objects'] = objPresenceCount
    imgDict['bbox'] = bbox_tensor
    imgDict['imgActual'] = imgActual.resize(size)
    return {filePath:imgDict}


# coco - instance of COCO class -> coco = COCO(jsonPath)
def downloadCOCO(root_path, classes, size=(128,128), coco=None, saveDict=False, mode="test"):
    # create a dictionary of images which can be used for training/validation
    dictImgs = {}
    # check if root image folder already exists if not then create one
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    

    # get category ids for all classes
    catIds  = coco.getCatIds(catNms=classes)
    imgIds = []

    # get images with atleast two different clases
    for data in itertools.combinations(catIds,2):
        imgIds.extend(coco.getImgIds(catIds=data))
    imgIds = list(set(imgIds))

    imgs = coco.loadImgs(imgIds)

    # Start Downloading
    print("Downloading %d images "%(len(imgIds)))
    for img in imgs:
        # get all annotations for the img
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)
        anns = coco.loadAnns(annIds)
        # anns = [x for x in anns if x['category_id'] in catIds]

        # download only if annotation are to the spec i.e. less than 5 anns
        imgDict = checkImgAnn(root_path, classes, img, catIds, anns, size, coco)
        
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
# dictImgs = downloadCOCO('/home/varun/work/courses/why2learn/hw/hw6/data',['car','motorcycle','stop sign'],(128,128),coco, True, "train")
# coco  = COCO('/home/varun/work/courses/why2learn/hw/annotations/instances_val2017.json')
# dictImgs = downloadCOCO('/home/varun/work/courses/why2learn/hw/hw6/data',['car','motorcycle','stop sign'],(128,128),coco, True, "test")
# print("The size of the dictionary is {} bytes".format(sys.getsizeof(dictImgs)))
# [x for x in temp4 if x['category_id'] in [3,13]]
# print()