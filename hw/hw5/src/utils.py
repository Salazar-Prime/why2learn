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

# def valid_imgae_ids(annFile):
#      coco = COCO(annFile)
#      class_list = ['bus','car']
#      coco_labels_inverse = {}


#         # get img path 
#         catIds = coco.getCatIds(catNms=class_list)
#         categories = coco.loadCats(catIds)
#         categories.sort(key=lambda x: x['id'])

#         # cocoInverse
#         for idx ,in_class in enumerate(class_list):
#             for c in categories:
#                 if c['name'] == in_class:
#                     coco_labels_inverse[c['id']] = idx
        
#         # loading annotations
#         scale = 4
#         #Retrieve Image list
#         imgIds = coco.getImgIds(catIds=catIds )

#         for data_img in all_images:
#             I = io.imread(data_img['coco_url'])
#         print(I.shape)
#         I = resize(I, (I.shape[0] // scale, I.shape[1] // scale), preserve_range=True)
#         print(I.shape)
#         if len(I.shape) == 2:
#             I = skimage.color.gray2rgb(I)

#         annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds ,iscrowd=False)
#         anns = coco.loadAnns(annIds)
#         fig , ax = plt.subplots(1,1)
#         image = np.uint8(I)



# jsonRootPath = '/home/varun/work/courses/why2learn/hw/annotations/'
# # Input
# input_json = os.path.join(jsonRootPath, 'instances_train2017.json')
# class_list = ['bus','car']

# ###########################
# #Mapping from COCO label to Class indices
# coco_labels_inverse = {}
# coco = COCO(input_json)
# catIds = coco.getCatIds(catNms=class_list)
# categories = coco.loadCats(catIds)
# categories.sort(key=lambda x: x['id'])
# print(categories)


#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
#         """
#         coco = self.coco
#         img_id = self.ids[index]
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         target = coco.loadAnns(ann_ids)

#         path = coco.loadImgs(img_id)[0]['file_name']

#         img = Image.open(os.path.join(self.root, path)).convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target
##############################################

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

    ## For valid images, return image dictionary
    imgDict = {}
    wScale, hScale = size[0]/w, size[1]/h
    imgDict['classID'] = classes.index(cls)
    imgDict['catID'] = catId
    imgDict['bbox'] = [wScale*boxMax[0],hScale*boxMax[1],wScale*(boxMax[0]+boxMax[2]),hScale*(boxMax[1]+boxMax[3])]
    imgDict['imgActual'] = imgActual.resize(size)
    return {filePath:imgDict}


# coco - instance of COCO class -> coco = COCO(jsonPath)
def downloadCOCO(root_path, classes, size=(128,128), coco=None, save=False):
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
    if save==True:
        with open (os.path.join(root_path, 'dictTrain.pkl'),'wb') as file:
            pickle.dump(dictImgs, file)
    return dictImgs
           


# coco  = COCO('/home/varun/work/courses/why2learn/hw/annotations/instances_train2017.json')
# dictImgs = downloadCOCO('/home/varun/work/courses/why2learn/hw/hw5/data',['cat','train','airplane'],(64,64),coco, True)
# print("The size of the dictionary is {} bytes".format(sys.getsizeof(dictImgs)))
