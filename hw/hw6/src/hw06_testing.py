#!/usr/bin/env python

##  multi_instance_object_detection.py

"""
This script is meant to be run with the PurdueDrEvalMultiDataset. Each image 
in this dataset contains up to 5 object instances drawn from the three categories: 
Dr_Eval, house, and watertower.  To each image is added background clutter that 
consists of randomly generated shapes and 20% noise.
"""

from distutils.log import debug
import random, time
import numpy as np
import torch
import os, sys

from pycocotools.coco import COCO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torch.optim as optim
import torchvision.transforms as tvt
import torchvision.transforms.functional as F
import torchvision.utils as tutils


from PIL import Image
from PIL import ImageDraw
from PIL import ImageTk
from PIL import ImageFont

import sys,os,os.path,glob,signal
import re
import functools
import math
import random
import copy
import gzip
import pickle

if sys.version_info[0] == 3:
    import tkinter as Tkinter
    from tkinter.constants import *
else:
    import Tkinter    
    from Tkconstants import *

import matplotlib.pyplot as plt
import logging        

# seed = 0           
# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# numpy.random.seed(seed)
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmarks=False
# os.environ['PYTHONHASHSEED'] = str(seed)

# USER imports
sys.path.append("/home/varun/work/courses/why2learn/hw/RPG-2.0.6")
from RegionProposalGenerator import *        

from model import pneumaNet
from dataloader import CocoDetection
import utils


rpg = RegionProposalGenerator(
                  dataroot_train = "/home/varun/work/courses/why2learn/hw/hw6/data/",
                  dataroot_test  = "/home/varun/work/courses/why2learn/hw/hw6/data/",
                  image_size = [120,120],
                  yolo_interval = 20,
                  path_saved_yolo_model = "/home/varun/work/courses/why2learn/saved_yolo_model_lr-4_bs_71_depth16.pt",
                  momentum = 0.5,
                  learning_rate = 1e-4,
                  epochs = 50,
                  batch_size = 1,
                  classes = ['car','motorcycle','stop sign'],
                  use_gpu = True,
              )

class batchedYOLO(RegionProposalGenerator.YoloLikeDetector):
    def __init__(self, rpg):
        super().__init__(rpg)
    
    def save_yolo_model(self, acc_max, acc, model, path): 
        if acc_max != 0:
            oldSaveName = "model_" + str(self.rpg.epochs) + "_" +  str(self.rpg.learning_rate) + "_" + str(self.rpg.batch_size) + "_" + str(model.depth) +"_" + str(np.round(acc_max,2)) + "_best.pt"   
            oldSaveName = os.path.join(path,oldSaveName) 
            os.remove(oldSaveName)                        
        saveName = "model_" + str(self.rpg.epochs) + "_" +  str(self.rpg.learning_rate) + "_" + str(self.rpg.batch_size) + "_" + str(model.depth) +"_" + str(np.round(acc,2)) + "_best.pt"   
        saveName = os.path.join(path,saveName)                         
        torch.save(model, saveName) 

    def run_code_for_testing_multi_instance_detection(self, net, display_labels=False, display_images=False):        
        # net.load_state_dict(torch.load(self.rpg.path_saved_yolo_model))
        net = net.to(self.rpg.device)
        yolo_interval = self.rpg.yolo_interval
        num_yolo_cells = (self.rpg.image_size[0] // yolo_interval) * (self.rpg.image_size[1] // yolo_interval)
        num_anchor_boxes =  5    # (height/width)   1/5  1/3  1/1  3/1  5/1
        yolo_tensor = torch.zeros( self.rpg.batch_size, num_yolo_cells, num_anchor_boxes, 8 )
        with torch.no_grad():
            for iter, data in enumerate(self.test_dataloader):
                im_tensor, seg_mask_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data
                if iter % 5 == 4:
                    print("\n\n\n\nShowing output for test batch %d: " % (iter+1))
                    im_tensor   = im_tensor.to(self.rpg.device)
                    seg_mask_tensor = seg_mask_tensor.to(self.rpg.device)                 
                    bbox_tensor = bbox_tensor.to(self.rpg.device)
                    bbox_label_tensor = bbox_label_tensor.to(self.rpg.device)
                    yolo_tensor = yolo_tensor.to(self.rpg.device)

                    output = net(im_tensor)
                    predictions = output.view(self.rpg.batch_size,num_yolo_cells,num_anchor_boxes,9)
                    for ibx in range(predictions.shape[0]):                             # for each batch image
                        icx_2_best_anchor_box = {ic : None for ic in range(36)}
                        for icx in range(predictions.shape[1]):                         # for each yolo cell
                            cell_predi = predictions[ibx,icx]               
                            prev_best = 0
                            for anchor_bdx in range(cell_predi.shape[0]):               # for each anchor box
                                if cell_predi[anchor_bdx][0] > cell_predi[prev_best][0]:
                                    prev_best = anchor_bdx
                            best_anchor_box_icx = prev_best   
                            icx_2_best_anchor_box[icx] = best_anchor_box_icx
                        sorted_icx_to_box = sorted(icx_2_best_anchor_box, 
                                    key=lambda x: predictions[ibx,x,icx_2_best_anchor_box[x]][0].item(), reverse=True)
                        retained_cells = sorted_icx_to_box[:5]
                    objects_detected = []
                    pt = []
                    correct = 0
                    total_objects = 0
                    for icx in retained_cells:
                        pred_vec = predictions[ibx,icx, icx_2_best_anchor_box[icx]]
                        class_labels_predi  = pred_vec[-4:]                        
                        class_labels_probs = torch.nn.Softmax(dim=0)(class_labels_predi)
                        class_labels_probs = class_labels_probs[:-1]
                        if torch.all(class_labels_probs < 0.2): 
                            predicted_class_label = None
                        else:
                            best_predicted_class_index = (class_labels_probs == class_labels_probs.max())
                            pt.append(best_predicted_class_index[0].item())
                            best_predicted_class_index = torch.nonzero(best_predicted_class_index, as_tuple=True)
                            predicted_class_label = self.rpg.class_labels[best_predicted_class_index[0].item()]
                            objects_detected.append(predicted_class_label)
                        gt = np.array(bbox_label_tensor[ibx,:].tolist())
                        correct += np.sum(gt==pt)
                        total_objects += np.sum(gt != 13)
                        acc = 100.0*correct/total_objects
                    print("[batch image=%d]  objects found in descending probability order: " % ibx, objects_detected)
                    print("[batch image=%d]  objects present: " % ibx, bbox_label_tensor[ibx,:])

                    print("Acc: %0.2f"%(acc))
                    logger = logging.getLogger()
                    old_level = logger.level
                    logger.setLevel(100)
                    if display_images:
                        plt.figure(figsize=[15,4])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                        padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.show()
                    logger.setLevel(old_level)
    
## set the dataloaders
# yolo.set_dataloaders(train=True)
# yolo.set_dataloaders(test=True)
yolo = batchedYOLO( rpg = rpg )

## custom dataloader
# prepare train dataloader
transform = tvt.Compose([tvt.ToTensor(),tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  
# coco  = COCO('/home/varun/work/courses/why2learn/hw/annotations/instances_train2017.json')
coco = []
dataserver_test = CocoDetection(transform, rpg.dataroot_test, rpg.class_labels, rpg.image_size, coco, loadDict=True, saveDict=False, mode="test")
yolo.test_dataloader = torch.utils.data.DataLoader(dataserver_test, batch_size=rpg.batch_size, shuffle=True, num_workers=8)

# model = yolo.NetForYolo(skip_connections=True, depth=8) 
model = torch.load("/home/varun/work/courses/why2learn/hw/hw6/runs/model_50_0.0001_71_4_38.31_best.pt")

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
num_layers = len(list(model.parameters()))
print("\n\nThe number of layers in the model: %d\n\n" % num_layers)

# train, test, both
mode = "test"
if mode=="train" or mode=="both":
    model = yolo.run_code_for_training_multi_instance_detection(model, display_images=False, display_labels=True)
if mode=="test" or mode=="both":
    yolo.run_code_for_testing_multi_instance_detection(model, display_images = True)
