#!/usr/bin/env python

##  multi_instance_object_detection.py

"""
This script is meant to be run with the PurdueDrEvalMultiDataset. Each image 
in this dataset contains up to 5 object instances drawn from the three categories: 
Dr_Eval, house, and watertower.  To each image is added background clutter that 
consists of randomly generated shapes and 20% noise.
"""

from distutils.log import debug
import random
import numpy
import torch
import os, sys
from pycocotools.coco import COCO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim

seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)

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
                  path_saved_yolo_model = "/home/varun/work/courses/why2learn/saved_yolo_model_lr-4.pt",
                  momentum = 0.9,
                  learning_rate = 1e-5,
                  epochs = 20,
                  batch_size = 1,
                  classes = ['car','motorcycle','stop sign'],
                  use_gpu = True,
              )


yolo = RegionProposalGenerator.YoloLikeDetector( rpg = rpg )

## set the dataloaders
# yolo.set_dataloaders(train=True)
# yolo.set_dataloaders(test=True)

## custom dataloader
# prepare train dataloader
transform = tvt.Compose([tvt.ToTensor(),tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  
coco  = COCO('/home/varun/work/courses/why2learn/hw/annotations/instances_val2017.json')
dataserver_test = CocoDetection(transform, rpg.dataroot_test, rpg.class_labels, rpg.image_size, coco, loadDict=True, saveDict=False, mode="test")
yolo.test_dataloader = torch.utils.data.DataLoader(dataserver_test, batch_size=rpg.batch_size, shuffle=True, num_workers=16)

model = yolo.NetForYolo(skip_connections=True, depth=8) 

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
num_layers = len(list(model.parameters()))
print("\n\nThe number of layers in the model: %d\n\n" % num_layers)

# %%
# train, test, both
mode = "test"
if mode=="train" or mode=="both":
    model = yolo.run_code_for_training_multi_instance_detection(model, display_images=False)
elif mode=="test" or mode=="both":
    yolo.run_code_for_testing_multi_instance_detection(model, display_images = False)


