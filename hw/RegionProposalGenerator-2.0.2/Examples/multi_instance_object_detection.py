#!/usr/bin/env python

##  multi_instance_object_detection.py

"""
This script is meant to be run with the PurdueDrEvalMultiDataset. Each image 
in this dataset contains up to 5 object instances drawn from the three categories: 
Dr_Eval, house, and watertower.  To each image is added background clutter that 
consists of randomly generated shapes and 20% noise.
"""

import random
import numpy
import torch
import os, sys


seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


from RegionProposalGenerator import *

rpg = RegionProposalGenerator(
                  dataroot_train = "./data/Purdue_Dr_Eval_multi_dataset_train_10000/",
                  dataroot_test  = "./data/Purdue_Dr_Eval_multi_dataset_test_1000/",
#                  dataroot_train = "/home/kak/ImageDatasets/Purdue_Dr_Eval_multi_dataset_train_10000/",
#                  dataroot_test = "/home/kak/ImageDatasets/Purdue_Dr_Eval_multi_dataset_test_1000/",
                  image_size = [128,128],
                  yolo_interval = 20,
                  path_saved_yolo_model = "./saved_yolo_model",
                  momentum = 0.9,
                  learning_rate = 1e-5,
                  epochs = 20,
                  batch_size = 4,
                  classes = ('Dr_Eval','house','watertower'),
                  use_gpu = True,
              )


yolo = RegionProposalGenerator.YoloLikeDetector( rpg = rpg )

## set the dataloaders
yolo.set_dataloaders(train=True)
yolo.set_dataloaders(test=True)

model = yolo.NetForYolo(skip_connections=True, depth=8) 

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
num_layers = len(list(model.parameters()))
print("\n\nThe number of layers in the model: %d\n\n" % num_layers)

model = yolo.run_code_for_training_multi_instance_detection(model, display_images=False)

yolo.run_code_for_testing_multi_instance_detection(model, display_images = True)

