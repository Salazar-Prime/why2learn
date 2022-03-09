#!/usr/bin/env python

##  single_instance_object_detection.py

"""
This script is meant to be run using the PurdueDrEvalDataset dataset.
Each image in this dataset contains only one meaningful object and several
background clutter shapes plus 20% random noise.

The point I wish to make through this exercise is that while the human eye may
provide an overall single-label classification for an otherwise busy image
on the basis of the "most meaningful" object found in the image, a neural network
ordinarily does not possess the ability to associate the "most meaningfulness" 
with any part of an image.  That is, commonly used neural networks will give
equal measure to all the pixels in an image in order to associate a single label
with the images.  In other words, neural networks are not capable of what
may be referred to as "cognitive discounting" in order to push into the 
background what does not appear to be relevant when examining the contents
of an image.
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


##  watch -d -n 0.5 nvidia-smi

from RegionProposalGenerator import *

rpg = RegionProposalGenerator(
                  dataroot_train = "./data/Purdue_Dr_Eval_dataset_train_10000/",
                  dataroot_test = "./data/Purdue_Dr_Eval_dataset_test_1000/",
#                  dataroot_train = "/home/kak/ImageDatasets/Purdue_Dr_Eval_dataset_train_10000/",
#                  dataroot_test = "/home/kak/ImageDatasets/Purdue_Dr_Eval_dataset_test_1000/",
                  image_size = [128,128],
                  path_saved_single_instance_detector_model = "./saved_single_instance_detector_model",
                  momentum = 0.9,
                  learning_rate = 1e-6,
                  epochs = 6,
                  batch_size = 4,
                  classes = ('Dr_Eval','house','watertower'),
                  use_gpu = True,
              )


detector = RegionProposalGenerator.SingleInstanceDetector( rpg = rpg )
detector.set_dataloaders(train=True)
detector.set_dataloaders(test=True)
model = detector.LOADnet(skip_connections=True, depth=8) 

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the single-instance detector: %d" % number_of_learnable_params)
num_layers = len(list(model.parameters()))
print("\n\nThe number of layers in the single-instance detector: %d\n\n" % num_layers)

detector.run_code_for_training_single_instance_detector(model, display_images=True)

import pymsgbox
response = pymsgbox.confirm("Finished training.  Start testing on unseen data?")
if response == "OK": 
    classifier.run_code_for_testing_single_instance_detector(model, display_images=True)
