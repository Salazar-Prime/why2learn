"""
Homework 7: Create GAN network  
Author: Varun Aggarwal
Last Modified: 11 Apr 2022
Modifed from DLStudioV2.1.6
"""

import random
import numpy
import torch
import os, sys

# # """
seed = 12134          
# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# numpy.random.seed(seed)
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmarks=False
# os.environ['PYTHONHASHSEED'] = str(seed)
# # """

from modelsv2 import discrimination, generation
from modelsv4 import discrimination, generation
##  watch -d -n 0.5 nvidia-smi

sys.path.append("/home/varun/work/courses/why2learn/hw/DLStudio-2.1.6")
from DLStudio import *
from AdversarialLearning import *

import sys

dls = DLStudio(                                                                                       
                  dataroot = "/home/varun/work/courses/why2learn/hw/hw7/data/train", 
                  image_size = [64,64],                                                               
                  path_saved_model = "./runs/saved_model", 
                  learning_rate = 5e-5,      ## <==  try smaller value if mode collapse
                  epochs = 500,
                  batch_size = 64,                                                                     
                  use_gpu = True,                                                                     
              )           

adversarial = AdversarialLearning(
                  dlstudio = dls,
                  ngpu = 1,    
                  latent_vector_size = 100,
                  beta1 = 0.5,  ## for the Adam optimizer
                  clipping_threshold = 0.005,              # remove this wgan             
              )

dcgan =   AdversarialLearning.DataModeling( dlstudio = dls, adversarial = adversarial )


# discriminator =  dcgan.DiscriminatorDG1()
# generator =  dcgan.GeneratorDG1()

discriminator =  discrimination()
generator =  generation()

num_learnable_params_disc = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the Discriminator: %d\n" % num_learnable_params_disc)
num_learnable_params_gen = sum(p.numel() for p in generator.parameters() if p.requires_grad)
print("\nThe number of learnable parameters in the Generator: %d\n" % num_learnable_params_gen)
num_layers_disc = len(list(discriminator.parameters()))
print("\nThe number of layers in the discriminator: %d\n" % num_layers_disc)
num_layers_gen = len(list(generator.parameters()))
print("\nThe number of layers in the generator: %d\n\n" % num_layers_gen)

dcgan.set_dataloader()

# dcgan.show_sample_images_from_dataset(dls)

dcgan.run_wgan_code(dls, adversarial, critic=discriminator, generator=generator, results_dir="/home/varun/work/courses/why2learn/hw/hw7/runsv4/results_DG1_face_v2-"+str(seed)+"-"+str(dls.learning_rate))

