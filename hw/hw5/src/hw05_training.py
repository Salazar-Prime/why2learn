"""
Homework 5: Create custom skipblock 
Author: Varun Aggarwal
Last Modified: 06 Mar 2022
"""

import random
import numpy
import torch
import os, sys

# USER imports
sys.path.append("/home/varun/work/courses/why2learn/hw/DLStudio-2.1.6/")
from DLStudio import *

sys.path.append("/home/varun/work/courses/why2learn/hw/DLStudio-2.1.6/Examples")
from model import pneumaNet


"""
seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)
"""


class tool(DLStudio):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # class fearInoculum(DLStudio.DetectAndLocalize):
    #     def __init__(
    #         self,
    #         dl_studio,
    #         dataserver_train=None,
    #         dataserver_test=None,
    #         dataset_file_train=None,
    #         dataset_file_test=None,
    #     ):
    #         super().__init__(
    #             dl_studio,
    #             dataserver_train,
    #             dataserver_test,
    #             dataset_file_train,
    #             dataset_file_test,
    #         )


fearInoculum = tool(
    dataroot="/home/varun/work/courses/why2learn/hw/DLStudio-2.1.6/Examples/data/",
    image_size=[32, 32],
    path_saved_model="/home/varun/work/courses/why2learn/hw/hw5/saves/saved_model.pt",
    momentum=0.9,
    learning_rate=1e-4,
    epochs=2,
    batch_size=4,
    classes=("rectangle", "triangle", "disk", "oval", "star"),
    use_gpu=True,
)


detector = DLStudio.DetectAndLocalize(dl_studio=dls)

dataserver_train = DLStudio.DetectAndLocalize.PurdueShapes5Dataset(
    train_or_test="train",
    dl_studio=dls,
    dataset_file="PurdueShapes5-10000-train.gz",
)
dataserver_test = DLStudio.DetectAndLocalize.PurdueShapes5Dataset(
    train_or_test="test", dl_studio=dls, dataset_file="PurdueShapes5-1000-test.gz"
)
detector.dataserver_train = dataserver_train
detector.dataserver_test = dataserver_test

detector.load_PurdueShapes5_dataset(dataserver_train, dataserver_test)

# model = detector.LOADnet2(skip_connections=True, depth=8)
model = pneumaNet(depth=16)

number_of_learnable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
print(
    "\n\nThe number of learnable parameters in the model: %d"
    % number_of_learnable_params
)

num_layers = len(list(model.parameters()))
print("\nThe number of layers in the model: %d\n\n" % num_layers)


detector.run_code_for_training_with_CrossEntropy_and_MSE_Losses(model)

import pymsgbox

response = pymsgbox.confirm("Finished training.  Start testing on unseen data?")
if response == "OK":
    detector.run_code_for_testing_detection_and_localization(model)
