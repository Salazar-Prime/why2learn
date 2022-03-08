"""
Homework 5: Create custom skipblock and train/test
            modified from DLStudio

Author: Varun Aggarwal
Last Modified: 07 Mar 2022
"""

import copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim
from pycocotools.coco import COCO
import os, sys, logging
import matplotlib.pyplot as plt
# USER imports
sys.path.append("/home/varun/work/courses/why2learn/hw/DLStudio-2.1.6/")
from DLStudio import *

sys.path.append("/home/varun/work/courses/why2learn/hw/DLStudio-2.1.6/Examples")
from model import pneumaNet

from dataloader import CocoDetection
import utils

class tool(DLStudio):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    class fearInoculum(DLStudio.DetectAndLocalize):
        def __init__(self, dl_studio, dataserver_train=None,dataserver_test=None,dataset_file_train=None,dataset_file_test=None,):
            super().__init__(dl_studio, dataserver_train,dataserver_test,dataset_file_train,dataset_file_test,)
            self.dl_studio = dl_studio
        
        def run_code_for_training_with_CrossEntropy_and_MSE_Losses(self, net):
            # create files for saving trainng logs
            filename_for_out1 = "performance_numbers_" + str(self.dl_studio.epochs) + "label.txt"
            filename_for_out2 = "performance_numbers_" + str(self.dl_studio.epochs) + "regres.txt"
            FILE1 = open(filename_for_out1, 'w')
            FILE2 = open(filename_for_out2, 'w')

            # copy the model
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)

            # setup criterions for backprop
            criterion1 = nn.CrossEntropyLoss()
            criterion2 = nn.MSELoss()
            optimizer = optim.SGD(net.parameters(), lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            
            # Start training 
            print("\n\nStarting training loop...\n\n")
            start_time = time.perf_counter()
            labeling_loss_tally = []   
            regression_loss_tally = [] 
            elapsed_time = 0.0   
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss_labeling = 0.0
                running_loss_regression = 0.0       
                for i, data in enumerate(self.train_dataloader):
                    inputs, bbox_gt, labels = data['image'], data['bbox'], data['label']
                    if i % 500 == 499:
                        current_time = time.perf_counter()
                        elapsed_time = current_time - start_time
                        print("\n\n\n[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]      Ground Truth:     " % 
                                 (epoch+1, self.dl_studio.epochs, i+1, elapsed_time) 
                               + ' '.join('%10s' % self.dataserver_train.class_labels[labels[j].item()] 
                                                                for j in range(self.dl_studio.batch_size)))
                    inputs = inputs.to(self.dl_studio.device)
                    labels = labels.to(self.dl_studio.device)
                    bbox_gt = bbox_gt.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    if self.debug:
                        self.dl_studio.display_tensor_as_image(
                          torchvision.utils.make_grid(inputs.cpu(), nrow=4, normalize=True, padding=2, pad_value=10))
                    outputs = net(inputs)
                    outputs_label = outputs[0]
                    bbox_pred = outputs[1]
                    if i % 500 == 499:
                        inputs_copy = inputs.detach().clone()
                        inputs_copy = inputs_copy.cpu()
                        bbox_pc = bbox_pred.detach().clone()
                        bbox_pc_copy = bbox_pred.detach().clone()
                        bbox_pc[bbox_pc<0] = 0
                        bbox_pc[bbox_pc>1] = 1
                        bbox_pc[torch.isnan(bbox_pc)] = 0
                        _, predicted = torch.max(outputs_label.data, 1)
                        print("[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]  Predicted Labels:     " % (epoch+1, self.dl_studio.epochs, i+1, elapsed_time)  + ' '.join('%10s' % self.dataserver_train.class_labels[predicted[j].item()] for j in range(self.dl_studio.batch_size)))
                        for idx in range(self.dl_studio.batch_size):
                            bbox_gt_copy = bbox_gt.detach().clone()
                            bbox_gt_copy = bbox_gt_copy.cpu()
                            bbox_pc_copy = bbox_pc_copy.cpu()
                            i1 = bbox_gt_copy[idx][1]
                            i2 = bbox_gt_copy[idx][3]
                            j1 = bbox_gt_copy[idx][0]
                            j2 = bbox_gt_copy[idx][2]
                            k1 = bbox_pc_copy[idx][1]
                            k2 = bbox_pc_copy[idx][3]
                            l1 = bbox_pc_copy[idx][0]
                            l2 = bbox_pc_copy[idx][2]
                            [j1,i1,j2,i2] = [int(x) for x in utils.unnormalize([j1,i1,j2,i2])]
                            [l1,k1,l2,k2] = [int(x) for x in utils.unnormalize([l1,k1,l2,k2])]
                            print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                            print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                            try:
                                inputs_copy[idx,0,i1:i2,j1] = 255
                                inputs_copy[idx,0,i1:i2,j2] = 255
                                inputs_copy[idx,0,i1,j1:j2] = 255
                                inputs_copy[idx,0,i2,j1:j2] = 255
                                inputs_copy[idx,2,k1:k2,l1] = 255                      
                                inputs_copy[idx,2,k1:k2,l2] = 255
                                inputs_copy[idx,2,k1,l1:l2] = 255
                                inputs_copy[idx,2,k2,l1:l2] = 255
                            except:
                                print("index out of bound, Skipping")
                    loss_labeling = criterion1(outputs_label, labels)
                    loss_labeling.backward(retain_graph=True)        
                    loss_regression = criterion2(bbox_pred, bbox_gt)
                    loss_regression.backward()
                    optimizer.step()
                    running_loss_labeling += loss_labeling.item()    
                    running_loss_regression += loss_regression.item()                
                    if i % 500 == 499:    
                        avg_loss_labeling = running_loss_labeling / float(500)
                        avg_loss_regression = running_loss_regression / float(500)
                        labeling_loss_tally.append(avg_loss_labeling)  
                        regression_loss_tally.append(avg_loss_regression)    
                        print("[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]       loss_labelling %.3f        loss_regression: %.3f " %  (epoch+1, self.dl_studio.epochs, i+1, elapsed_time, avg_loss_labeling, avg_loss_regression))
                        FILE1.write("%.3f\n" % avg_loss_labeling)
                        FILE1.flush()
                        FILE2.write("%.3f\n" % avg_loss_regression)
                        FILE2.flush()
                        running_loss_labeling = 0.0
                        running_loss_regression = 0.0
                    if i%500==499 and epoch == self.dl_studio.epochs-1:
                        logger = logging.getLogger()
                        old_level = logger.level
                        logger.setLevel(100)
                        plt.figure(figsize=[8,3])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(inputs_copy, normalize=False,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.show()
                        logger.setLevel(old_level)
            print("\nFinished Training\n")
            self.save_model(net)
            plt.figure(figsize=(10,5))
            plt.title("Labeling Loss vs. Iterations")
            plt.plot(labeling_loss_tally)
            plt.xlabel("iterations")
            plt.ylabel("labeling loss")
            plt.legend()
            plt.savefig("labeling_loss.png")
            plt.show()
            plt.title("regression Loss vs. Iterations")
            plt.plot(regression_loss_tally)
            plt.xlabel("iterations")
            plt.ylabel("regression loss")
            plt.legend()
            plt.savefig("regression_loss.png")
            plt.show()




invincible = tool(
    dataroot='/home/varun/work/courses/why2learn/hw/hw5/data',
    image_size=[64, 64],
    path_saved_model="/home/varun/work/courses/why2learn/hw/hw5/saves/saved_model.pt",
    momentum=0.9,
    learning_rate=1e-4,
    epochs=2,
    batch_size=4,
    classes=['cat','train','airplane'],
    use_gpu=True,
)


detector = tool.fearInoculum(dl_studio=invincible)
transform = tvt.Compose([tvt.ToTensor(),tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  

# prepare train dataloader
coco  = COCO('/home/varun/work/courses/why2learn/hw/annotations/instances_train2017.json')
dataserver_train = CocoDetection(transform, invincible.dataroot, invincible.class_labels, invincible.image_size, coco, loadDict=True, saveDict=False, mode="train")
detector.dataserver_train = dataserver_train
train_dataloader = torch.utils.data.DataLoader(dataserver_train, batch_size=invincible.batch_size, shuffle=True, num_workers=16)
detector.train_dataloader =train_dataloader

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