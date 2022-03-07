import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim

import os, sys

# USER imports
sys.path.append("/home/varun/work/courses/why2learn/hw/DLStudio-2.1.6/")
from DLStudio import *

sys.path.append("/home/varun/work/courses/why2learn/hw/DLStudio-2.1.6/Examples")
from model import pneumaNet

def run_code_for_training_with_CrossEntropy_and_MSE_Losses(self, net):
    print("\nStart Training\n")
    loss_record_labeling = []
    loss_record_regression = []

    if net.skip_connections == True:
        filename_for_out1 = "training_loss_" + str(self.dl_studio.epochs) + "_epochs_with_skip_connections_label.txt"
        filename_for_out2 = "training_loss_" + str(self.dl_studio.epochs) + "_epochs_with_skip_connections_regres.txt"
    else:
        filename_for_out1 = "training_loss_" + str(self.dl_studio.epochs) +"_epochs_without_skip_connections_label.txt"
        filename_for_out2 = "training_loss_" +str(self.dl_studio.epochs) +"_epochs_without_skip_connections_regres.txt"
    
    path_to_save = './Train Results/'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    FILE1 = open(path_to_save + filename_for_out1, 'w')
    FILE2 = open(path_to_save + filename_for_out2, 'w')
    net = copy.deepcopy(net)
    net = net.to(self.dl_studio.device)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.SGD([
        {'params':
        net.classifier_params, 'lr':
        2 * 1e-5}
        ],
        lr=self.dl_studio.learning_rate,
        momentum=self.dl_studio.momentum)
    for epoch in range(self.dl_studio.epochs):

        running_loss_labeling = 0.0
        running_loss_regression = 0.0
        for i, data in enumerate(self.train_dataloader):

            inputs, bbox_gt, labels = data['image'],
            data['bbox'], data['label']
            if self.dl_studio.debug_train and i % 500 == 499:
                print("\n\n[epoch=%d iter=%d:] Ground Truth: " % (epoch+1, i+1) + ' '.join('%15s' %
                self.dataserver_train.class_labels[labels[j].item()] for j in range(self.dl_studio.batch_size)))

            inputs = inputs.to(self.dl_studio.device)
            labels = labels.to(self.dl_studio.device)
            bbox_gt = bbox_gt.to(self.dl_studio.device)
            optimizer.zero_grad()
            if self.debug:
                self.dl_studio.display_tensor_as_image(torchvision.utils.make_grid(inputs.cpu(), nrow=4,normalize=True, padding=2,pad_value=10))
            outputs = net(inputs)
            outputs_label = outputs[0]
            bbox_pred = outputs[1]
            if self.dl_studio.debug_train and i % 500 == 499:
                inputs_copy = inputs.detach().clone()
                inputs_copy = inputs_copy.cpu()
                bbox_pc = bbox_pred.detach().clone()
                bbox_pc[bbox_pc<0] = 0
                bbox_pc[bbox_pc>63] = 63
                bbox_pc[torch.isnan(bbox_pc)] = 0
                _, predicted = torch.max(outputs_label.data, 1)
                print("[epoch=%d iter=%d:] Predicted Labels: " % (epoch+1, i+1) + ' '.join('%15s' % self.dataserver_train.class_labels [predicted[j].item()] for j in range(len(labels))))
                for idx in range(len(labels)):
                    i1 = int(bbox_gt[idx][1])
                    i2 = int(bbox_gt[idx][3])
                    j1 = int(bbox_gt[idx][0])
                    j2 = int(bbox_gt[idx][2])
                    k1 = int(bbox_pc[idx][1])
                    k2 = int(bbox_pc[idx][3])
                    l1 = int(bbox_pc[idx][0])
                    l2 = int(bbox_pc[idx][2])
                    print("gt_bb: [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                    print("pred_bb:[%d,%d,%d,%d]"%(l1,k1,l2,k2))

                    # circle the ground truth in red
                    inputs_copy[idx,0,i1:i2+1,j1] = 255
                    inputs_copy[idx,0,i1:i2+1,j2] = 255
                    inputs_copy[idx,0,i1,j1:j2+1] = 255
                    inputs_copy[idx,0,i2,j1:j2+1] = 255
                    inputs_copy[idx,1,i1:i2+1,j1] = 0
                    inputs_copy[idx,1,i1:i2+1,j2] = 0
                    inputs_copy[idx,1,i1,j1:j2+1] = 0
                    inputs_copy[idx,1,i2,j1:j2+1] = 0
                    inputs_copy[idx,2,i1:i2+1,j1] = 0
                    inputs_copy[idx,2,i1:i2+1,j2] = 0
                    inputs_copy[idx,2,i1,j1:j2+1] = 0
                    inputs_copy[idx,2,i2,j1:j2+1] = 0

                    # circle the predicted bbox in blue
                    inputs_copy[idx,2,k1:k2+1,l1] = 255
                    inputs_copy[idx,2,k1:k2+1,l2] = 255
                    inputs_copy[idx,2,k1,l1:l2+1] = 255
                    inputs_copy[idx,2,k2,l1:l2+1] = 255
                    inputs_copy[idx,0,k1:k2+1,l1] = 0
                    inputs_copy[idx,0,k1:k2+1,l2] = 0
                    inputs_copy[idx,0,k1,l1:l2+1] = 0
                    inputs_copy[idx,0,k2,l1:l2+1] = 0
                    inputs_copy[idx,1,k1:k2+1,l1] = 0
                    inputs_copy[idx,1,k1:k2+1,l2] = 0
                    inputs_copy[idx,1,k1,l1:l2+1] = 0
                    inputs_copy[idx,1,k2,l1:l2+1] = 0
            
            loss_labeling = criterion1(outputs_label,labels)
            loss_labeling.backward(retain_graph=True)

            loss_regression = criterion2(bbox_pred,bbox_gt)
            loss_regression.backward()
            optimizer.step()
            running_loss_labeling += loss_labeling.item()

            running_loss_regression += loss_regression.item()
            if i % 500 == 499:
                avg_loss_labeling = running_loss_labeling / float(500)
                avg_loss_regression = running_loss_regression / float(500)
                loss_record_labeling.append(avg_loss_labeling)
                loss_record_regression.append(avg_loss_regression)
                print("\n[epoch:%d, iteration:%5d]loss_labeling: %.3f loss_regression:%.3f " % (epoch + 1, i + 1,avg_loss_labeling, avg_loss_regression))
                FILE1.write("%.3f\n" % avg_loss_labeling)
                FILE1.flush()
                FILE2.write("%.3f\n" % avg_loss_regression)
                FILE2.flush()
                running_loss_labeling = 0.0
                running_loss_regression = 0.0
            if self.dl_studio.debug_train and i%500==499:
                if net.skip_connections == True:
                    self.dl_studio.display_tensor_as_image(torchvision.utils.make_grid(inputs_copy, normalize=True),"Training_Results_With_Skip_Connections_[Epoch=%d,_Iter=%d]" %(epoch+1, i+1))
                else:
                    self.dl_studio.display_tensor_as_image(torchvision.utils.make_grid(inputs_copy, normalize=True),"Training_Results_Without_Skip_Connections_[Epoch=%d,_Iter=%d]" % (epoch+1, i+1))

    print("\nFinished Training\n")
    self.save_model(net)
    return loss_record_labeling, loss_record_regression

