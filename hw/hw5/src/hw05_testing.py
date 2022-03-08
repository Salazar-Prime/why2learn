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
        
        def run_code_for_testing_detection_and_localization(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            correct = 0
            total = 0
            confusion_matrix = torch.zeros(len(self.dataserver_test.class_labels), 
                                           len(self.dataserver_test.class_labels))
            class_correct = [0] * len(self.dataserver_test.class_labels)
            class_total = [0] * len(self.dataserver_test.class_labels)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    images, bounding_box, labels = data['image'], data['bbox'], data['label']
                    if len(labels) != 4:
                        continue
                    labels = labels.tolist()
                    if self.dl_studio.debug_test and i % 50 == 0:
                        print("\n\n[i=%d:] Ground Truth:     " %i + ' '.join('%10s' % 
                         self.dataserver_test.class_labels[labels[j]] for j in range(self.dl_studio.batch_size)))
                    outputs = net(images)
                    outputs_label = outputs[0]
                    outputs_regression = outputs[1]
                    outputs_regression[outputs_regression < 0] = 0
                    outputs_regression[outputs_regression > 31] = 31
                    outputs_regression[torch.isnan(outputs_regression)] = 0
                    output_bb = outputs_regression.tolist()
                    _, predicted = torch.max(outputs_label.data, 1)
                    predicted = predicted.tolist()
                    if self.dl_studio.debug_test and i % 50 == 0:
                        print("[i=%d:] Predicted Labels: " %i + ' '.join('%10s' % 
                              self.dataserver_test.class_labels[predicted[j]] for j in range(self.dl_studio.batch_size)))
                        for idx in range(self.dl_studio.batch_size):
                            i1 = int(bounding_box[idx][1])
                            i2 = int(bounding_box[idx][3])
                            j1 = int(bounding_box[idx][0])
                            j2 = int(bounding_box[idx][2])
                            k1 = int(output_bb[idx][1])
                            k2 = int(output_bb[idx][3])
                            l1 = int(output_bb[idx][0])
                            l2 = int(output_bb[idx][2])
                            print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                            print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                            images[idx,0,i1:i2,j1] = 255
                            images[idx,0,i1:i2,j2] = 255
                            images[idx,0,i1,j1:j2] = 255
                            images[idx,0,i2,j1:j2] = 255
                            images[idx,2,k1:k2,l1] = 255                      
                            images[idx,2,k1:k2,l2] = 255
                            images[idx,2,k1,l1:l2] = 255
                            images[idx,2,k2,l1:l2] = 255
                        logger = logging.getLogger()
                        old_level = logger.level
                        logger.setLevel(100)
                        plt.figure(figsize=[8,3])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(images, normalize=True,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.show()
                        logger.setLevel(old_level)
                    for label,prediction in zip(labels,predicted):
                        confusion_matrix[label][prediction] += 1
                    total += len(labels)
                    correct +=  [predicted[ele] == labels[ele] for ele in range(len(predicted))].count(True)
                    comp = [predicted[ele] == labels[ele] for ele in range(len(predicted))]
                    for j in range(self.dl_studio.batch_size):
                        label = labels[j]
                        class_correct[label] += comp[j]
                        class_total[label] += 1
            print("\n")
            for j in range(len(self.dataserver_test.class_labels)):
                print('Prediction accuracy for %5s : %2d %%' % (
              self.dataserver_test.class_labels[j], 100 * class_correct[j] / class_total[j]))
            print("\n\n\nOverall accuracy of the network on the 1000 test images: %d %%" % 
                                                                   (100 * correct / float(total)))
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                "
            for j in range(len(self.dataserver_test.class_labels)):  
                                 out_str +=  "%15s" % self.dataserver_test.class_labels[j]   
            print(out_str + "\n")
            for i,label in enumerate(self.dataserver_test.class_labels):
                out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                                 for j in range(len(self.dataserver_test.class_labels))]
                out_percents = ["%.2f" % item.item() for item in out_percents]
                out_str = "%12s:  " % self.dataserver_test.class_labels[i]
                for j in range(len(self.dataserver_test.class_labels)): 
                                                       out_str +=  "%15s" % out_percents[j]
                print(out_str)




invincible = tool(
    dataroot='/home/varun/work/courses/why2learn/hw/hw5/data',
    image_size=[64, 64],
    path_saved_model="/home/varun/work/courses/why2learn/hw/hw5/saves/saved_model.pt",
    momentum=0.9,
    learning_rate=1e-4,
    epochs=10,
    batch_size=4,
    classes=['cat','train','airplane'],
    use_gpu=True,
)


detector = tool.fearInoculum(dl_studio=invincible)

transform = tvt.Compose([tvt.ToTensor(),tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  

# prepare test dataloader
coco  = COCO('/home/varun/work/courses/why2learn/hw/annotations/instances_val2017.json')
dataserver_test = CocoDetection(transform, invincible.dataroot, invincible.class_labels, invincible.image_size, coco, loadDict=True, saveDict=False, mode="test")
detector.dataserver_test = dataserver_test
test_dataloader = torch.utils.data.DataLoader(dataserver_test, batch_size=invincible.batch_size, shuffle=False, num_workers=16)
detector.test_dataloader = test_dataloader

model = pneumaNet(depth=16)

detector.run_code_for_testing_detection_and_localization(model)
