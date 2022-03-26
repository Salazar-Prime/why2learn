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
                  learning_rate = 1e-4,
                  epochs = 20,
                  batch_size = 1,
                  classes = ['car','motorcycle','stop sign'],
                  use_gpu = True,
              )

class batchedYOLO(RegionProposalGenerator.YoloLikeDetector):
    def __init__(self, rpg):
        super().__init__(rpg)
    def run_code_for_training_multi_instance_detection(self, net, display_labels=False, display_images=False):
        yolo_debug = False
        filename_for_out1 = "performance_numbers_" + str(self.rpg.epochs) + "label.txt"                                 
        FILE1 = open(filename_for_out1, 'w')                                                                               
        net = net.to(self.rpg.device)                                                                                  
        criterion1 = nn.BCELoss()                    # For the first element of the 8 element yolo vector              ## (3)
        criterion2 = nn.MSELoss()                    # For the regiression elements (indexed 2,3,4,5) of yolo vector   ## (4)
        criterion3 = nn.CrossEntropyLoss()           # For the last three elements of the 8 element yolo vector        ## (5)
        print("\n\nLearning Rate: ", self.rpg.learning_rate)
        optimizer = optim.SGD(net.parameters(), lr=self.rpg.learning_rate, momentum=self.rpg.momentum)                 ## (6)
        print("\n\nStarting training loop...\n\n")
        start_time = time.perf_counter()
        Loss_tally = []
        elapsed_time = 0.0
        yolo_interval = self.rpg.yolo_interval                                                                         ## (7)
        num_yolo_cells = (self.rpg.image_size[0] // yolo_interval) * (self.rpg.image_size[1] // yolo_interval)         ## (8)
        num_anchor_boxes =  5    # (height/width)   1/5  1/3  1/1  3/1  5/1                                            ## (9)
        max_obj_num  = 5                                                                                               ## (10)
        ## The 8 in the following is the size of the yolo_vector for each anchor-box in a given cell.  The 8 elements 
        ## are: [obj_present, bx, by, bh, bw, c1, c2, c3] where bx and by are the delta diffs between the centers
        ## of the yolo cell and the center of the object bounding box in terms of a unit for the cell width and cell 
        ## height.  bh and bw are the height and the width of object bounding box in terms of the cell height and width.
        for epoch in range(self.rpg.epochs):                                                                           ## (11)
            print("")
            running_loss = 0.0                                                                                         ## (12)
            for iter, data in enumerate(self.train_dataloader):   
                if yolo_debug:
                    print("\n\n\n======================================= iteration: %d ========================================\n" % iter)
                yolo_tensor = torch.zeros( self.rpg.batch_size, num_yolo_cells, num_anchor_boxes, 8 )                  ## (13)
                im_tensor, seg_mask_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data                ## (14)
                im_tensor   = im_tensor.to(self.rpg.device)                                                            ## (15)
                seg_mask_tensor = seg_mask_tensor.to(self.rpg.device)                 
                bbox_tensor = bbox_tensor.to(self.rpg.device)
                bbox_label_tensor = bbox_label_tensor.to(self.rpg.device)
                yolo_tensor = yolo_tensor.to(self.rpg.device)
                if yolo_debug:
                    logger = logging.getLogger()
                    old_level = logger.level
                    logger.setLevel(100)
                    plt.figure(figsize=[15,4])
                    plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor,normalize=True,padding=3,pad_value=255).cpu(), (1,2,0)))
                    plt.show()
                    logger.setLevel(old_level)
                cell_height = yolo_interval                                                                            ## (16)
                cell_width = yolo_interval                                                                             ## (17)
                if yolo_debug:
                    print("\n\nnum_objects_in_image: ")
                    print(num_objects_in_image)
                num_cells_image_width = self.rpg.image_size[0] // yolo_interval                                        ## (18)
                num_cells_image_height = self.rpg.image_size[1] // yolo_interval                                       ## (19)
                height_center_bb = torch.zeros(im_tensor.shape[0], 1).float().to(self.rpg.device)                      ## (20)
                width_center_bb = torch.zeros(im_tensor.shape[0], 1).float().to(self.rpg.device)                       ## (21)
                obj_bb_height = torch.zeros(im_tensor.shape[0], 1).float().to(self.rpg.device)                         ## (22)
                obj_bb_width =  torch.zeros(im_tensor.shape[0], 1).float().to(self.rpg.device)                         ## (23)

                ## idx is for object index
                for idx in range(max_obj_num):                                                                         ## (24)
                    ## In the mask, 1 means good image instance in batch, 0 means bad image instance in batch
                    batch_mask = torch.ones( self.rpg.batch_size, dtype=torch.int8).to(self.rpg.device)
                    if yolo_debug:
                        print("\n\n               ================  object indexed %d ===============              \n\n" % idx)
                    ## Note that the bounding-box coordinates are in the (x,y) format, with x-positive going to
                    ## right and the y-positive going down. A bbox is specified by (x_min,y_min,x_max,y_max):
                    if yolo_debug:
                        print("\n\nshape of bbox_tensor: ", bbox_tensor.shape)
                        print("\n\nbbox_tensor:")
                        print(bbox_tensor)
                    ## in what follows, the first index (set to 0) is for the batch axis
                    height_center_bb =  (bbox_tensor[:,idx,1] + bbox_tensor[:,idx,3]) // 2                             ## (25)
                    width_center_bb =  (bbox_tensor[:,idx,0] + bbox_tensor[:,idx,2]) // 2                              ## (26)
                    obj_bb_height = bbox_tensor[:,idx,3] -  bbox_tensor[:,idx,1]                                       ## (27)
                    obj_bb_width = bbox_tensor[:,idx,2] - bbox_tensor[:,idx,0]                                         ## (28)
                    if (obj_bb_height < 4.0) or (obj_bb_width < 4.0): continue                                         ## (29)

                    cell_row_indx =  (height_center_bb / yolo_interval).int()          ## for the i coordinate         ## (30)
                    cell_col_indx =  (width_center_bb / yolo_interval).int()           ## for the j coordinates        ## (31)
                    cell_row_indx = torch.clamp(cell_row_indx, max=num_cells_image_height - 1)                         ## (32)
                    cell_col_indx = torch.clamp(cell_col_indx, max=num_cells_image_width - 1)                          ## (33)

                    ## The bh and bw elements in the yolo vector for this object:  bh and bw are measured relative 
                    ## to the size of the grid cell to which the object is assigned.  For example, bh is the 
                    ## height of the bounding-box divided by the actual height of the grid cell.
                    bh  =  obj_bb_height.float() / yolo_interval                                                       ## (34)
                    bw  =  obj_bb_width.float()  / yolo_interval                                                       ## (35)

                    ## You have to be CAREFUL about object center calculation since bounding-box coordinates
                    ## are in (x,y) format --- with x-positive going to the right and y-positive going down.
                    obj_center_x =  (bbox_tensor[:,idx][2].float() +  bbox_tensor[:,idx][0].float()) / 2.0             ## (36)
                    obj_center_y =  (bbox_tensor[:,idx][3].float() +  bbox_tensor[:,idx][1].float()) / 2.0             ## (37)
                    ## Now you need to switch back from (x,y) format to (i,j) format:
                    yolocell_center_i =  cell_row_indx*yolo_interval + float(yolo_interval) / 2.0                      ## (38)
                    yolocell_center_j =  cell_col_indx*yolo_interval + float(yolo_interval) / 2.0                      ## (39)
                    del_x  =  (obj_center_x.float() - yolocell_center_j.float()) / yolo_interval                       ## (40)
                    del_y  =  (obj_center_y.float() - yolocell_center_i.float()) / yolo_interval                       ## (41)
                    class_label_of_object = bbox_label_tensor[:,idx].tolist()                                            ## (42)
                    ## When batch_size is only 1, it is easy to discard an image that has no known objects in it.
                    ## To generalize this notion to arbitrary batch sizes, you will need a batch mask to indicate
                    ## the images in a batch that should not be considered in the rest of this code.

                    ## update the batch_mask - set to zero is class is 13 i.e. no object present
                    batch_mask = batch_mask.masked_fill_(torch.BoolTensor([1 if i==13 else 0 for i in cls]).to(self.rpg.device),0)
                    batch_mask = batch_mask.tolist()
                    # if class_label_of_object == 13: continue  
                    for i in batch_mask:
                        if i != 0:
                            sAR = obj_bb_height.float() / obj_bb_width.float()                                                  ## (44)
                            if AR <= 0.2:               anch_box_index = 0                                                     ## (45)
                            if 0.2 < AR <= 0.5:         anch_box_index = 1                                                     ## (46)
                            if 0.5 < AR <= 1.5:         anch_box_index = 2                                                     ## (47)
                            if 1.5 < AR <= 4.0:         anch_box_index = 3                                                     ## (48)
                            if AR > 4.0:                anch_box_index = 4                                                     ## (49)
                            yolo_vector = torch.FloatTensor([0,del_x.item(), del_y.item(), bh.item(), bw.item(), 0, 0, 0] )    ## (50)
                            yolo_vector[0] = 1                                                                                 ## (51)
                            yolo_vector[5 + class_label_of_object] = 1                                                         ## (52)
                            yolo_cell_index =  cell_row_indx.tolist() * num_cells_image_width  +  cell_col_indx.tolist()           ## (53)
                            yolo_tensor[0,yolo_cell_index, anch_box_index] = yolo_vector                                       ## (54)
                            yolo_tensor_aug = torch.zeros(self.rpg.batch_size, num_yolo_cells, \
                                                                        num_anchor_boxes,9).float().to(self.rpg.device)         ## (55) 
                            yolo_tensor_aug[:,:,:,:-1] =  yolo_tensor                                                          ## (56)
                            if yolo_debug: 
                                print("\n\nyolo_tensor specific: ")
                                print(yolo_tensor[0,18,2])
                                print("\nyolo_tensor_aug_aug: ") 
                                print(yolo_tensor_aug[0,18,2])
                ## If no object is present, throw all the prob mass into the extra 9th ele of yolo_vector
                for icx in range(num_yolo_cells):                                                                      ## (57)
                    for iax in range(num_anchor_boxes):                                                                ## (58)
                        if yolo_tensor_aug[0,icx,iax,0] == 0:                                                          ## (59)
                            yolo_tensor_aug[0,icx,iax,-1] = 1                                                          ## (60)
                if yolo_debug:
                    logger = logging.getLogger()
                    old_level = logger.level
                    logger.setLevel(100)
                    plt.figure(figsize=[15,4])
                    plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                        padding=3, pad_value=255).cpu(), (1,2,0)))
                    plt.show()

                optimizer.zero_grad()                                                                                  ## (61)
                output = net(im_tensor)                                                                                ## (62)
                predictions_aug = output.view(self.rpg.batch_size,num_yolo_cells,num_anchor_boxes,9)                   ## (63)
                loss = torch.tensor(0.0, requires_grad=True).float().to(self.rpg.device)                               ## (64)
                for icx in range(num_yolo_cells):                                                                      ## (65)
                    for iax in range(num_anchor_boxes):                                                                ## (66)
                        pred_yolo_vector = predictions_aug[0,icx,iax]                                                  ## (67)
                        target_yolo_vector = yolo_tensor_aug[0,icx,iax]                                                ## (68)
                        ##  Estiming presence/absence of object and the Binary Cross Entropy section:
                        object_presence = nn.Sigmoid()(torch.unsqueeze(pred_yolo_vector[0], dim=0))                    ## (69)
                        target_for_prediction = torch.unsqueeze(target_yolo_vector[0], dim=0)                          ## (70)
                        bceloss = criterion1(object_presence, target_for_prediction)                                   ## (71)
                        loss += bceloss                                                                                ## (72)
                        ## MSE section for regression params:
                        pred_regression_vec = pred_yolo_vector[1:5]                                                    ## (73)
                        pred_regression_vec = torch.unsqueeze(pred_regression_vec, dim=0)                              ## (74)
                        target_regression_vec = torch.unsqueeze(target_yolo_vector[1:5], dim=0)                        ## (75)
                        regression_loss = criterion2(pred_regression_vec, target_regression_vec)                       ## (76)
                        loss += regression_loss                                                                        ## (77)
                        ##  CrossEntropy section for object class label:
                        probs_vector = pred_yolo_vector[5:]                                                            ## (78)
                        probs_vector = torch.unsqueeze( probs_vector, dim=0 )                                          ## (79)
                        target = torch.argmax(target_yolo_vector[5:])                                                  ## (80)
                        target = torch.unsqueeze( target, dim=0 )                                                      ## (81)
                        class_labeling_loss = criterion3(probs_vector, target)                                         ## (82)
                        loss += class_labeling_loss                                                                    ## (83)
                if yolo_debug:
                    print("\n\nshape of loss: ", loss.shape)
                    print("\n\nloss: ", loss)
                loss.backward()                                                                                        ## (84)
                optimizer.step()                                                                                       ## (85)
                running_loss += loss.item()                                                                            ## (86)
                if iter%1000==999:                                                                                     ## (87)
                    if display_images:
                        print("\n\n\n")                ## for vertical spacing for the image to be displayed later
                    current_time = time.perf_counter()
                    elapsed_time = current_time - start_time 
                    avg_loss = running_loss / float(1000)                                                              ## (88)
                    print("\n[epoch:%d/%d, iter=%4d  elapsed_time=%5d secs]      mean value for loss: %7.4f" % 
                                                        (epoch+1,self.rpg.epochs, iter+1, elapsed_time, avg_loss))     ## (89)
                    Loss_tally.append(running_loss)
                    FILE1.write("%.3f\n" % avg_loss)
                    FILE1.flush()
                    running_loss = 0.0                                                                                 ## (90)
                    if display_labels:
                        predictions = output.view(self.rpg.batch_size,num_yolo_cells,num_anchor_boxes,9)               ## (91)
                        if yolo_debug:
                            print("\n\nyolo_vector for first image in batch, cell indexed 18, and AB indexed 2: ")
                            print(predictions[0, 18, 2])
                        for ibx in range(predictions.shape[0]):                             # for each batch image     ## (92)
                            icx_2_best_anchor_box = {ic : None for ic in range(36)}                                    ## (93)
                            for icx in range(predictions.shape[1]):                         # for each yolo cell       ## (94)
                                cell_predi = predictions[ibx,icx]                                                      ## (95)
                                prev_best = 0                                                                          ## (96)
                                for anchor_bdx in range(cell_predi.shape[0]):                                          ## (97)
                                    if cell_predi[anchor_bdx][0] > cell_predi[prev_best][0]:                           ## (98)
                                        prev_best = anchor_bdx                                                         ## (99)
                                best_anchor_box_icx = prev_best                                                        ## (100)
                                icx_2_best_anchor_box[icx] = best_anchor_box_icx                                       ## (101)
                            sorted_icx_to_box = sorted(icx_2_best_anchor_box,                                   
                                    key=lambda x: predictions[ibx,x,icx_2_best_anchor_box[x]][0].item(), reverse=True)   ## (102)
                            retained_cells = sorted_icx_to_box[:5]                                                     ## (103)
                            objects_detected = []                                                                      ## (104)
                            for icx in retained_cells:                                                                 ## (105)
                                pred_vec = predictions[ibx,icx, icx_2_best_anchor_box[icx]]                            ## (106)
                                class_labels_predi  = pred_vec[-4:]                                                    ## (107)
                                class_labels_probs = torch.nn.Softmax(dim=0)(class_labels_predi)                       ## (108)
                                class_labels_probs = class_labels_probs[:-1]                                           ## (109)
                                if torch.all(class_labels_probs < 0.25):                                               ## (110)
                                    predicted_class_label = None                                                       ## (111)
                                else:                                                                                
                                    best_predicted_class_index = (class_labels_probs == class_labels_probs.max())      ## (112)
                                    best_predicted_class_index =torch.nonzero(best_predicted_class_index,as_tuple=True)## (113)
                                    predicted_class_label =self.rpg.class_labels[best_predicted_class_index[0].item()] ## (114)
                                    objects_detected.append(predicted_class_label)                                     ## (115)
                            print("[batch image=%d]  objects found in descending probability order: " % ibx, 
                                                                                                    objects_detected)     ## (116)
                    if display_images:
                        logger = logging.getLogger()
                        old_level = logger.level
                        logger.setLevel(100)
                        plt.figure(figsize=[15,4])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                            padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.show()
                        logger.setLevel(old_level)
        print("\nFinished Training\n")
        plt.figure(figsize=(10,5))
        plt.title("Loss vs. Iterations")
        plt.plot(Loss_tally)
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("training_loss.png")
        plt.show()
        torch.save(net.state_dict(), self.rpg.path_saved_yolo_model)
        return net

## set the dataloaders
# yolo.set_dataloaders(train=True)
# yolo.set_dataloaders(test=True)
yolo = batchedYOLO( rpg = rpg )

## custom dataloader
# prepare train dataloader
transform = tvt.Compose([tvt.ToTensor(),tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  
coco  = COCO('/home/varun/work/courses/why2learn/hw/annotations/instances_train2017.json')
dataserver_train = CocoDetection(transform, rpg.dataroot_train, rpg.class_labels, rpg.image_size, coco, loadDict=True, saveDict=False, mode="train")
yolo.train_dataloader = torch.utils.data.DataLoader(dataserver_train, batch_size=rpg.batch_size, shuffle=True, num_workers=16)

model = yolo.NetForYolo(skip_connections=True, depth=8) 

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
num_layers = len(list(model.parameters()))
print("\n\nThe number of layers in the model: %d\n\n" % num_layers)

# %%
# train, test, both
mode = "both"
if mode=="train" or mode=="both":
    model = yolo.run_code_for_training_multi_instance_detection(model, display_images=False)
if mode=="test" or mode=="both":
    yolo.run_code_for_testing_multi_instance_detection(model, display_images = True)


