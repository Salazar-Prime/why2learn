"""
Homework 4: Create Convolution Neural Network (CNN)  
Author: Varun Aggarwal
Last Modified: 10 Feb 2022
"""

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as tvt

import numpy as np
import seaborn as sns
import pickle

# User imports
from dataLoader import dataLoader
from model import mynet

def run_code_for_validation(net, valDataLoader, classes):
    net = net.to(device)
    confusion = np.zeros((len(classes),len(classes)))

    for i,data in enumerate(valDataLoader):
        (inputs, labels) = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)
        prediction = torch.argmax(outputs, dim=1)
         
        for j, gt in enumerate(labels):
            confusion[gt][prediction[j]] +=1
    return confusion

def cf_plot(no,confusion):
    sns.heatmap(confusion, fmt="0.0f", cmap='Blues', annot=True, xticklabels=dt.classes, yticklabels=dt.classes)
    plt.title("Net %d: Accuracy-%0.2f"%(no,(100*np.trace(confusion)/np.sum(confusion))))
    plt.savefig("../saves/net%d_confusion_matrix.png"%(no))
    plt.close()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

batch=256
dataPath = "../hw04_coco_data/Val"
transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5,0.5, 0.5), (0.5, 0.5, 0.5))])            
           
dt = dataLoader(dataPath,transform)
valDataLoader = DataLoader(dataset = dt, batch_size = batch, shuffle = False, num_workers = 16)

# Net 1 - Validation 
net = torch.load("../saves/net1.pth")
net.eval()
confusion1 = run_code_for_validation(net, valDataLoader, dt.classes)
cf_plot(1,confusion1)

# Net 2 - Validation 
net = torch.load("../saves/net2.pth")
net.eval()
confusion2 = run_code_for_validation(net, valDataLoader, dt.classes)
cf_plot(2,confusion2)

# Net 3 - Validation 
net = torch.load("../saves/net3.pth")
net.eval()
confusion3 = run_code_for_validation(net, valDataLoader, dt.classes)
cf_plot(3,confusion3)

# save loss history
confusion_hist = [confusion1,confusion2,confusion3]
with open("../saves/confusion_hist.pickle", 'wb') as f:
    pickle.dump(confusion_hist, f)
