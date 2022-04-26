"""
Homework 8: Create GRU network  
Author: Varun Aggarwal
Last Modified: 25 Apr 2022
Modifed from DLStudioV2.2.2
"""

import random
import numpy
import torch
import os, sys


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


sys.path.append("/home/varun/work/courses/why2learn/hw/DLStudio-2.2.2")
from DLStudio import *
from DataPrediction import *
import modelpmGRU as pmGRU

type_GRU = "torch"
# type_GRU = "pmGRU"


dataroot = "/home/varun/work/courses/why2learn/hw/DLStudio-2.2.2/Examples/data/"

dataset_archive_train = "sentiment_dataset_train_200.tar.gz"
dataset_archive_test =  "sentiment_dataset_test_200.tar.gz"

path_to_saved_embeddings = "/home/varun/work/courses/why2learn/hw/DLStudio-2.2.2/Examples/runs/"

if type_GRU == "torch":
    dls = DLStudio(
                    dataroot = dataroot,
                    path_saved_model = "/home/varun/work/courses/why2learn/hw/hw8/runs/saved_model_GRU.pt",
                    momentum = 0.9,
                    learning_rate =  1e-3,
                    epochs = 3,
                    batch_size = 1,
                    classes = ('negative','positive'),
                    use_gpu = True,
                )
elif type_GRU=="pmGRU":
    dls = DLStudio(
                    dataroot = dataroot,
                    path_saved_model = "/home/varun/work/courses/why2learn/hw/hw8/runs/saved_model_pmGRU.pt",
                    momentum = 0.9,
                    learning_rate =  1e-3,
                    epochs = 3,
                    batch_size = 1,
                    classes = ('negative','positive'),
                    use_gpu = True,
                )


text_cl = DLStudio.TextClassificationWithEmbeddings( dl_studio = dls )

dataserver_train = DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset(
                                train_or_test = 'train',
                                dl_studio = dls,
                                dataset_file = dataset_archive_train,
                                path_to_saved_embeddings = path_to_saved_embeddings,
                )
dataserver_test = DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset(
                                train_or_test = 'test',
                                dl_studio = dls,
                                dataset_file = dataset_archive_test,
                                path_to_saved_embeddings = path_to_saved_embeddings,
                )
text_cl.dataserver_train = dataserver_train
text_cl.dataserver_test = dataserver_test

text_cl.load_SentimentAnalysisDataset(dataserver_train, dataserver_test)

if type_GRU=="torch":
    model = text_cl.GRUnetWithEmbeddings(input_size=300, hidden_size=100, output_size=2, num_layers=2)
elif type_GRU=="pmGRU":
    model = pmGRU.custom_pmGRU(input_size=300, hidden_size=100, output_size=2, batch_size=dls.batch_size, num_layers=1)


## TESTING:
text_cl.run_code_for_testing_text_classification_with_GRU_word2vec(model)


