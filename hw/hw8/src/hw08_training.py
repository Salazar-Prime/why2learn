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

#dataroot = "/home/kak/TextDatasets/sentiment_dataset/"
# dataroot = "./data/TextDatasets/sentiment_dataset/"
dataroot = "/home/varun/work/courses/why2learn/hw/DLStudio-2.2.2/Examples/data/"

dataset_archive_train = "sentiment_dataset_train_200.tar.gz"
#dataset_archive_train = "sentiment_dataset_train_200.tar.gz"

dataset_archive_test =  "sentiment_dataset_test_200.tar.gz"
#dataset_archive_test = "sentiment_dataset_test_200.tar.gz"

path_to_saved_embeddings = "/home/varun/work/courses/why2learn/hw/DLStudio-2.2.2/Examples/runs/"
# path_to_saved_embeddings = "/home/kak/TextDatasets/word2vec/"
#path_to_saved_embeddings = "./data/TextDatasets/word2vec/"


dls = DLStudio(
                  dataroot = dataroot,
                  path_saved_model = "/home/varun/work/courses/why2learn/hw/DLStudio-2.2.2/Examples/runs/saved_model",
                  momentum = 0.9,
                  learning_rate =  1e-3,
                  epochs = 5,
                  batch_size = 1,
                  classes = ('negative','positive'),
                  use_gpu = True,
              )

# type_GRU = "torch"
type_GRU = "pmGRU"

if type_GRU=="torch":
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

    model = text_cl.GRUnetWithEmbeddings(input_size=300, hidden_size=100, output_size=2, num_layers=2)

    number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_layers = len(list(model.parameters()))

    print("\n\nThe number of layers in the model: %d" % num_layers)
    print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)

    ## TRAINING:
    print("\nStarting training\n")
    text_cl.run_code_for_training_for_text_classification_with_GRU_word2vec(model, display_train_loss=True)
elif type_GRU=="pmGRU":
    predictor = DataPrediction(
                  dlstudio = dls,
                  input_size = 5,      # means that each entry consists of one observation and 4 values for encoding datetime
                  hidden_size = 256,
                  output_size = 1,     # for the prediction 
                  sequence_length = 90,
                  ngpu = 1,    
              )

    # model = DataPrediction.pmGRU(predictor)
    # def __init__(self, input_size, hidden_size, output_size, num_layers=1, batch_size=1): 
    model = pmGRU.custom_pmGRU(predictor.input_size,predictor.hidden_size, predictor.output_size, dls.batch_size)


    print("\n\nmodel: ", model)

    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n\nThe number of learnable parameters in pmGRU: %d\n" % num_learnable_params)

    dataframes = predictor.construct_dataframes_from_datafiles()
    dataframes_normalized = predictor.data_normalizer( dataframes )
    predictor.construct_sequences_from_data(dataframes_normalized)
    dataloader = predictor.set_dataloader()

    trained_model = predictor.run_code_for_training_data_predictor(dataloader, model)

    print("\n\n\nFinished training.  Starting evaluation on unseen data.\n\n")
    predictions, gt_for_predictions =  predictor.run_code_for_evaluation_on_unseen_data(trained_model)  
    predictor.display_sample_predictions(predictions, gt_for_predictions)
# ## TESTING:
# text_cl.run_code_for_testing_text_classification_with_GRU_word2vec(model)


