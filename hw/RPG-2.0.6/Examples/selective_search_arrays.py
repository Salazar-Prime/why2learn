#!/usr/bin/env python

##  selective_search_arrays.py

##  XXXXXXXXXXXXXXXXxThe goal of this script is to demonstrate hue thresholding by applying the



import torch, numpy, os, random

seed = 0               ### reproducibility seems to work only when the seed is to ZERO 
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)



from RegionProposalGenerator import *

rpg = RegionProposalGenerator(
               sigma = 0.8,
               max_iterations = 40,
               kay = 0.05,
#               kay = 100,
               color_homogeneity_thresh = 3,
               color_var_thresh = 2,
               texture_homogeneity_thresh = 3
      )


#rpg.graph_based_segmentation_for_arrays(1)

#rpg.graph_based_segmentation_for_arrays(2)

"""
pixel_blobs = { 1: [(12,4),(12,5),(13,6),(14,7)],
                2: [(13,5),(14,4),(15,4),(16,8)],
                3: [(14,6),(15,7),(16,8),(17,9)],
                4: [(14,6),(15,7),(16,8),(17,9), (17,10)],
                5: [(4,4),(4,5),(5,6),(6,7)],
                6: [(5,5),(6,4),(7,4),(8,8)],
              }
"""

pixel_blobs = rpg.graph_based_segmentation_for_arrays(1)

print "\n\n\npixel_blobs produced: ", pixel_blobs

merged_blobs = rpg.selective_search_for_region_proposals( pixel_blobs, "array1.png" )

print "\n\n\nmerged blobs: ", merged_blobs

rpg.visualize_segmentation(merged_blobs, "array1.png" )

