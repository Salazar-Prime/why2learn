#!/usr/bin/env python

##  selective_search.py

import torch, numpy, os, random

seed = 0           
random.seed(seed)
torch.manual_seed(seed)


from RegionProposalGenerator import *

rpg = RegionProposalGenerator(
###            The first 6 options affect only the graph-based part of algo
               sigma = 0.8,
               max_iterations = 40,
               kay = 0.05,
#               image_normalization_required = True,
               image_size_reduction_factor = 4,
               min_size_for_graph_based_blobs = 1,

###            The next 4 options affect only the SS part of the algo
               color_homogeneity_thresh = [20,20,20],      
               gray_var_thresh = 50,            
               texture_homogeneity_thresh = 120,           
               max_num_blobs_expected = 40
      )

image_name = "wallpic2.jpg"

segmented_graph,color_map = rpg.graph_based_segmentation(image_name)

rpg.visualize_segmentation_in_pseudocolor(segmented_graph[0], color_map, "graph_based" )

print("\n\n\nNumber of region proposals after graph-based segmentation: %d" % len(segmented_graph[0]))

merged_blobs, color_map = rpg.selective_search_for_region_proposals( segmented_graph, image_name )

print("\n\n\nNumber of region proposals after selective-search based merging: %d" % len(merged_blobs))

rpg.visualize_segmentation_with_mean_gray(merged_blobs, "ss_based_segmentation_in_bw" )

