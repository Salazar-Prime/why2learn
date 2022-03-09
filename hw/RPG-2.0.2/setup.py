#!/usr/bin/env python

### setup.py

from setuptools import setup, find_packages
import sys, os

setup(name='RegionProposalGenerator',
      version='2.0.2',
      author='Avinash Kak',
      author_email='kak@purdue.edu',
      maintainer='Avinash Kak',
      maintainer_email='kak@purdue.edu',
      url='https://engineering.purdue.edu/kak/distRPG/RegionProposalGenerator-2.0.2.html',
      download_url='https://engineering.purdue.edu/kak/distRPG/RegionProposalGenerator-2.0.2.tar.gz',
      description='An educational module for experimenting with single-instance and multi-instance object detection and for generating region proposals with graph-based algorithms',
      long_description='''

Consult the module API page at

      https://engineering.purdue.edu/kak/distRPG/RegionProposalGenerator-2.0.2.html

for all information related to this module, including information related
to the latest changes to the code.  The page at the URL shown above lists
all of the module functionality you can invoke in your own code.

::

        Single-Instance and Multi-Instance Object Detection:
    
            Say you wish to experiment with YOLO-like logic for multi-instance
            object detection, you would need to construct an instance of the
            RegionProposalGenerator class and invoke the methods shown below on
            this instance:
        
            rpg = RegionProposalGenerator(
                              dataroot = "./data/",
                              image_size = [128,128],
                              yolo_interval = 20,
                              path_saved_yolo_model = "./saved_yolo_model",
                              momentum = 0.9,
                              learning_rate = 1e-6,
                              epochs = 40,
                              batch_size = 4,
                              classes = ('Dr_Eval','house','watertower'),
                              use_gpu = True,
                          )
            yolo = RegionProposalGenerator.YoloLikeDetector( rpg = rpg )
            yolo.set_dataloaders(train=True)
            yolo.set_dataloaders(test=True)
            model = yolo.NetForYolo(skip_connections=True, depth=8) 
            model = yolo.run_code_for_training_multi_instance_detection(model, display_images=False)
            yolo.run_code_for_training_multi_instance_detection(model, display_images = True)
            
    
        Graph-Based Algorithms for Region Proposals:
    
            To generate region proposals, you would need to construct an instance
            of the RegionProposalGenerator class and invoke the methods shown below
            on this instance:
        
            rpg = RegionProposalGenerator(
                           ###  The first 6 options affect only the graph-based part of the algo
                           sigma = 1.0,
                           max_iterations = 40,
                           kay = 0.05,
                           image_normalization_required = True,
                           image_size_reduction_factor = 4,
                           min_size_for_graph_based_blobs = 4,
                           ###  The next 4 options affect only the Selective Search part of the algo
                           color_homogeneity_thresh = [20,20,20],
                           gray_var_thresh = 16000,           
                           texture_homogeneity_thresh = 120,
                           max_num_blobs_expected = 8,
                  )
            
            image_name = "images/mondrian.jpg"
            segmented_graph,color_map = rpg.graph_based_segmentation(image_name)
            rpg.visualize_segmentation_in_pseudocolor(segmented_graph[0], color_map, "graph_based" )
            merged_blobs, color_map = rpg.selective_search_for_region_proposals( segmented_graph, image_name )
            rpg.visualize_segmentation_with_mean_gray(merged_blobs, "ss_based_segmentation_in_bw" )

    
          ''',

      license='Python Software Foundation License',
      keywords='object detection, image segmentation, computer vision',
      platforms='All platforms',
      classifiers=['Topic :: Scientific/Engineering :: Image Recognition', 'Programming Language :: Python :: 3.8'],
      packages=['RegionProposalGenerator']
)
