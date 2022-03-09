#!/usr/bin/env python

##  torchvision_based_image_processing.py

##  See the comment blocks associated with the methods called in this
##  file to appreciate the image processing step being demonstrated.


from RegionProposalGenerator import *

image_file = "images/fruitlets2.jpg"

rpg = RegionProposalGenerator()

##  For examining the 'R' plane of the color image:
rpg.accessing_one_color_plane( image_file, 0 )

rpg.working_with_hsv_color_space( image_file )

rpg.histogramming_the_image( image_file )

kernel = [ [-1, 0, 1], [0, 0, 0], [-1, 0, 1] ]
rpg.convolutions_with_pytorch( image_file, kernel )

rpg.histogramming_and_thresholding( image_file )
