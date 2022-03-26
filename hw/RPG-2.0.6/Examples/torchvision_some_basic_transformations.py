#!/usr/bin/env python

##  torchvision_some_basic_transformations.py

##  See the comment blocks in the method 
##
##       graying_resizing_binarizing()
##
##  to appreciate what is being demonstrated here.

from RegionProposalGenerator import *


image_file = "images/fruitlets2.jpg"

rpg = RegionProposalGenerator()

rpg.graying_resizing_binarizing( image_file )


