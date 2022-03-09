#!/usr/bin/env python

##  interactive_graph_based_segmentation.py

##  The goal of this script is to demonstrate the following:
##
##     1) How to choose just a portion of the image for graph-based segmentation; and
##
##     2) How to apply the graph-based segmentation algorithm to just the portion selectd.


##  SPECIFYING IMAGE PORTION FOR SEGMENTATION:
##
##    The module gives you two different methods for specifying a portion of the image
##    for applying RegionProposalGeneator to:
##
##       i) You can click at a point and then drag the mouse to define a rectangular
##          portion of the image;
##
##       2) You can specify any polygonal shaped area by clicking the mouse at the
##          vertices of the polygon you have in mind.
##
##    The first of these is provided by the method:
##
##            extract_image_region_interactively_by_dragging_mouse()
##
##    and the second by
##
##            extract_image_region_interactively_through_mouse_clicks()


from RegionProposalGenerator import *

image_file = "images/polkadots.jpg"             ## 600x450

rpg = RegionProposalGenerator(
#               sigma = 0.8,
               kay = 0.05,
               image_size_reduction_factor = 1,
               max_iterations = 10,
               min_size_for_graph_based_blobs = 1,
       )


pil_image_section = rpg.extract_image_region_interactively_through_mouse_clicks(image_file)

#pil_image_portion = rpg.extract_image_region_interactively_by_dragging_mouse(image_file)

rpg.graph_based_segmentation( pil_image_section )

