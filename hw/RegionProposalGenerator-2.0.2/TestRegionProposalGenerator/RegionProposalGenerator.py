__version__   = '2.0.2'
__author__    = "Avinash Kak (kak@purdue.edu)"
__date__      = '2021-April-7'   
__url__       = 'https://engineering.purdue.edu/kak/distRPG/RegionProposalGenerator-2.0.2.html'
__copyright__ = "(C) 2021 Avinash Kak. Python Software Foundation."

__doc__ = '''

RegionProposalGenerator.py

Version: ''' + __version__ + '''
   
Author: Avinash Kak (kak@purdue.edu)

Date: ''' + __date__ + '''


@title
CHANGE LOG:

  Version 2.0.2:

    This version fixes a couple of bugs in the YOLO-based logic for
    multi-instance object detection.

  Version 2.0.1:

    This module has gone through several changes since its last
    public-release version as I was experimenting with different ways of
    imparting to the students the sudden increase in model complexity as
    one goes from single-instance object detection to multi-instance object
    detection.  These experiments led to the creation of two new datasets,
    PurdueDrEvalDataset and PurdueDrEvalMultiDataset, the former for
    playing with single-instance object detection and the latter for doing
    the same with multi-instance object detection.  The module also
    includes two inner classes, SingleInstanceDetector and
    YoloLikeDetector, the former a reference implementation for single
    instance object detection and the latter a YOLO-like reference
    implementation for multi-instance object detection. [By the way,
    "DrEval" in the names of the two datasets mentioned here has a
    connection with "Dr Evil" in the Austin Powers movies.]

  Version 1.0.5:

    In keeping with the tutorial nature of this module, this version
    includes methods that come in handy for batch-based processing of
    images. These methods carry names like "displaying_and_histogramming_
    images_in_batchX()" where X is 1, 2, and 3.  The rest of the module,
    especially the part that deals with constructing region proposals
    remains unchanged.

  Version 1.0.4:

    This is the first public release version of the module.


@title
INTRODUCTION:

    Single-Instance vs. Multi-Instance Detection:

    This module was created for experimenting with the logic of object
    detection with neural networks.  On the face of it, object detection in
    images sounds like a well-defined problem that should lend itself to
    well-defined solutions.  Unfortunately, the reality is otherwise.  Yes,
    simple examples of the problem -- such as when the images contain
    single object instances and with no competing clutter in the background
    -- the problem can be solved straightforwardly with a neural network.
    However, the object detection problems that are encountered in real
    life are rarely that simple.  A practically useful framework for object
    detection must be able to recognize and localize all possible instances
    of the objects of interest in a given image.

    So how does one solve the problem multi-instance object detection and
    localization with a neural network?

    The last half-dozen years have seen the emergence of the following
    three competition-grade neural-network based approaches for
    multi-instance object detection: R-CNN, YOLO, and SSD.  My Week 9
    lecture material for Purdue's Deep Learning class provides a brief
    overview of these approaches.

    The goal of the present module is for it serve as a tool for
    experimenting with single-instance and multi-instance object detection
    and to appreciate how much more difficult the latter problem is.  Yes,
    you could directly download the code for, say, YOLOv3 (version 3 of
    YOLO) and have it do the right thing on your images, but that would not
    serve the purpose of education.  In order to understand in depth the
    logic used in the successful object detectors, you have no choice but
    to write your own code for at least the key elements of that logic.
    That's what the purpose my module is --- to provide you with a platform
    for experimenting with the key steps of the well known object
    detectors.

    Graph-Based Algorithms for Region Proposals:

    A second goal of this module is to provide implementations for a couple
    of the more modern graph-based approaches for generating region
    proposals.  At this point, the reader might ask: What is a region
    proposal?  A region proposal is a blob of pixels that is highly likely
    to contain an object instance.  Another way of saying same thing is
    that region proposals are pixel blobs that look different from the
    general background in the images.  While it is possible to use a neural
    network for generating region proposals, as demonstrated by the success
    of RPN (Region Proposal Network) in the R-CNN based multi-instance
    object detection, the RPG module is concerned primarily with the
    non-neural-network based methods -- the graph-based methods -- for
    generating region proposals.  I believe that becoming familiar with the
    non-learning based methods for constructing region proposals still has
    considerable value.  Consider, for example, the problem of detecting
    objects in satellite images where you simply do not have access to the
    amount of training data you would need for a neural-network based
    approach to work.

    With regard to the graph-based method for generating region proposals,
    RPG (RegionProposalGenerator) implements elements of the Selective
    Search (SS) algorithm for object detection as proposed by Uijlings, van
    de Sande, Gevers, and Smeulders.  The Selective Search algorithm sits
    on top of the graph-based image segmentation algorithm of Felzenszwalb
    and Huttenlocher (FH) whose implementation is also included in the RPG
    module.  The RPG module first processes an image with the FH
    graph-based algorithm for image segmentation to divide an image into
    pixel blobs.  The module subsequently invokes elements of the SS
    algorithm to selectively merge the blobs on the basis of three
    properties: homogeneity of the color, grayscale variance, and texture
    homogeneity.

    The FH algorithm is based on creating a graph-based representation of
    an image in which, at the beginning, each pixel is a single vertex and
    the edge between two vertices that stand for two adjacent pixels
    represents the difference between some pixel property (such as the
    color difference) at the two pixels.  Subsequently, for the vertex
    merging logic, each vertex u, that after the first iteration stands for
    a grouping of pixels, is characterized by a property called Int(u),
    which is the largest value of the inter-pixel color difference between
    the adjacent pixels.  In order to account for the fact that, at the
    beginning, each vertex consists of only one pixel [which would not
    allow for the calculation of Int(u)], the unary property of the pixels
    at a vertex is extended from Int(u) to MInt(u) with the addition of a
    vertex-size dependent number equal to k/|C| where "k" is a
    user-specified parameter and |C| the cardinality of the set of pixels
    represented by the vertex u in the graph.

    As mentioned above, initially the edges in the graph representation of
    an image are set to the color difference between the two 8-adjacent
    pixels that correspond to two different vertices.  Subsequently, as the
    vertices are merged, an edge, E(u,v), between two vertices u and v is
    set to the smallest value of the inter-pixel color difference for two
    adjacent pixels that belong to the two vertices. At each iteration of
    the algorithm, two vertices u and v are merged provided E(u,v) is less
    than the smaller of the MInt(u) or MInt(v) attributes at the two
    vertices.  My experience is that for most images the algorithm
    terminates of its own accord after a small number of iterations while
    the vertex merging condition can be satisfied.

    Since the algorithm is driven by the color differences between
    8-adjacent pixels, the FH algorithm is likely to create too fine a
    segmentation of an image.  The segments produced by FH can be made
    larger by using the logic of SS that allows blobs of pixels to merge
    into larger blobs provided doing so makes sense based on the inter-blob
    values for mean color levels, color variances, texture values, etc.


@title
INSTALLATION:

    The RegionProposalGenerator class was packaged using setuptools.  For
    installation, execute the following command in the source directory
    (this is the directory that contains the setup.py file after you have
    downloaded and uncompressed the package):
 
            sudo python3 setup.py install

    On Linux distributions, this will install the module file at a location
    that looks like

             /usr/local/lib/python3.8/dist-packages/

    If you do not have root access, you have the option of working directly
    off the directory in which you downloaded the software by simply
    placing the following statements at the top of your scripts that use
    the RegionProposalGenerator class:

            import sys
            sys.path.append( "pathname_to_RegionProposalGenerator_directory" )

    To uninstall the module, simply delete the source directory, locate
    where the RegionProposalGenerator module was installed with "locate
    RegionProposalGenerator" and delete those files.  As mentioned above,
    the full pathname to the installed version is likely to look like
    /usr/local/lib/python2.7/dist-packages/RegionProposalGenerator*

    If you want to carry out a non-standard install of the
    RegionProposalGenerator module, look up the on-line information on
    Disutils by pointing your browser to

              http://docs.python.org/dist/dist.html

@title
USAGE:

    Single-Instance and Multi-Instance Detection:

    If you wish to experiment with YOLO-like logic for multi-instance
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
                       ###  The first 6 options affect only the Graph-Based part of the algo
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


@title
CONSTRUCTOR PARAMETERS: 

    Of the 10 constructor parameters listed below, the first six are meant for
    the FH algorithm and the last four for the SS algorithm.

    sigma: Controls the size of the Gaussian kernel used for smoothing the
                    image before its gradient is calculated.  Assuming the
                    pixel sampling interval to be unity, a sigma of 1 gives
                    you a 7x7 smoothing operator with Gaussian weighting.
                    The default for this parameter is 1.

    max_iterations: Sets an upper limit on the number of iterations of the
                    graph-based FH algorithm for image segmentation.

    kay: This is the same as the "k" parameter in the FH algorithm.  As
                    mentioned in the Introduction above, the Int(u)
                    property of the pixels represented by each vertex in
                    the graph representation of the image is extended to
                    MInt(u) by the addition of a number k/|C| where |C| is
                    the cardinality of the set of pixels at that vertex.

    image_normalization_required: This applies Torchvision's image
                    normalization to the pixel values in the image.

    image_size_reduction_factor: As mentioned at the beginning of this
                    document, RegionProposalGenerator is really not meant
                    for production work.  The code is pure Python and, even
                    with that, not at all optimized.  The focus of the
                    module is primarily on easy understandability of what
                    the code is doing so that you can experiment with the
                    algorithm itself.  For the module to produce results
                    within a reasonable length of time, you can use this
                    constructor parameter to downsize the array of pixels
                    that the module must work with.  Set this parameter to
                    a value so that the initial graph constructed from the
                    image has no more than around 3500 vertices if you
                    don't want to wait too long for the results.

    min_size_for_graph_based_blobs: This declares a threshold on the
                   smallest size you'd like to see (in terms of the number
                   of pixels) in a segmented blob in the output of the
                   graph-based segmenter.  (I typically use values from 1
                   to 4 for this parameter.)

    color_homogeneity_thresh:  

                    This and the next three constructor options are meant
                    specifically for the SS algorithm that sits on top of
                    the FH algorithm for further merging of the pixel blobs
                    produced by FH.  This constructor option specifies the
                    maximum allowable difference between the mean color
                    values in two pixel blobs for them to be merged.

    gray_var_thresh:

                   This option declares the maximum allowable difference in 
                   the variances in the grayscale in two blobs if they are
                   to be merged. 

    texture_homogeneity_thresh:

                   The RegionProposalGenerator module characterizes the
                   texture of the pixels in each segmented blob by its LBP
                   (Local Binary Patterns) texture.  We want the LBP
                   texture values for two different blobs to be within the
                   value specified by this constructor option if those
                   blobs are to be merged.

    max_num_blobs_expected:

                   If you only want to extract a certain number of the
                   largest possible blobs, you can do that by giving a
                   value to this constructor option.


@title
Inner Classes:

    (1)  PurdueDrEvalDataset

         This is the dataset to use if you are experimenting with
         single-instance object detection.  The dataset contains three
         kinds of objects in its images: Dr. Eval, and two "objects" in his
         neighborhood: a house and a watertower.  Each 128x128 image in the
         dataset contains one of these objects after it is randomly scaled
         and colored. Each image also contains substantial structured noise
         in addition to 20% Gaussian noise.  Examples of these images are
         shown in the Week 9 lecture material in Purdue's Deep Learning
         class.

    (2)  PurdueDrEvalMultiDataset

         This is the dataset to use if you are experimenting with
         multi-instance object detection.  Each image in the dataset
         contains randomly chosen multiple instances of the same three
         kinds of objects as mentioned above: Dr. Eval, house, and
         watertower.  The number of object instances in each image is
         limited to a maximum of five.  The images contain substantial
         amount of structured noise in addition to 20% random noise.
         
         The reason for why the above two datasets have "DrEval" in their
         names: After having watched every contemporary movie at Netflix
         that was worth watching, my wife and I decided to revisit some of
         old movies that we had enjoyed a long time back.  That led us to
         watching again a couple of Austin Powers movies.  If you are too
         young to know what I am talking about, these movies are spoofs on
         the James Bond movies in which the great comedian Mike Myers plays
         both Austin Powers and his nemesis Dr. Evil.  Around the same
         time, I was writing code for the two datasets mentioned above.
         One of the three objects types in these images is a human-like
         cartoon figure that I needed a name for.  So, after Dr. Evil in
         the movies, I decided to call this cartoon figure Dr Eval and to
         refer to the datasets as Dr Eval datasets. As you all know, "Eval"
         is an important word for people like us.  All programming
         languages provide a function with a name like "eval()".

    (3)  SingleInstanceDetector

         This provides a reference implementation for constructing a
         single-instance object detector to be used for the
         PurdueDrEvalDataset dataset. For the detection and regression
         network, it uses the LOADnet2 network from DLStudio with small
         modifications to account for the larger 128x128 images in the
         dataset.

    (4)  YoloLikeDetector

         The code in this inner class provides an implementation for the
         key elements of the YOLO logic for multi-instance object
         detection.  Each training image is divided into a grid of cells
         and it is the responsibility of a cell that contains the center of
         an object bounding box to provide at the output of the neural
         network an estimate for the exact location of the center of the
         object bounding box vis-a-vis the center of the cell.  That cell
         must also lead to an estimate for the height and the width of the
         true bounding for the object instance.


@title
PUBLIC METHODS:

    Many of these method are related to using this module for experimenting
    with the traditional graph-based algorithms for constructing region
    proposals in images:

    (1)  selective_search_for_region_proposals()

         This method implements elements of the Selective Search (SS)
         algorithm proposed by Uijlings, van de Sande, Gevers, and
         Smeulders for creating region proposals for object detection.
         As mentioned elsewhere here, this algorithm sits on top of
         the graph based image segmentation algorithm that was
         proposed by Felzenszwalb and Huttenlocher. 

    (2)  graph_based_segmentation()

         This is an implementation of the Felzenszwalb and Huttenlocher
         (FH) algorithm for graph-based segmentation of images.  At the
         moment, it is limited to working on grayscale images.

    (3)  display_tensor_as_image()

         This method converts the argument tensor into a photo image that
         you can display in your terminal screen. It can convert tensors of
         three different shapes into images: (3,H,W), (1,H,W), and (H,W),
         where H, for height, stands for the number of pixel in the
         vertical direction and W, for width, the same along the horizontal
         direction. When the first element of the shape is 3, that means
         that the tensor represents a color image in which each pixel in
         the (H,W) plane has three values for the three color channels.  On
         the other hand, when the first element is 1, that stands for a
         tensor that will be shown as a grayscale image.  And when the
         shape is just (H,W), that is automatically taken to be for a
         grayscale image.

    (4)  graying_resizing_binarizing()

         This is a demonstration of some of the more basic and commonly
         used image transformations from the torchvision.transformations
         module.  The large comment blocks are meant to serve as tutorial
         introduction to the syntax used for invoking these
         transformations.  The transformations shown can be used for
         converting a color image into a grayscale image, for resizing an
         image, for converting a PIL.Image into a tensor and a tensor back
         into an PIL.Image object, and so on.

    (5)  accessing_one_color_plane()

         This method shows how can access the n-th color plane of the
         argument color image.

    (6)  working_with_hsv_color_space()

         Illustrates converting an RGB color image into its HSV
         representation.

    (7)  histogramming_the_image()

         PyTorch based experiments with histogramming the grayscale
         and the color values in an image

    (8)  histogramming_and_thresholding():

         This method illustrates using the PyTorch functionality for
         histogramming and thresholding individual images.

    (9)  convolutions_with_pytorch()

         This method calls on torch.nn.functional.conv2d() for
         demonstrating a single image convolution with a specified
         kernel.

    (10) gaussian_smooth()

         This method smooths an image with a Gaussian of specified
         sigma.  You can do the same much faster by using the
         functionality programmed into torch.nn.functional.

    (11) visualize_segmentation_in_pseudocolor()

         After an image has been segmented, this method can be used to
         assign a random color to each blob in the segmented output
         for a better visualization of the segmentation.

    (12) visualize_segmentation_with_mean_gray()

         If the visualization produced by the previous method appears
         too chaotic, you can use this method to assign the mean color
         to each each blob in the output of an image segmentation
         algorithm.  The mean color is derived from the pixel values
         in the blob.

    (13) extract_image_region_interactively_by_dragging_mouse()

         You can use this method to apply the graph-based segmentation
         and the selective search algorithms to just a portion of your
         image.  This method extract the portion you want.  You click
         at the upper left corner of the rectangular portion of the
         image you are interested in and you then drag the mouse
         pointer to the lower right corner.  Make sure that you click
         on "save" and "exit" after you have delineated the area.

    (14) extract_image_region_interactively_through_mouse_clicks()

         This method allows a user to use a sequence of mouse clicks in
         order to specify a region of the input image that should be
         subject to further processing.  The mouse clicks taken together
         define a polygon. The method encloses the polygonal region by a
         minimum bounding rectangle, which then becomes the new input image
         for the rest of processing.

    (15) displaying_and_histogramming_images_in_batch1(image_dir, batch_size)

         This method is the first of three such methods in this module for
         illustrating the functionality of matplotlib for simultaneously
         displaying multiple images and the results obtained from them in a
         gridded arrangement.  The core idea in this method is to call
         "plt.subplots(2,batch_size)" to create 'batch_size' number of
         subplot objects, called "axes", in the form of a '2xbatch_size'
         array. We use the first row of this grid to display each image in
         its own subplot object.  And we use the second row of the grid to
         display the histograms of the corresponding images in the first
         row.

    (16) displaying_and_histogramming_images_in_batch2(image_dir, batch_size)

         I now show a second approach to displaying multiple images and
         their corresponding histograms in a gridded display.  In this
         method we call on "torchvision.utils.make_grid()" to construct a
         grid for us.  The grid is created by giving an argument like
         "nrow=4" to it.  The grid object returned by the call to
         make_grid() is a tensor unto itself. Such a tensor object is
         converted into a numpy array so that it can be displayed by
         matplotlib's "imshow()" function.

    (17) displaying_and_histogramming_images_in_batch3(image_dir, batch_size)

         This method illustrates two things: (1) The syntax used for the
         'singular' version of the subplot function "plt.subplot()" ---
         although I'll be doing so by actually calling "fig.add_subplot()".
         And (2) How you can put together multiple multi-image plots by
         creating multiple Figure objects.  'Figure' is the top-level
         container of plots in matplotlib.  This method creates two
         separate Figure objects, one as a container for all the images in
         a batch and the other as a container for all the histograms for
         the images.  The two Figure containers are displayed in two
         separate windows on your computer screen.

@title
THE DATASETS INCLUDED:

    Download the archive

        datasets_for_RPG.tar.gz

    through the link "Download the image datasets for RPG" at the main
    webpage for this module and store the archive in the Examples directory
    of your install of the module. Subsequently, execute the following
    command in the Examples directory:

        tar zxvf datasets_for_RPG.tar.gz

    This command will create a 'data' subdirectory in the Examples directory
    and deposit the following datasets in it:

        Purdue_Dr_Eval_Dataset-clutter-10-noise-20-size-10000-train.gz
        Purdue_Dr_Eval_Dataset-clutter-10-noise-20-size-1000-test.gz

        Purdue_Dr_Eval_Multi_Dataset-clutter-10-noise-20-size-10000-train.gz
        Purdue_Dr_Eval_Multi_Dataset-clutter-10-noise-20-size-1000-test.gz

    In the naming convention used for the archives, the string 'clutter-10'
    means that each image will have a maximum of 10 clutter objects in it,
    and the string 'noise-20' means that I have added 20% Gaussian noise to
    each image. The string 'size-10000' means that the dataset consists of
    '10,000' images.

    You will also find two smaller datasets in the overall archive that 
    contain just 30 images each.  Those are for debugging your code.



@title 
THE Examples DIRECTORY:

    Single-Instance and Multi-Instance Detections:

    The following scripts in the Examples directory of the distribution
    relate to the use of this module for single-instance and multi-instance
    object detections:

        single_instance_object_detection.py

        multi_instance_object_detection.py

    The first script carries out its single-instance detections in the
    images in the PurdueDrEvalDataset dataset and the second script carries
    out multi-instance detections in the PurdueDrEvalMultiDataset.  In the
    former dataset, each 128x128 image has only one instance of a
    meaningful object along with structured artifacts and 20% random noise.
    And, in the latter dataset, each image has up to five instances of
    meaningful objects along with the structured artifacts and 20% random
    noise.

    Graph-Based Algorithms for Region Proposals:

    The Examples directory contains the following scripts for experimenting
    with graph-based algorithms for constructing region proposals:

        selective_search.py

        interactive_graph_based_segmentation.py

        torchvision_some_basic_transformations.py    

        torchvision_based_image_processing.py

        multi_image_histogramming_and_display.py  

    The Examples directory also illustrates the sort of region proposal
    results you can obtain with the graph-based algorithms in this module.
    The specific illustrations are in the following subdirectories of the
    Examples directory:

        Examples/color_blobs/

        Examples/mondrian/

        Examples/wallpic2/

    Each subdirectory contains at least the following two files:
    the function 

        selective_search.py
        and
        the image file specific for that subdirectory.

    All you have to do is to execute selective_search.py in that directory to
    see the results on the image in that directory.



@title
CAVEATS:

    As mentioned earlier, this module is NOT meant for production work
    for constructing region proposals --- simply because it would be
    much too slow for full-sized images.  Yes, you can work with
    images of any size provided you set the constructor option
    image_size_reduction_factor appropriately.  However, note that
    setting this parameter to a large value causes a significant loss
    of resolution in the image, which reduces the quality of the
    output results.

    For full sized images, you will be better off using the
    implementations provided by the original authors of the
    algorithms.

    The goal of this module is solely to provide code that is a
    self-documented implementation of the algorithms involved.  This
    is to make it easy to experiment with the logic of the algorithms.
    

@title
BUGS:

    Please notify the author if you encounter any bugs.  When sending
    email, please place the string 'RegionProposalGenerator' in the
    subject line to get past the author's spam filter.


@title
ABOUT THE AUTHOR:

    The author, Avinash Kak, is a professor of Electrical and Computer
    Engineering at Purdue University.  For all issues related to this
    module, contact the author at kak@purdue.edu If you send email,
    please place the string "RegionProposalGenerator" in your subject
    line to get past the author's spam filter.

@title
COPYRIGHT:

    Python Software Foundation License

    Copyright 2021 Avinash Kak

@endofdocs
'''

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tvt
import torchvision.transforms.functional as F
import torchvision.utils as tutils
import torch.optim as optim
import numpy as np
import time

from PIL import Image
from PIL import ImageDraw
from PIL import ImageTk
from PIL import ImageFont
import sys,os,os.path,glob,signal
import re
import functools
import math
import random
import copy
import gzip
import pickle
if sys.version_info[0] == 3:
    import tkinter as Tkinter
    from tkinter.constants import *
else:
    import Tkinter    
    from Tkconstants import *

import matplotlib.pyplot as plt
import logging                        ## for suppressing matplotlib warning messages

#____________________________Top level utility functions  ___________________________________
def _gaussian(sigma):
    '''
    A 1-D Gaussian smoothing operator is generated by assuming that the pixel
    sampling interval is a unit distance.  We truncate the operator a 3 times the
    value of sigma.  So when sigma is set to 1, you get a 7-element operator.  On the
    other hand, when sigma is set to 2, you get a 13-element operator, and so on.
    '''
    win_half_width = int(3 * sigma)
    xvals = range(-win_half_width, win_half_width+1)
    gauss = lambda x: math.exp(-((x**2)/(2*float(sigma**2))))
    operator = [gauss(x) for x in xvals]
    summed = functools.reduce( lambda x, y: x+y, operator )
    operator = [x/summed for x in operator]
    return operator

def _convolution_1D(input_array, operator):
    '''
    Since the Gaussian kernel is separable in its x and y dependencies, 2D convolution
    of an image with the kernel can be decomposed into a sequence of 1D convolutions
    first with the rows of the image and then another sequence of 1D convolutions
    with the columns of the output from the first.  This function carries out a 1D
    convolution.
    '''
    height,width = input_array.shape
    result_array = np.zeros((height, width), dtype="float")
    w = len(operator)                   # should be an odd number
    op_half_width = int((w-1)/2)
    for i in range(height):
        for j in range(width):
            accumulated = 0.0
            for k in range(-op_half_width,op_half_width+1):
                if (j+k) >= 0 and (j+k) < width:
                    accumulated += input_array[i,(j+k)] * operator[k + op_half_width]
            result_array[(i,j)] = accumulated
    return result_array

def _convolution_2D(input_array, operator):
    '''
    Since the Gaussian kernel is separable in its x and y dependencies, 2D convolution
    of an image with the kernel can be decomposed into a sequence of 1D convolutions
    first with the rows of the image and then another sequence of 1D convolutions
    with the columns of the output from the first.  This function orchestrates the
    invocation of 1D convolutions.
    '''
    result_conv_along_x = _convolution_1D(input_array, operator)
    result_conv_along_y = _convolution_1D(result_conv_along_x.transpose(), operator)
    final_result = result_conv_along_y.transpose()
    return final_result

def _line_intersection(line1, line2):                  ### needed for interactive extraction of
                                                       ### of an image portion by using mouse clicks
    '''                                                                                                  
    Each line is defined by a 4-tuple, with its first two elements defining the                          
    coordinates of the first endpoint and the two elements defining the coordinates                      
    of the second endpoint.  This function defines a predicate that tells us whether                     
    or not two given line segments intersect.                                                            
    '''
    line1_endpoint1_x = line1[0]
    line1_endpoint1_y = line1[1]
    line1_endpoint2_x = line1[2]
    line1_endpoint2_y = line1[3]
    line2_endpoint1_x = line2[0] + 0.5
    line2_endpoint1_y = line2[1] + 0.5
    line2_endpoint2_x = line2[2] + 0.5
    line2_endpoint2_y = line2[3] + 0.5
    if max([line1_endpoint1_x,line1_endpoint2_x]) <= min([line2_endpoint1_x,line2_endpoint2_x]):
        return 0
    elif max([line1_endpoint1_y,line1_endpoint2_y]) <= min([line2_endpoint1_y,line2_endpoint2_y]):
        return 0
    elif max([line2_endpoint1_x,line2_endpoint2_x]) <= min([line1_endpoint1_x,line1_endpoint2_x]):
        return 0
    elif max([line2_endpoint1_y,line2_endpoint2_y]) <= min([line1_endpoint1_y,line1_endpoint2_y]):
        return 0
    # Use homogeneous representation of lines:      
    hom_rep_line1 = _cross_product((line1_endpoint1_x,line1_endpoint1_y,1),(line1_endpoint2_x,line1_endpoint2_y,1))
    hom_rep_line2 = _cross_product((line2_endpoint1_x,line2_endpoint1_y,1),(line2_endpoint2_x,line2_endpoint2_y,1))
    hom_intersection = _cross_product(hom_rep_line1, hom_rep_line2)
    if hom_intersection[2] == 0:
        return 0
    intersection_x = hom_intersection[0] / (hom_intersection[2] * 1.0)
    intersection_y = hom_intersection[1] / (hom_intersection[2] * 1.0)
    if intersection_x >= line1_endpoint1_x and intersection_x <= line1_endpoint2_x and \
                                      intersection_y >= line1_endpoint1_y and intersection_y <= line1_endpoint2_y:
        return 1
    return 0

def _cross_product(vector1, vector2):             ### needed by the above line intersection tester
    '''
    Returns the vector cross product of two triples
    '''
    (a1,b1,c1) = vector1
    (a2,b2,c2) = vector2
    p1 = b1*c2 - b2*c1
    p2 = a2*c1 - a1*c2
    p3 = a1*b2 - a2*b1
    return (p1,p2,p3)

def ctrl_c_handler( signum, frame ):             
    print("Killed by Ctrl C")                       
    os.kill( os.getpid(), signal.SIGKILL )       
signal.signal( signal.SIGINT, ctrl_c_handler )   


#______________________________  RegionProposalGenerator Class Definition  ________________________________

class RegionProposalGenerator(object):

    # Class variables: 
    region_mark_coords = {}
    drawEnable = startX = startY = 0
    canvas = None

    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''RegionProposalGenerator constructor can only be called with keyword arguments for 
                      the following keywords: dataroot_train, dataroot_test, image_size, data_image, 
                      binary_or_gray_or_color, kay, image_size_reduction_factor, max_iterations, sigma, 
                      image_normalization_required, momentum, min_size_for_graph_based_blobs, 
                      max_num_blobs_expected, path_saved_RPN_model, path_saved_single_instance_detector_model,
                      path_saved_yolo_model, learning_rate, epochs, batch_size, classes, debug_train, 
                      debug_test, use_gpu, color_homogeneity_thresh, gray_var_thresh, texture_homogeneity_thresh, 
                      yolo_interval, and debug''')
        dataroot_train = dataroot_test = data_image = sigma = image_size_reduction_factor = kay = momentum = None
        learning_rate = epochs = min_size_for_graph_based_blobs = max_num_blobs_expected = path_saved_RPN_model = None
        path_saved_single_instance_detector_model = batch_size = use_gpu = binary_or_gray_or_color =  max_iterations = None
        image_normalization_required = classes = debug_train = color_homogeneity_thresh = gray_var_thresh = None
        image_size = texture_homogeneity_thresh = debug = debug_test = path_saved_yolo_model = yolo_interval = None

        if 'dataroot_train' in kwargs                :   dataroot_train = kwargs.pop('dataroot_train')
        if 'dataroot_test' in kwargs                 :   dataroot_test = kwargs.pop('dataroot_test')
        if 'image_size' in kwargs                    :   image_size = kwargs.pop('image_size')
        if 'path_saved_RPN_model' in kwargs          :   path_saved_RPN_model = kwargs.pop('path_saved_RPN_model')
        if 'path_saved_single_instance_detector_model' in kwargs   :   
                 path_saved_single_instance_detector_model = kwargs.pop('path_saved_single_instance_detector_model')
        if 'path_saved_yolo_model' in kwargs         :   path_saved_yolo_model = kwargs.pop('path_saved_yolo_model')
        if 'yolo_interval' in kwargs                 :   yolo_interval = kwargs.pop('yolo_interval')

        if 'momentum' in kwargs                      :   momentum = kwargs.pop('momentum')
        if 'learning_rate' in kwargs                 :   learning_rate = kwargs.pop('learning_rate')
        if 'epochs' in kwargs                        :   epochs = kwargs.pop('epochs')
        if 'batch_size' in kwargs                    :   batch_size = kwargs.pop('batch_size')
        if 'classes' in kwargs                       :   classes = kwargs.pop('classes')
        if 'debug_train' in kwargs                   :   debug_train = kwargs.pop('debug_train')
        if 'debug_test' in kwargs                    :   debug_test = kwargs.pop('debug_test')
        if 'use_gpu' in kwargs                       :   use_gpu = kwargs.pop('use_gpu')
        if 'data_image' in kwargs                    :   data_image = kwargs.pop('data_image')
        if 'sigma' in kwargs                         :   sigma = kwargs.pop('sigma')
        if 'kay' in kwargs                           :   kay = kwargs.pop('kay')
        if 'image_size_reduction_factor' in kwargs   :   image_size_reduction_factor = kwargs.pop('image_size_reduction_factor')
        if 'binary_or_gray_or_color' in kwargs       :   binary_or_gray_or_color = kwargs.pop('binary_or_gray_or_color')
        if 'image_normalization_required' in kwargs  :   image_normalization_required = kwargs.pop('image_normalization_required')
        if 'max_iterations' in kwargs                :   max_iterations=kwargs.pop('max_iterations')
        if 'color_homogeneity_thresh' in kwargs      :   color_homogeneity_thresh = kwargs.pop('color_homogeneity_thresh')
        if 'gray_var_thresh' in kwargs               :    gray_var_thresh = kwargs.pop('gray_var_thresh')
        if 'texture_homogeneity_thresh' in kwargs    :   texture_homogeneity_thresh = kwargs.pop('texture_homogeneity_thresh')
        if 'min_size_for_graph_based_blobs' in kwargs :  min_size_for_graph_based_blobs = kwargs.pop('min_size_for_graph_based_blobs')
        if 'max_num_blobs_expected' in kwargs        :  max_num_blobs_expected = kwargs.pop('max_num_blobs_expected')
        if 'debug' in kwargs                         :   debug = kwargs.pop('debug') 
        if len(kwargs) != 0: raise ValueError('''You have provided unrecognizable keyword args''')
        if dataroot_train:
            self.dataroot_train = dataroot_train
        if dataroot_test:
            self.dataroot_test = dataroot_test
        if image_size:   
            self.image_size = image_size      
        if  path_saved_RPN_model:
            self.path_saved_RPN_model = path_saved_RPN_model
        if  path_saved_single_instance_detector_model:
            self.path_saved_single_instance_detector_model = path_saved_single_instance_detector_model
        if  path_saved_yolo_model:
            self.path_saved_yolo_model = path_saved_yolo_model
        if yolo_interval:
            self.yolo_interval = yolo_interval
        if classes:
            self.class_labels = classes
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 1e-6
        if momentum:
            self.momentum = momentum
        if epochs:
            self.epochs = epochs
        if batch_size:
            self.batch_size = batch_size
        if use_gpu is not None:
            self.use_gpu = use_gpu
            if use_gpu is True:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda:0")
                else:
                    raise Exception("You requested GPU support, but there's no GPU on this machine")
            else:
                self.device = torch.device("cpu")
        if debug_train:                             
            self.debug_train = debug_train
        else:
            self.debug_train = 0
        if debug_test:                             
            self.debug_test = debug_test
        else:
            self.debug_test = 0
        if data_image: 
            self.data_im_name = data_image
            self.data_im =  Image.open(data_image)
            self.original_im = Image.open(data_image)
        if binary_or_gray_or_color:
            self.binary_or_gray_or_color = binary_or_gray_or_color
        if sigma is not None: 
            self.sigma = sigma
        else:
            self.sigma = 0
        if kay is not None:   self.kay = kay
        if image_size_reduction_factor is not None:
            self.image_size_reduction_factor = image_size_reduction_factor
        else:
            self.image_size_reduction_factor = 1
        if image_normalization_required is not None:
            self.image_normalization_required = image_normalization_required
        else:
            self.image_normalization_required = False
        if max_iterations is not None:
            self.max_iterations = max_iterations
        else:
            self.max_iterations = 40
        if color_homogeneity_thresh is not None:
            self.color_homogeneity_thresh = color_homogeneity_thresh
        if gray_var_thresh is not None:
            self.gray_var_thresh = gray_var_thresh
        if texture_homogeneity_thresh is not None:
            self.texture_homogeneity_thresh = texture_homogeneity_thresh
        if min_size_for_graph_based_blobs is not None:
            self.min_size_for_graph_based_blobs = min_size_for_graph_based_blobs
        if max_num_blobs_expected is not None:
            self.max_num_blobs_expected = max_num_blobs_expected
        self.image_portion_delineation_coords = []
        if debug:                             
            self.debug = debug
        else:
            self.debug = 0
        self.iterations_used = 0


    ###%%%
    #######################################################################################################
    ########################  Start Definition of Inner Class PurdueDrEvalDataset  ########################
    class PurdueDrEvalDataset(torch.utils.data.Dataset):        
        """
        This is the dataset to use if you are experimenting with single-instance object
        detection.  The dataset contains three kinds of objects in its images:
        Dr. Eval, and two "objects" in his neighborhood: a house and a watertower.
        Each 128x128 image in the dataset contains one of these objects after it is
        randomly scaled and colored and substantial structured noise in addition to
        20% Gaussian noise.  Examples of these images are shown in Week 9 lecture
        material in Purdue's Deep Learning class.

        In order to understand the implementation of the dataloader for the Dr Eval
        dataset for single-instance-based object detection, note that the top-level
        directory for the dataset is organized as follows:

                                          dataroot
                                             |
                                             |
               ______________________________________________________________________
              |           |           |              |               |               | 
              |           |           |              |               |               |
          Dr_Eval       house      watertower    mask_Dr_Eval    mask_house     mask_watertower
              |           |           |              |               |               |
              |           |           |              |               |               |
            images      images      images      binary images    binary images   binary images
        

        As you can see, the three main image directories are Dr_Eval, house, and
        watertower. For each image in each of these directories, the mask for the
        object of interest is supplied in the corresponding directory whose name
        carries the prefix 'mask'.

        For example, if you have an image named 29.jpg in the Dr_Eval directory, you
        will have an image of the same name in the mask_Dr_Eval directory that will
        just be the mask for the Dr_Eval object in the former image

        As you can see, the dataset does not directly provide the bounding boxes for
        object localization.  So the implementation of the __getitem__() function in
        the dataloader must include code that calculates the bounding boxes from the
        masks.  This you can see in the definition of the dataloader shown below.

        Since this is a ``non-standard'' organization of the of data, the dataloader
        must also provide for the indexing of the images so that they can be subject
        to a fresh randomization that is carried out by PyTorch's
        torch.utils.data.DataLoader class for each epoch of training.  The
        index_dataset() function is provided for that purpose.

        After the dataset is downloaded for the first time, the index_dataset()
        function stores away the information as a PyTorch ``.pt'' file so that it can
        be downloaded almost instantaneously at subsequent attempts.

        One final note about the dataset: Under the hood, the dataset consists of the
        pathnames to the image files --- and NOT the images themselves.  It is the
        job of the multi-threaded ``workers'' provided by torch.utils.data.DataLoader
        to actually download the images from those pathnames.
        """
        def __init__(self, rpg, train_or_test, dataroot_train=None, dataroot_test=None, transform=None):
            super(RegionProposalGenerator.PurdueDrEvalDataset, self).__init__()
            self.rpg = rpg
            self.train_or_test = train_or_test
            self.dataroot_train = dataroot_train
            self.dataroot_test  = dataroot_test
            self.database_train = {}
            self.database_test = {}
            self.dataset_size_train = None
            self.dataset_size_test = None
            if train_or_test == 'train':
                self.training_dataset = self.index_dataset()
            if train_or_test == 'test':
                self.testing_dataset = self.index_dataset()
            self.class_labels = None

        def index_dataset(self):
            if self.train_or_test == 'train':
                dataroot = self.dataroot_train
            elif self.train_or_test == 'test': 
                dataroot = self.dataroot_test
            entry_index = 0
            if self.train_or_test == 'train' and dataroot == self.dataroot_train:
                if '10000' in self.dataroot_train and os.path.exists("torch_saved_Purdue_Dr_Eval_dataset_train_10000.pt"):
                    print("\nLoading training data from torch saved file")
                    self.database_train = torch.load("torch_saved_Purdue_Dr_Eval_dataset_train_10000.pt")
                    self.dataset_size_train =  len(self.database_train)
                else: 
                    print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                          """the dataset for this script. First time loading could take\n"""
                          """up to 3 minutes.  Any subsequent attempts will only take\n"""
                          """a few seconds.\n\n\n""")
                    if os.path.exists(dataroot):      
                        files = glob.glob(dataroot + "/*")  
                        files = [os.path.split(file)[1] for file in files]
                        class_names = sorted([file for file in files if not file.startswith("mask")])
                        if self.train_or_test == 'train':
                            self.class_labels = class_names
                        image_label_dict = {class_names[i] : i for i in range(len(class_names))}
                        for image_class in class_names:
                            image_names = glob.glob(dataroot + image_class + "/*")
                            for image_name in image_names:
                                image_real_name = os.path.split(image_name)[-1]
                                mask_name = dataroot + "mask_" + image_class + "/" + image_real_name
                                if self.train_or_test == 'train':
                                    self.database_train[entry_index] = [image_label_dict[image_class], image_name, mask_name]
                                elif self.train_or_test == 'test':
                                    self.database_test[entry_index] = [image_label_dict[image_class], image_name, mask_name]
                                entry_index += 1
                        if self.train_or_test == 'train':           
                            all_training_images = list(self.database_train.values())
                            random.shuffle(all_training_images)
                            self.database_train = {i : all_training_images[i] for i in range(len(all_training_images))}
                            torch.save(self.database_train, "torch_saved_Purdue_Dr_Eval_dataset_train_10000.pt")
                            self.dataset_size_train = entry_index
                        else:
                            all_testing_images = list(self.database_test.values())
                            random.shuffle(all_testing_images)
                            self.database_test = {i : all_testing_images[i] for i in range(len(all_testing_images))}
                            self.dataset_size_test = entry_index
            else:
                if os.path.exists(dataroot):      
                    files = glob.glob(dataroot + "/*")  
                    files = [os.path.split(file)[1] for file in files]
                    class_names = sorted([file for file in files if not file.startswith("mask")])
                    image_label_dict = {class_names[i] : i for i in range(len(class_names))}
                    for image_class in class_names:
                        image_names = glob.glob(dataroot + image_class + "/*")
                        for image_name in image_names:
                            image_real_name = os.path.split(image_name)[-1]
                            mask_name = dataroot + "mask_" + image_class + "/" + image_real_name
                            if self.train_or_test == 'train':
                                self.database_train[entry_index] = [image_label_dict[image_class], image_name, mask_name]
                            elif self.train_or_test == 'test':
                                self.database_test[entry_index] = [image_label_dict[image_class], image_name, mask_name]
                            entry_index += 1
                    if self.train_or_test == 'train':
                        self.dataset_size_train = entry_index
                    if self.train_or_test == 'test':
                        self.dataset_size_test = entry_index
                    if self.train_or_test == 'train':           
                        all_training_images = self.database_train.values()
                        random.shuffle(all_training_images)
                        self.database_train = {i : all_training_images[i] for i in range(len(all_training_images))}
                        torch.save(self.database_train, "torch_saved_Purdue_Dr_Eval_dataset_train_10000.pt")
                        self.dataset_size_train = entry_index
                    else:
                        all_testing_images = list(self.database_test.values())
                        random.shuffle(all_testing_images)
                        self.database_test = {i : all_testing_images[i] for i in range(len(all_testing_images))}

        def __len__(self):
            if self.train_or_test == 'train':
                return self.dataset_size_train
            elif self.train_or_test == 'test':
                return self.dataset_size_test

        def __getitem__(self, idx):
            if self.train_or_test == 'train':       
                image_label, image_name, mask_name = self.database_train[idx]
            elif self.train_or_test == 'test':
                image_label, image_name, mask_name = self.database_test[idx]
            im = Image.open(image_name)
            mask = Image.open(mask_name)
            mask_data = mask.getdata()
            non_zero_pixels = []                                                                                   
            for k,pixel_val in enumerate(mask_data): 
                x = k % self.rpg.image_size[1]
                y = k // self.rpg.image_size[0]
                if pixel_val != 0:     
                    non_zero_pixels.append((x,y)) 
            ## x-coord increases to the left and y-coord increases going downward;  origin at upper-left
            x_min = min([pixel[0] for pixel in non_zero_pixels])
            x_max = max([pixel[0] for pixel in non_zero_pixels])
            y_min = min([pixel[1] for pixel in non_zero_pixels])
            y_max = max([pixel[1] for pixel in non_zero_pixels])
            bbox = [x_min,y_min,x_max,y_max]
            im_tensor = tvt.ToTensor()(im)
            mask_tensor = tvt.ToTensor()(mask)
            bbox_tensor = torch.tensor(bbox, dtype=torch.float)
            return im_tensor,mask_tensor,bbox_tensor,image_label


    ###%%%
    #######################################################################################################
    ######################  Start Definition of Inner Class PurdueDrEvalMultiDataset  #####################
    class PurdueDrEvalMultiDataset(torch.utils.data.Dataset):        
        """
        This is the dataset to use if you are experimenting with multi-instance object
        detection.  As with the previous dataset, it contains three kinds of objects
        in its images: Dr. Eval, and two "objects" in his neighborhood: a house and a
        watertower.  Each 128x128 image in the dataset contains up to 5 instances of
        these objects. The instances are randomly scaled and colored and exact number
        of instances in each image is also chosen randomly. Subsequently, background
        clutter is added to the images --- these are again randomly chosen
        shapes. The number of clutter objects is also chosen randomly but cannot
        exceed 10.  In addition to the structured clutter, I add 20% Gaussian noise
        to each image.  Examples of these images are shown in Week 9 lecture material
        in Purdue's Deep Learning class.

        On account of the much richer structure of the image annotations, this
        dataset is organized very differently from the previous one:


                                          dataroot
                                             |
                                             |
                                 ___________________________
                                |                           |
                                |                           |
                           annotations.p                  images


        Since each image is allowed to contain instances of the three different types
        of "meaningful" objects, it is not possible to organize the images on the
        basis of what they contain.

        As for the annotations, the annotation for each 128x128 image is a dictionary
        that contains information related to all the object instances in the image. Here
        is an example of the annotation for an image that has three instances in it:

            annotation:  {'filename': None, 
                          'num_objects': 3, 
                          'bboxes': {0: (67, 72, 83, 118), 
                                     1: (65, 2, 93, 26), 
                                     2: (16, 68, 53, 122), 
                                     3: None, 
                                     4: None}, 
                          'bbox_labels': {0: 'Dr_Eval', 
                                          1: 'house', 
                                          2: 'watertower', 
                                          3: None, 
                                          4: None}, 
                          'seg_masks': {0: <PIL.Image.Image image mode=1 size=128x128 at 0x7F5A06C838E0>, 
                                        1: <PIL.Image.Image image mode=1 size=128x128 at 0x7F5A06C837F0>, 
                                        2: <PIL.Image.Image image mode=1 size=128x128 at 0x7F5A06C838B0>, 
                                        3: None, 
                                        4: None}
                         }

        The annotations for the individual images are stored in a global Python
        dictionary called 'all_annotations' whose keys consist of the pathnames to
        the individual image files and the values the annotations dict for the
        corresponding images.  The filename shown above in the keystroke diagram,
        'annotations.p' is what you get by calling 'pickle.dump()' on the
        'all_annotations' dictionary.
        """

        def __init__(self, rpg, train_or_test, dataroot_train=None, dataroot_test=None, transform=None):
            super(RegionProposalGenerator.PurdueDrEvalMultiDataset, self).__init__()
            self.rpg = rpg
            self.train_or_test = train_or_test
            self.dataroot_train = dataroot_train
            self.dataroot_test  = dataroot_test
            self.database_train = {}
            self.database_test = {}
            self.dataset_size_train = None
            self.dataset_size_test = None
            if train_or_test == 'train':
                self.training_dataset = self.index_dataset()
            if train_or_test == 'test':
                self.testing_dataset = self.index_dataset()
            self.class_labels = None

        def index_dataset(self):
            if self.train_or_test == 'train':
                dataroot = self.dataroot_train
            elif self.train_or_test == 'test': 
                dataroot = self.dataroot_test
            if self.train_or_test == 'train' and dataroot == self.dataroot_train:
                if '10000' in self.dataroot_train and os.path.exists("torch_saved_Purdue_Dr_Eval_multi_dataset_train_10000.pt"):
                    print("\nLoading training data from torch saved file")
                    self.database_train = torch.load("torch_saved_Purdue_Dr_Eval_multi_dataset_train_10000.pt")
                    self.dataset_size_train =  len(self.database_train)
                else: 
                    print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                          """the dataset for this script. First time loading could take\n"""
                          """up to 3 minutes.  Any subsequent attempts will only take\n"""
                          """a few seconds.\n\n\n""")
                    if os.path.exists(dataroot):      
                        all_annotations = pickle.load( open( dataroot + '/annotations.p', 'rb') )
                        all_image_paths = sorted(glob.glob(dataroot + "images/*"))
                        all_image_names = [os.path.split(filename)[1] for filename in all_image_paths]
                        for idx,image_name in enumerate(all_image_names):        
                            annotation = all_annotations[image_name]
                            image_path = dataroot + "images/" + image_name
                            self.database_train[idx] = [image_path, annotation]
                        all_training_images = list(self.database_train.values())
                        random.shuffle(all_training_images)
                        self.database_train = {i : all_training_images[i] for i in range(len(all_training_images))}
                        torch.save(self.database_train, "torch_saved_Purdue_Dr_Eval_multi_dataset_train_10000.pt")
                        self.dataset_size_train = len(all_training_images)
            elif self.train_or_test == 'test' and dataroot == self.dataroot_test:
                if os.path.exists(dataroot):      
                    all_annotations = pickle.load( open( dataroot + '/annotations.p', 'rb') )
                    all_image_paths = sorted(glob.glob(dataroot + "images/*"))
                    all_image_names = [os.path.split(filename)[1] for filename in all_image_paths]                    
                    for idx,image_name in enumerate(all_image_names):        
                        annotation = all_annotations[image_name]
                        image_path = dataroot + "images/" + image_name
                        self.database_test[idx] =  [image_path, annotation]
                    all_testing_images = list(self.database_test.values())
                    random.shuffle(all_testing_images)
                    self.database_test = {i : all_testing_images[i] for i in range(len(all_testing_images))}
                    self.dataset_size_test = len(all_testing_images)

        def __len__(self):
            if self.train_or_test == 'train':
                return self.dataset_size_train
            elif self.train_or_test == 'test':
                return self.dataset_size_test

        def __getitem__(self, idx):
            if self.train_or_test == 'train':       
                image_path, annotation = self.database_train[idx]
            elif self.train_or_test == 'test':
                image_path, annotation = self.database_test[idx]
            im = Image.open(image_path)
            im_tensor = tvt.ToTensor()(im)
            seg_mask_tensor = torch.zeros(5,128,128)
            bbox_tensor     = torch.zeros(5,4, dtype=torch.uint8)
            bbox_label_tensor    = torch.zeros(5, dtype=torch.uint8) + 13
            num_objects_in_image = annotation['num_objects']
            obj_class_labels = sorted(self.rpg.class_labels)
            self.obj_class_label_dict = {obj_class_labels[i] : i for i in range(len(obj_class_labels))}
            for i in range(num_objects_in_image):
                seg_mask = annotation['seg_masks'][i]
                bbox     = annotation['bboxes'][i]
                label    = annotation['bbox_labels'][i]
                bbox_label_tensor[i] = self.obj_class_label_dict[label]
                seg_mask_arr = np.array(seg_mask)
                seg_mask_tensor[i] = torch.from_numpy(seg_mask_arr)
                bbox_tensor[i] = torch.LongTensor(bbox)      
            return im_tensor, seg_mask_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image


    ###%%%
    #######################################################################################################
    ########################  Start Definition of Inner Class SingleInstanceDetector  #####################

    class SingleInstanceDetector(nn.Module):             
        """
        This class demonstrates single-instance object detection on the images in the
        PurdueDrEvalDataset dataset.  Although these image are complex, in the sense
        that each image contains multiple clutter objects in addition to random
        noise, nonetheless we know that each image contains only a single meaningful
        object instance.  The LOADnet network used for detection is adaptation of the
        the LOADnet2 network from DLStudio to the case of 128x128 sized input images.
        The LOADnet network uses the SkipBlock as a building-block element for
        dealing the problems caused by vanishing gradients.
        """
        def __init__(self, rpg):
            super(RegionProposalGenerator.SingleInstanceDetector, self).__init__()
            self.rpg = rpg
            self.dataserver_train = None
            self.dataserver_test =  None
            self.train_dataloader = None
            self.test_dataloader = None

        def show_sample_images_from_dataset(self, rpg):
            data = next(iter(self.train_dataloader))    
            real_batch = data[0]
            first_im = real_batch[0]
            self.rpg.display_tensor_as_image(torchvision.utils.make_grid(real_batch, padding=2, pad_value=1, normalize=True))

        def set_dataloaders(self, train=False, test=False):
            if train:
                self.dataserver_train = RegionProposalGenerator.PurdueDrEvalDataset(self.rpg, 
                                                        "train", dataroot_train=self.rpg.dataroot_train)
                self.train_dataloader = torch.utils.data.DataLoader(self.dataserver_train, 
                                                       self.rpg.batch_size, shuffle=True, num_workers=4)
            if test:
                self.dataserver_test = RegionProposalGenerator.PurdueDrEvalDataset(self.rpg, 
                                                           "test", dataroot_test=self.rpg.dataroot_test)
                self.test_dataloader = torch.utils.data.DataLoader(self.dataserver_test, 
                                                      self.rpg.batch_size, shuffle=False, num_workers=4)


        def check_dataloader(self, how_many_batches_to_show, train=False, test=False):
            if train:      
                dataloader = self.train_dataloader
            if test:
                dataloader = self.test_dataloader
            for i, data in enumerate(dataloader): 
                if i >= how_many_batches_to_show:  
                    break
                im_tensor,mask_tensor,bbox_tensor, image_label = data
                logger = logging.getLogger()
                old_level = logger.level
                logger.setLevel(100)
                plt.figure(figsize=[15,4])
                plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                plt.show()
                plt.figure(figsize=[15,4])
                plt.imshow(np.transpose(torchvision.utils.make_grid(mask_tensor, normalize=True,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                plt.show()
                logger.setLevel(old_level)
                print("\n\nbbox tensor for batch:")
                print(bbox_tensor)
                print("\n\nimage labels for batch: ", image_label)


        class SkipBlock(nn.Module):
            """
            This is a building-block class that I have used in several networks
            """
            def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                super(RegionProposalGenerator.SingleInstanceDetector.SkipBlock, self).__init__()
                self.downsample = downsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convo1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                norm_layer1 = nn.BatchNorm2d
                norm_layer2 = nn.BatchNorm2d
                self.bn1 = norm_layer1(out_ch)
                self.bn2 = norm_layer2(out_ch)
                if downsample:
                    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
            def forward(self, x):
                identity = x                                     
                out = self.convo1(x)                              
                out = self.bn1(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo2(out)                              
                    out = self.bn2(out)                              
                    out = torch.nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
                return out

        class LOADnet(nn.Module):
            """
            The acronym 'LOAD' stands for 'LOcalization And Detection'.
            """ 
            def __init__(self, skip_connections=True, depth=8):
                super(RegionProposalGenerator.SingleInstanceDetector.LOADnet, self).__init__()
                if depth not in [8,10,12,14,16]:
                    sys.exit("LOADnet has only been tested for 'depth' values 8, 10, 12, 14, and 16")
                self.depth = depth // 2
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.bn1  = nn.BatchNorm2d(64)
                self.bn2  = nn.BatchNorm2d(128)
                self.bn3  = nn.BatchNorm2d(256)
                self.skip64_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64_arr.append(RegionProposalGenerator.SingleInstanceDetector.SkipBlock(64, 
                                                                                 64,skip_connections=skip_connections))
                self.skip64ds = RegionProposalGenerator.SingleInstanceDetector.SkipBlock(64,64,downsample=True, 
                                                                                     skip_connections=skip_connections)
                self.skip64to128 = RegionProposalGenerator.SingleInstanceDetector.SkipBlock(64, 128, 
                                                                                    skip_connections=skip_connections )
                self.skip128_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128_arr.append(RegionProposalGenerator.SingleInstanceDetector.SkipBlock(128,128,
                                                                                    skip_connections=skip_connections))
                self.skip128ds = RegionProposalGenerator.SingleInstanceDetector.SkipBlock(128,128,
                                                                    downsample=True, skip_connections=skip_connections)
                self.skip128to256 = RegionProposalGenerator.SingleInstanceDetector.SkipBlock(128, 256, 
                                                                                    skip_connections=skip_connections )
                self.skip256_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip256_arr.append(RegionProposalGenerator.SingleInstanceDetector.SkipBlock(256,256,
                                                                                    skip_connections=skip_connections))
                self.skip256ds = RegionProposalGenerator.SingleInstanceDetector.SkipBlock(256,256,
                                                                    downsample=True, skip_connections=skip_connections)
                self.fc1 =  nn.Linear(8192, 1000)
                self.fc2 =  nn.Linear(1000, 3)
                ##  the rest of this __init__() is for regression
                self.conv_seqn = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2,2),
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2,2)
                )
                self.fc_seqn = nn.Sequential(
                    nn.Linear(65536, 1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 4)
                )

            def forward(self, x):
                x = self.pool(torch.nn.functional.relu(self.conv1(x)))          
                xR = x.clone()
                ## The labeling section:
                x1 = nn.MaxPool2d(2,2)(torch.nn.functional.relu(self.conv2(x)))       
                for i,skip64 in enumerate(self.skip64_arr[:self.depth//4]):
                    x1 = skip64(x1)                
                x1 = self.skip64ds(x1)
                for i,skip64 in enumerate(self.skip64_arr[self.depth//4:]):
                    x1 = skip64(x1)                
                x1 = self.bn1(x1)
                x1 = self.skip64to128(x1)
                for i,skip128 in enumerate(self.skip128_arr[:self.depth//4]):
                    x1 = skip128(x1)                
                x1 = self.bn2(x1)
                x1 = self.skip128ds(x1)
                x1 = x1.view(-1, 8192 )
                x1 = torch.nn.functional.relu(self.fc1(x1))
                x1 = self.fc2(x1)                                  
                ## for bounding box regression:
                x2 = self.conv_seqn(xR)
                x2 = x2.view(x.size(0), -1)
                x2 = self.fc_seqn(x2)          
                return x1,x2
    
        def run_code_for_training_single_instance_detector(self, net, display_images=False):        
            filename_for_out1 = "performance_numbers_" + str(self.rpg.epochs) + "label.txt"
            filename_for_out2 = "performance_numbers_" + str(self.rpg.epochs) + "regres.txt"
            FILE1 = open(filename_for_out1, 'w')
            FILE2 = open(filename_for_out2, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.rpg.device)
            criterion1 = nn.CrossEntropyLoss()
            criterion2 = nn.MSELoss()
            optimizer = optim.SGD(net.parameters(), lr=self.rpg.learning_rate, momentum=self.rpg.momentum)
            print("\n\nStarting training loop...\n\n")
            start_time = time.perf_counter()
            labeling_loss_tally = []
            regression_loss_tally = []
            elapsed_time = 0.0
            for epoch in range(self.rpg.epochs):  
                print("")
                running_loss_labeling = 0.0
                running_loss_regression = 0.0       
                for i, data in enumerate(self.train_dataloader):
                    gt_too_small = False
                    im_tensor,mask_tensor,bbox_tensor, image_label = data
                    if i % 500 == 499:
                        current_time = time.perf_counter()
                        elapsed_time = current_time - start_time 
                        print("\n\n[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]   Ground Truth:     " % 
                                (epoch+1, self.rpg.epochs, i+1, elapsed_time) + 
                                ' '.join('%15s' % self.rpg.class_labels[image_label[j].item()] 
                                for j in range(self.rpg.batch_size)))
                    im_tensor = im_tensor.to(self.rpg.device)
                    image_label = image_label.to(self.rpg.device)
                    bbox_tensor = bbox_tensor.to(self.rpg.device)
                    optimizer.zero_grad()
                    outputs = net(im_tensor)
                    outputs_label = outputs[0]
                    bbox_pred = outputs[1]
                    bbox_gt = bbox_tensor
                    if i % 500 == 499:
                        inputs_copy = im_tensor.detach().clone()
                        inputs_copy = inputs_copy.cpu()
                        bbox_pc = bbox_pred.detach().clone()
                        bbox_pc[bbox_pc<0] = 0
                        bbox_pc[bbox_pc>127] = 127
                        bbox_pc[torch.isnan(bbox_pc)] = 0
                        _, predicted = torch.max(outputs_label.data, 1)
                        print("[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]   Predicted Labels: " % 
                         (epoch+1, self.rpg.epochs, i+1, elapsed_time) + 
                         ' '.join('%15s' % self.rpg.class_labels[predicted[j].item()] for j in range(self.rpg.batch_size)))
                        for idx in range(self.rpg.batch_size):
                            i1 = int(bbox_gt[idx][1])
                            i2 = int(bbox_gt[idx][3])
                            j1 = int(bbox_gt[idx][0])
                            j2 = int(bbox_gt[idx][2])
                            k1 = int(bbox_pc[idx][1])
                            k2 = int(bbox_pc[idx][3])
                            l1 = int(bbox_pc[idx][0])
                            l2 = int(bbox_pc[idx][2])
                            print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                            print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                            inputs_copy[idx,0,i1:i2,j1] = 255
                            inputs_copy[idx,0,i1:i2,j2] = 255
                            inputs_copy[idx,0,i1,j1:j2] = 255
                            inputs_copy[idx,0,i2,j1:j2] = 255
                            inputs_copy[idx,2,k1:k2,l1] = 255                      
                            inputs_copy[idx,2,k1:k2,l2] = 255
                            inputs_copy[idx,2,k1,l1:l2] = 255
                            inputs_copy[idx,2,k2,l1:l2] = 255
                    loss_labeling = criterion1(outputs_label, image_label)
                    loss_labeling.backward(retain_graph=True)        
                    loss_regression = criterion2(bbox_pred, bbox_tensor)
                    loss_regression.backward()
                    optimizer.step()
                    running_loss_labeling += loss_labeling.item()    
                    running_loss_regression += loss_regression.item()                
                    if i % 500 == 499:    
                        avg_loss_labeling = running_loss_labeling / float(500)
                        avg_loss_regression = running_loss_regression / float(500)
                        print("\n[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]   loss_labeling: %.3f      loss_regression: %.3f  " % 
                                    (epoch + 1, self.rpg.epochs, i + 1, elapsed_time, avg_loss_labeling, avg_loss_regression))
                        FILE1.write("%.3f\n" % avg_loss_labeling)
                        FILE1.flush()
                        FILE2.write("%.3f\n" % avg_loss_regression)
                        FILE2.flush()
                        labeling_loss_tally.append(avg_loss_labeling)
                        regression_loss_tally.append(avg_loss_regression)
                        running_loss_labeling = 0.0
                        running_loss_regression = 0.0
                        if display_images:
                            logger = logging.getLogger()
                            old_level = logger.level
                            logger.setLevel(100)
                            plt.figure(figsize=[15,4])
                            plt.imshow(np.transpose(torchvision.utils.make_grid(inputs_copy, normalize=False, 
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                            plt.show()
                            logger.setLevel(old_level)
            print("\nFinished Training\n")
            self.save_single_instance_detector_model(net)
            plt.figure(figsize=(10,5))
            plt.title("Labeling Loss vs. Iterations")
            plt.plot(labeling_loss_tally)
            plt.xlabel("iterations")
            plt.ylabel("labeling loss")
            plt.legend()
            plt.savefig("labeling_loss.png")
            plt.show()
            plt.title("regression Loss vs. Iterations")
            plt.plot(regression_loss_tally)
            plt.xlabel("iterations")
            plt.ylabel("regression loss")
            plt.legend()
            plt.savefig("regression_loss.png")
            plt.show()
    
        def save_single_instance_detector_model(self, model):
            '''
            Save the trained single instance detector model to a disk file
            '''
            torch.save(model.state_dict(), self.rpg.path_saved_single_instance_detector_model)

        def run_code_for_testing_single_instance_detector(self, net, display_images=False):
            net.load_state_dict(torch.load(self.rpg.path_saved_single_instance_detector_model))
            correct = 0
            total = 0
            confusion_matrix = torch.zeros(len(self.rpg.class_labels), len(self.rpg.class_labels))
            class_correct = [0] * len(self.rpg.class_labels)
            class_total = [0] * len(self.rpg.class_labels)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    im_tensor,mask_tensor,bbox_tensor, image_label = data
                    image_label = image_label.tolist()
                    if i % 50 == 49:
                        print("\n\n[i=%4d]       Ground Truth: " % (i+1) + 
                         ' '.join('%15s' % self.rpg.class_labels[image_label[j]] for j in range(self.rpg.batch_size)))
                    outputs = classifier_net(im_tensor)
                    outputs_label = outputs[0]
                    bbox_pred = outputs[1]
                    bbox_gt = bbox_tensor
                    _, predicted = torch.max(outputs_label.data, 1)
                    if i % 50 == 49:
                        inputs_copy = im_tensor.detach().clone()
                        inputs_copy = inputs_copy.cpu()
                        bbox_pc = bbox_pred.detach().clone()
                        bbox_pc[bbox_pc<0] = 0
                        bbox_pc[bbox_pc>127] = 127
                        bbox_pc[torch.isnan(bbox_pc)] = 0
                        print("[i=%4d]   Predicted Labels: " % (i+1) + 
                             ' '.join('%15s' % self.rpg.class_labels[predicted[j].item()] for j in range(self.rpg.batch_size)))
                        for idx in range(self.rpg.batch_size):
                            i1 = int(bbox_gt[idx][1])
                            i2 = int(bbox_gt[idx][3])
                            j1 = int(bbox_gt[idx][0])
                            j2 = int(bbox_gt[idx][2])
                            k1 = int(bbox_pc[idx][1])
                            k2 = int(bbox_pc[idx][3])
                            l1 = int(bbox_pc[idx][0])
                            l2 = int(bbox_pc[idx][2])
                            print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                            print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                            inputs_copy[idx,0,i1:i2,j1] = 255
                            inputs_copy[idx,0,i1:i2,j2] = 255
                            inputs_copy[idx,0,i1,j1:j2] = 255
                            inputs_copy[idx,0,i2,j1:j2] = 255
                            inputs_copy[idx,2,k1:k2,l1] = 255                      
                            inputs_copy[idx,2,k1:k2,l2] = 255
                            inputs_copy[idx,2,k1,l1:l2] = 255
                            inputs_copy[idx,2,k2,l1:l2] = 255
                        if display_images:
                            logger = logging.getLogger()
                            old_level = logger.level
                            logger.setLevel(100)
                            plt.figure(figsize=[15,4])
                            plt.imshow(np.transpose(torchvision.utils.make_grid(inputs_copy, normalize=False, 
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                            plt.show()
                            logger.setLevel(old_level)
                    for label,prediction in zip(image_label, predicted):
                        confusion_matrix[label][prediction] += 1
                    total += len(image_label)
                    correct +=  [predicted[ele] == image_label[ele] for ele in range(len(predicted))].count(True)
                    comp = [predicted[ele] == image_label[ele] for ele in range(len(predicted))]

                    for j in range(len(image_label)):
                        label = image_label[j]
                        class_correct[label] += comp[j]
                        class_total[label] += 1
            print("\n")
            for j in range(len(self.rpg.class_labels)):
                print('Prediction accuracy for %5s : %2d %%' % (
              self.rpg.class_labels[j], 100 * class_correct[j] / float(class_total[j])))
            print("\n\n\nOverall accuracy of the network on the 1000 test images: %d %%" % (100 * correct / float(total)))
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                "
            for j in range(len(self.rpg.class_labels)):  
                out_str +=  "%15s" % self.rpg.class_labels[j]   
            print(out_str + "\n")
            for i,label in enumerate(self.rpg.class_labels):
                out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                                 for j in range(len(self.rpg.class_labels))]
                out_percents = ["%.2f" % item.item() for item in out_percents]
                out_str = "%12s:  " % self.rpg.class_labels[i]
                for j in range(len(self.rpg.class_labels)): 
                    out_str +=  "%15s" % out_percents[j]
                print(out_str)
    

    ###%%%
    #######################################################################################################
    ##########################  Start Definition of Inner Class YoloLikeDetector  #########################

    class YoloLikeDetector(nn.Module):             
        """
        The primary purpose of this class is to demonstrate multi-instance object 
        detection with YOLO-like logic.  Toward that end, the following method
        of this class:

            run_code_for_training_multi_instance_detection()

        shows code based on my understanding of how YOLO works for multi-instance
        object detection.  A key parameter of the logic I have presented is in the
        form the variable 'yolo_interval'.  The image gridding that is required is
        based on the value assigned to this variable.  The grid is represented by an
        SxS array of cells where S is the image width divided by yolo_interval. So if
        'yolo_interval=20', you will get a 6x6 grid of cells over the image. [In the
        current version 0 of the implementation, I have not bothered with the bottom
        8 rows and the right-most 8 columns of the image that get left out of the
        area covered by such a grid.]

        Perhaps the most important element of the YOLO logic is defining a set of
        Anchor Boxes for each cell in the SxS grid.  The anchor boxes are
        characterized by their aspect ratios.  By aspect ratio I mean the
        'height/width' characterization of the boxes.  My implementation provides for
        5 anchor boxes for each cell with the following aspect ratios: 1/5, 1/3, 1/1,
        3/1, 5/1.  

        At training time, each instance in the image is assigned to that cell whose
        central pixel is closest to the center of the bounding box for the instance.
        After the cell assignment, the instance is assigned to that anchor box whose
        aspect ratio comes closest to matching the aspect ratio of the instance.

        The assigning of an object instance to a <cell, anchor_box> pair is encoded
        in the form of a '5+C' element long vector where C is the number of classes
        for the object instances.  In our cases, C is 3 for the three classes
        'Dr_Eval', 'house' and 'watertower', there we end up with an 8-element vector
        encoding when we assign an instance to a <cell, anchor_box> pair.  The last C
        elements of the encoding vector can be thought as a one-hot representation of
        the class label for the instance.

        The first five elements of the vector encoding for each anchor box in a cell
        are set as follows: The first element is 1 if an object instance was actually
        assigned to that anchor box. The next two elements are the (x,y)
        displacements of the center of the actual bounding box for the instance
        vis-a-vis the center of the cell. These two displacements are expressed as a
        fraction of the width and the height of the cell.  The next two elements of
        the vector encoding are the actual height and the actual width of the true
        bounding box for the instance in question as a multiple of the cell
        dimension.

        The 8-element vectors for each of the anchor boxes for all the cells in the
        SxS grid are concatenated together to form a large vector for regression.
        Since S is 6 in the implementation shown and we have 5 anchor boxes for each
        cell of the grid, we end up with a 1440 element tensor representation for
        each training image.

        """
        def __init__(self, rpg):
            super(RegionProposalGenerator.YoloLikeDetector, self).__init__()
            self.rpg = rpg
            self.train_dataloader = None
            self.test_dataloader = None

        def show_sample_images_from_dataset(self, rpg):
            data = next(iter(self.train_dataloader))    
            real_batch = data[0]
            first_im = real_batch[0]
            self.rpg.display_tensor_as_image(torchvision.utils.make_grid(real_batch, padding=2, pad_value=1, normalize=True))

        def set_dataloaders(self, train=False, test=False):
            if train:
                dataserver_train = RegionProposalGenerator.PurdueDrEvalMultiDataset(self.rpg, 
                                                       "train", dataroot_train=self.rpg.dataroot_train)
                self.train_dataloader = torch.utils.data.DataLoader(dataserver_train, 
                                                      self.rpg.batch_size, shuffle=True, num_workers=4)
            if test:
                dataserver_test = RegionProposalGenerator.PurdueDrEvalMultiDataset(self.rpg, 
                                                          "test", dataroot_test=self.rpg.dataroot_test)
                self.test_dataloader = torch.utils.data.DataLoader(dataserver_test, 
                                                     self.rpg.batch_size, shuffle=False, num_workers=4)

        def check_dataloader(self, how_many_batches_to_show, train=False, test=False):
            if train:      
                dataloader = self.train_dataloader
            if test:
                dataloader = self.test_dataloader
            for idx, data in enumerate(dataloader): 
                if idx >= how_many_batches_to_show:  
                    break
                im_tensor, seg_mask_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data
                print("\n\nNumber of objects in the batch images: ", num_objects_in_image)
                print("\n\nlabels for the objects found:")
                print(bbox_label_tensor)

                mask_shape = seg_mask_tensor.shape
                logger = logging.getLogger()
                old_level = logger.level
                logger.setLevel(100)
                #  Let's now display the batch images:
                plt.figure(figsize=[15,4])
                plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                plt.show()
                #  Let's now display the batch with JUST the masks:
                composite_mask_tensor = torch.zeros(im_tensor.shape[0], 1,128,128)
                for bdx in range(im_tensor.shape[0]):
                    for i in range(num_objects_in_image[bdx]):
                         composite_mask_tensor[bdx] += seg_mask_tensor[bdx][i]
                plt.figure(figsize=[15,4])
                plt.imshow(np.transpose(torchvision.utils.make_grid(composite_mask_tensor, normalize=True,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                plt.show()
                #  Let's now display the batch and masks in a side-by-side display:
                display_image_and_mask_tensor = torch.zeros(2*im_tensor.shape[0], 3,128,128)
                display_image_and_mask_tensor[:im_tensor.shape[0],:,:,:]  = im_tensor
                display_image_and_mask_tensor[im_tensor.shape[0]:,:,:,:]  = composite_mask_tensor
                plt.figure(figsize=[15,4])
                plt.imshow(np.transpose(torchvision.utils.make_grid(display_image_and_mask_tensor, normalize=False,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                plt.show()
                #  Let's now display the batch with GT bboxes for the objects:
                im_with_bbox_tensor = torch.clone(im_tensor)
                for bdx in range(im_tensor.shape[0]):
                    bboxes_for_image = bbox_tensor[bdx]
                    for i in range(num_objects_in_image[bdx]):
                        ii = bbox_tensor[bdx][i][0].item()
                        ji = bbox_tensor[bdx][i][1].item()
                        ki = bbox_tensor[bdx][i][2].item()
                        li = bbox_tensor[bdx][i][3].item()
                        im_with_bbox_tensor[bdx,:,ji,ii:ki] = 255    
                        im_with_bbox_tensor[bdx,:,li,ii:ki] = 255                
                        im_with_bbox_tensor[bdx,:,ji:li,ii] = 255  
                        im_with_bbox_tensor[bdx,:,ji:li,ki] = 255  
                plt.figure(figsize=[15,4])
                plt.imshow(np.transpose(torchvision.utils.make_grid(im_with_bbox_tensor, normalize=False,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                plt.show()
                #  Let's now display the batch with GT bboxes and the object labels
                im_with_bbox_tensor = torch.clone(im_tensor)
                for bdx in range(im_tensor.shape[0]):
                    labels_for_image = bbox_label_tensor[bdx]
                    bboxes_for_image = bbox_tensor[bdx]
                    for i in range(num_objects_in_image[bdx]):
                        ii = bbox_tensor[bdx][i][0].item()
                        ji = bbox_tensor[bdx][i][1].item()
                        ki = bbox_tensor[bdx][i][2].item()
                        li = bbox_tensor[bdx][i][3].item()
                        im_with_bbox_tensor[bdx,:,ji,ii:ki] = 40    
                        im_with_bbox_tensor[bdx,:,li,ii:ki] = 40                
                        im_with_bbox_tensor[bdx,:,ji:li,ii] = 40  
                        im_with_bbox_tensor[bdx,:,ji:li,ki] = 40  
                        im_pil = tvt.ToPILImage()(im_with_bbox_tensor[bdx]).convert('RGBA')
                        text = Image.new('RGBA', im_pil.size, (255,255,255,0))
                        draw = ImageDraw.Draw(text)
                        horiz = ki-10 if ki>10 else ki
                        vert = li
                        label = self.rpg.class_labels[labels_for_image[i]]
                        label = "wtower" if label == "watertower" else label
                        label = "Dr Eval" if label == "Dr_Eval" else label
                        draw.text( (horiz,vert), label, fill=(255,255,255,200) )
                        im_pil = Image.alpha_composite(im_pil, text)
                        im_with_bbox_tensor[bdx] = tvt.ToTensor()(im_pil.convert('RGB'))

                plt.figure(figsize=[15,4])
                plt.imshow(np.transpose(torchvision.utils.make_grid(im_with_bbox_tensor, normalize=False,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                plt.show()
                logger.setLevel(old_level)


        class SkipBlock(nn.Module):
            """
            This is a building-block class that I have used in several networks
            """
            def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                super(RegionProposalGenerator.YoloLikeDetector.SkipBlock, self).__init__()
                self.downsample = downsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convo1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                norm_layer1 = nn.BatchNorm2d
                norm_layer2 = nn.BatchNorm2d
                self.bn1 = norm_layer1(out_ch)
                self.bn2 = norm_layer2(out_ch)
                if downsample:
                    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
            def forward(self, x):
                identity = x                                     
                out = self.convo1(x)                              
                out = self.bn1(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo2(out)                              
                    out = self.bn2(out)                              
                    out = torch.nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
                return out

        class NetForYolo(nn.Module):
            """
            The YOLO approach to multi-instance detection is based entirely on regression.  As
            was mentioned earlier in the comment block associated with the enclosing
            class, each image is represented by a 1440 element tensor that consists
            of 8-element encodings for each anchor box for every cell in the SxS
            gridding of an image.  The network I show below is a modification of the
            network class LOADnet presented earlier for the case that all we want to
            do is regression.
            """ 
            def __init__(self, skip_connections=True, depth=8):
                super(RegionProposalGenerator.YoloLikeDetector.NetForYolo, self).__init__()
                if depth not in [8,10,12,14,16]:
                    sys.exit("This network has only been tested for 'depth' values 8, 10, 12, 14, and 16")
                self.depth = depth // 2
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.bn1  = nn.BatchNorm2d(64)
                self.bn2  = nn.BatchNorm2d(128)
                self.bn3  = nn.BatchNorm2d(256)
                self.skip64_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64_arr.append(RegionProposalGenerator.YoloLikeDetector.SkipBlock(64, 64,
                                                                                    skip_connections=skip_connections))
                self.skip64ds = RegionProposalGenerator.YoloLikeDetector.SkipBlock(64,64,downsample=True, 
                                                                                     skip_connections=skip_connections)
                self.skip64to128 = RegionProposalGenerator.YoloLikeDetector.SkipBlock(64, 128, 
                                                                                    skip_connections=skip_connections )
                self.skip128_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128_arr.append(RegionProposalGenerator.YoloLikeDetector.SkipBlock(128,128,
                                                                                    skip_connections=skip_connections))
                self.skip128ds = RegionProposalGenerator.YoloLikeDetector.SkipBlock(128,128,
                                                                    downsample=True, skip_connections=skip_connections)
                self.skip128to256 = RegionProposalGenerator.YoloLikeDetector.SkipBlock(128, 256, 
                                                                                    skip_connections=skip_connections )
                self.skip256_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip256_arr.append(RegionProposalGenerator.YoloLikeDetector.SkipBlock(256,256,
                                                                                    skip_connections=skip_connections))
                self.skip256ds = RegionProposalGenerator.YoloLikeDetector.SkipBlock(256,256,
                                                                    downsample=True, skip_connections=skip_connections)
                self.fc_seqn = nn.Sequential(
                    nn.Linear(8192, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, 2048),
                    nn.ReLU(inplace=True),
                    nn.Linear(2048, 1440)
                )

            def forward(self, x):
                x = self.pool(torch.nn.functional.relu(self.conv1(x)))          
                x = nn.MaxPool2d(2,2)(torch.nn.functional.relu(self.conv2(x)))       
                for i,skip64 in enumerate(self.skip64_arr[:self.depth//4]):
                    x = skip64(x)                
                x = self.skip64ds(x)
                for i,skip64 in enumerate(self.skip64_arr[self.depth//4:]):
                    x = skip64(x)                
                x = self.bn1(x)
                x = self.skip64to128(x)
                for i,skip128 in enumerate(self.skip128_arr[:self.depth//4]):
                    x = skip128(x)                
                x = self.bn2(x)
                x = self.skip128ds(x)
                x = x.view(-1, 8192 )
                x = self.fc_seqn(x)
                return x

    
        def run_code_for_training_multi_instance_detection(self, net, display_images=False):        
            yolo_debug = False
            filename_for_out1 = "performance_numbers_" + str(self.rpg.epochs) + "label.txt"
            filename_for_out2 = "performance_numbers_" + str(self.rpg.epochs) + "regres.txt"
            FILE1 = open(filename_for_out1, 'w')
            FILE2 = open(filename_for_out2, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.rpg.device)
            criterion1 = nn.CrossEntropyLoss()
            criterion2 = nn.MSELoss()
            optimizer = optim.SGD(net.parameters(), lr=self.rpg.learning_rate, momentum=self.rpg.momentum)
            print("\n\nStarting training loop...\n\n")
            start_time = time.perf_counter()
            Loss_tally = []
            elapsed_time = 0.0
            yolo_interval = self.rpg.yolo_interval
            num_yolo_cells = (self.rpg.image_size[0] // yolo_interval) * (self.rpg.image_size[1] // yolo_interval)
            num_anchor_boxes =  5    # (height/width)   1/5  1/3  1/1  3/1  5/1
            #  The 8 in the following is the size of the yolo_vector for each
            #  anchor-box in a given cell.  The 8 elements are: [obj_present, bx, by,
            #  bh, bw, c1, c2, c3] where bx and by are the delta diffs between the centers
            #  of the yolo cell and the center of the object bounding box in terms of
            #  a unit for the cell width and cell height.  bh and bw are the height 
            #  and the width of object bounding box in terms of the cell height and width.
            yolo_tensor = torch.zeros( self.rpg.batch_size, num_yolo_cells, num_anchor_boxes, 8 )

            class AnchorBox:
                """
                About the role of the 'adx' constructor parameter:  Recall that our goal is to use
                the annotations for each batch to fill up the 'yolo_tensor' that was defined above.
                For case of 5 anchor boxes per cell, this tensor has the following shape:

                         torch.zeros( self.rpg.batch_size, num_yolo_cells, 5, 8 )

                The index 'adx' shown below tells us which of the 5 dimensions on the third axis
                of the 'yolo_tensor' be RESERVED for an anchor box.  We will reserve the 
                coordinate 0 on the third axis for the "1/1" anchor boxes, the coordinate 1 for
                the "1/3" anchor boxes, and so on.  This coordinate choice is set by 'adx'. 
                """
                #               aspect_ratio top_left_corner  anchor_box height & width   anchor_box index
                def __init__(self,   AR,     tlc,             ab_height,   ab_width,      adx):     
                    self.AR = AR
                    self.tlc = tlc
                    self.ab_height = ab_height
                    self.ab_width = ab_width
                    self.adx = adx
                def __str__(self):
                    return "AnchorBox type (h/w): %s    tlc for yolo cell: %s    anchor-box height: %d     \
                       anchor-box width: %d   adx: %d" % (self.AR, str(self.tlc), self.ab_height, self.ab_width, self.adx)

            for epoch in range(self.rpg.epochs):  
                print("")
                running_loss = 0.0
                for iter, data in enumerate(self.train_dataloader):   
                    im_tensor, seg_mask_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data
                    im_tensor   = im_tensor.to(self.rpg.device)
                    seg_mask_tensor = seg_mask_tensor.to(self.rpg.device)                 
                    bbox_tensor = bbox_tensor.to(self.rpg.device)
                    bbox_label_tensor = bbox_label_tensor.to(self.rpg.device)
                    yolo_tensor = yolo_tensor.to(self.rpg.device)
                    if yolo_debug:
                        logger = logging.getLogger()
                        old_level = logger.level
                        logger.setLevel(100)
                        plt.figure(figsize=[15,4])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.show()
                        logger.setLevel(old_level)
                    cell_height = yolo_interval
                    cell_width = yolo_interval
                    obj_centers = {ibx : {idx : None for idx in range(num_objects_in_image[ibx])} 
                                                                       for ibx in range(im_tensor.shape[0])}
                    ##  2D indexing for access to the anchor boxes is implicit:
                    ##  ALSO:  Aspect Ratio (AR) is the ratio of the height to the width of a bounding box
                    ##         Therefore, in the name "anchor_boxes_1_3", AR is 1/3 when height/width = 1/3
                    num_cells_image_width = self.rpg.image_size[0] // yolo_interval
                    num_cells_image_height = self.rpg.image_size[1] // yolo_interval
                    ## The pair in the second arg to the AnchorBox constructor are the pixel coordinates of the
                    ## top-left corner of the yolo cell in question.

                    anchor_boxes_1_1 = [[AnchorBox("1/1", (i*yolo_interval,j*yolo_interval), yolo_interval, yolo_interval,  0) 
                                                                        for i in range(0,num_cells_image_height)]
                                                                           for j in range(0,num_cells_image_width)]
                    anchor_boxes_1_3 = [[AnchorBox("1/3",(i*yolo_interval,j*yolo_interval), yolo_interval, 3*yolo_interval, 1) 
                                                                        for i in range(0,num_cells_image_height)]
                                                                           for j in range(0,num_cells_image_width)]
                    anchor_boxes_3_1 = [[AnchorBox("3/1",(i*yolo_interval,j*yolo_interval), 3*yolo_interval, yolo_interval, 2)
                                                                        for i in range(0,num_cells_image_height)]
                                                                           for j in range(0,num_cells_image_width)]
                    anchor_boxes_1_5 = [[AnchorBox("1/5",(i*yolo_interval,j*yolo_interval), yolo_interval, 5*yolo_interval, 3)
                                                                        for i in range(0,num_cells_image_height)]
                                                                           for j in range(0,num_cells_image_width)]
                    anchor_boxes_5_1 = [[AnchorBox("5/1",(i*yolo_interval,j*yolo_interval), 5*yolo_interval, yolo_interval, 4) 
                                                                        for i in range(0,num_cells_image_height)]
                                                                           for j in range(0,num_cells_image_width)]
                    for ibx in range(im_tensor.shape[0]):
                        for idx in range(num_objects_in_image[ibx]):
                            ## Note that the bounding-box coordinates are in the (x,y) format, with x-positive going to
                            ## right and the y-positive going down. A bbox is specified by (x_min,y_min,x_max,y_max):

                            height_center_bb =  (bbox_tensor[ibx][idx][1].item() + bbox_tensor[ibx][idx][3].item()) // 2
                            width_center_bb =  (bbox_tensor[ibx][idx][0].item() + bbox_tensor[ibx][idx][2].item()) // 2
                            obj_bb_height = bbox_tensor[ibx][idx][3].item() -  bbox_tensor[ibx][idx][1].item()
                            obj_bb_width = bbox_tensor[ibx][idx][2].item() - bbox_tensor[ibx][idx][0].item()

                            label = self.rpg.class_labels[bbox_label_tensor[ibx][idx].item()]
                            if (obj_bb_height < 4) or (obj_bb_width < 4):
                                continue
                            AR = float(obj_bb_height) / float(obj_bb_width)
                            if yolo_debug:
                                print("\n\n[Image ibx: %d  object: %d]   obj label: %s   obj_bb_height: %d    obj_bb_width: %d" % 
                                                                           (ibx, idx, label, obj_bb_height, obj_bb_width))
                            ## The following cell_row_idx and cell_col_idx are in the (i,j) format with i being the row
                            ## index for a pixel in the image and j being the column index.  The variables
                            ## cell_row_idx and cell_col_idx refer to index of the cell to be used in the yolo_interval
                            ## based gridding of the image.  The anchor boxes are centered at the centers of the grid cells:

                            cell_row_idx =  height_center_bb // yolo_interval          ## for the i coordinate
                            cell_col_idx = width_center_bb // yolo_interval            ## for the j coordinates
                            cell_row_idx = 5 if cell_row_idx > 5 else cell_row_idx
                            cell_col_idx = 5 if cell_col_idx > 5 else cell_col_idx

                            if AR <= 0.2:
                                anchbox = anchor_boxes_1_5[cell_row_idx][cell_col_idx]
                            elif AR <= 0.5:
                                anchbox = anchor_boxes_1_3[cell_row_idx][cell_col_idx]
                            elif AR <= 1.5:
                                anchbox = anchor_boxes_1_1[cell_row_idx][cell_col_idx]
                            elif AR <= 4:
                                anchbox = anchor_boxes_3_1[cell_row_idx][cell_col_idx]
                            elif AR > 4:
                                anchbox = anchor_boxes_5_1[cell_row_idx][cell_col_idx]

                            if yolo_debug:
                                print("iter=%d  cell_row_idx: " % iter, cell_row_idx)
                                print("iter=%d  cell_col_idx: " % iter, cell_col_idx)
                                print("iter=%d  Anchor box used for image: %d and object: %d: " % (iter, ibx,idx), anchbox)
                            # The bh and bw elements in the yolo vector for this object:
                            # bh and bw are measured relative to the size of the grid cell to which the object is
                            # assigned.  For example, bh is the height of the bounding-box divided by the actual
                            # heigh of the grid cell.
                            bh  =  float(obj_bb_height) / float(yolo_interval)
                            bw  =  float(obj_bb_width)  / float(yolo_interval)
                            # You have to be CAREFUL about object center calculation since bounding-box coordinates
                            # are in (x,y) format --- with x-positive going to the right and y-positive going down.
                            obj_center_x =  float(bbox_tensor[ibx][idx][2].item() +  bbox_tensor[ibx][idx][0].item()) / 2.0
                            obj_center_y =  float(bbox_tensor[ibx][idx][3].item() + bbox_tensor[ibx][idx][1].item()) / 2.0

                            # Now you need to switch back from (x,y) format to (i,j) format:
                            yolocell_center_i =  cell_row_idx*yolo_interval + float(yolo_interval) / 2.0
                            yolocell_center_j =  cell_col_idx*yolo_interval + float(yolo_interval) / 2.0
                            if yolo_debug:
                                print("object center located at: obj_center_x: %.3f    obj_center_y: %0.3f" % 
                                                                                          (obj_center_x, obj_center_y))
                                print("yolocell center located at: yolocell_center_j: %.3f   yolocell_center_i: %0.3f" % 
                                                                                (yolocell_center_j, yolocell_center_i))
                            del_x = float(obj_center_x - yolocell_center_j) / yolo_interval
                            del_y = float(obj_center_y - yolocell_center_i) / yolo_interval
                            yolo_vector = [1, del_x, del_y, bh, bw, 0, 0, 0]
                            yolo_vector[5 + bbox_label_tensor[ibx][idx].item()] = 1
                            ## Remember because the coordinate reversal between (x,y) and (i,j) formats, cell_row_idx
                            ## is the index along the horizontal dimension and cell_col_idx is along the vertical dimension.
                            yolo_cell_index =  cell_row_idx * num_cells_image_width  +  cell_col_idx
                            if yolo_debug:
                                print("iter=%d  yolo_vector: " % iter, yolo_vector)
                                print("iter=%d  yolo_cell_index: " % iter, yolo_cell_index)
                            yolo_tensor[ibx, yolo_cell_index, anchbox.adx] = torch.FloatTensor( yolo_vector )
                    if yolo_debug:
                        logger = logging.getLogger()
                        old_level = logger.level
                        logger.setLevel(100)
                        plt.figure(figsize=[15,4])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.show()
                    yolo_tensor_flattened = yolo_tensor.view(im_tensor.shape[0], -1)
                    optimizer.zero_grad()
                    output = net(im_tensor)
                    loss = criterion2(output, yolo_tensor_flattened)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    if iter%500==499:    
                        if display_images:
                            print("\n\n\n")
                        current_time = time.perf_counter()
                        elapsed_time = current_time - start_time 
                        avg_loss = running_loss / float(100)
                        print("[epoch:%d/%d, iter=%4d  elapsed_time=%5d secs]      mean MSE loss: %7.4f" % 
                                                                  (epoch+1,self.rpg.epochs, iter+1, elapsed_time, avg_loss))
                        Loss_tally.append(running_loss)
                        FILE1.write("%.3f\n" % avg_loss)
                        FILE1.flush()
                        running_loss = 0.0
                        if display_images:
                            predictions = output.view(4,36,5,8)
                            for ibx in range(predictions.shape[0]):                             # for each batch image
                                icx_2_best_anchor_box = {ic : None for ic in range(36)}
                                for icx in range(predictions.shape[1]):                         # for each yolo cell
                                    cell_predi = predictions[ibx,icx]               
                                    prev_best = 0
                                    for anchor_bdx in range(cell_predi.shape[0]):
                                        if cell_predi[anchor_bdx][0] > cell_predi[prev_best][0]:
                                            prev_best = anchor_bdx
                                    best_anchor_box_icx = prev_best   
                                    icx_2_best_anchor_box[icx] = best_anchor_box_icx
                                sorted_icx_to_box = sorted(icx_2_best_anchor_box, 
                                           key=lambda x: predictions[ibx,x,icx_2_best_anchor_box[x]][0].item(), reverse=True)
                                retained_cells = sorted_icx_to_box[:5]
                                objects_detected = []
                                for icx in retained_cells:
                                    pred_vec = predictions[ibx,icx, icx_2_best_anchor_box[icx]]
                                    class_labels_predi  = pred_vec[-3:]                        
                                    if torch.all(class_labels_predi < 0.2): 
                                        predicted_class_label = None
                                    else:
                                        best_predicted_class_index = (class_labels_predi == 
                                                                          class_labels_predi.max()).nonzero().squeeze().item()
                                        predicted_class_label = self.rpg.class_labels[best_predicted_class_index]
                                        objects_detected.append(predicted_class_label)
#                                    print("[batch image=%d] most probable object instance detected: ", objects_detected[-1])
#                                    if yolo_debug:
                                print("[batch image=%d]  objects found in descending probability order: " % ibx, objects_detected)
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


        def save_yolo_model(self, model):
            '''
            Save the trained yolo model to a disk file
            '''
            torch.save(model.state_dict(), self.rpg.path_saved_yolo_model)


        def run_code_for_testing_multi_instance_detection(self, net, display_images=False):        
            net.load_state_dict(torch.load(self.rpg.path_saved_yolo_model))
            net = net.to(self.rpg.device)
            yolo_interval = self.rpg.yolo_interval
            num_yolo_cells = (self.rpg.image_size[0] // yolo_interval) * (self.rpg.image_size[1] // yolo_interval)
            num_anchor_boxes =  5    # (height/width)   1/5  1/3  1/1  3/1  5/1
            yolo_tensor = torch.zeros( self.rpg.batch_size, num_yolo_cells, num_anchor_boxes, 8 )
            with torch.no_grad():
                for iter, data in enumerate(self.test_dataloader):
                    im_tensor, seg_mask_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data
                    if iter % 50 == 49:
                        print("\n\n\n\nShowing output for test batch %d: " % (iter+1))
                        im_tensor   = im_tensor.to(self.rpg.device)
                        seg_mask_tensor = seg_mask_tensor.to(self.rpg.device)                 
                        bbox_tensor = bbox_tensor.to(self.rpg.device)
                        bbox_label_tensor = bbox_label_tensor.to(self.rpg.device)
                        yolo_tensor = yolo_tensor.to(self.rpg.device)

                        output = net(im_tensor)
                        
                        predictions = output.view(4,36,5,8)

                        for ibx in range(predictions.shape[0]):                             # for each batch image
                            icx_2_best_anchor_box = {ic : None for ic in range(36)}
                            for icx in range(predictions.shape[1]):                         # for each yolo cell
                                cell_predi = predictions[ibx,icx]               
                                prev_best = 0
                                for anchor_bdx in range(cell_predi.shape[0]):
                                    if cell_predi[anchor_bdx][0] > cell_predi[prev_best][0]:
                                        prev_best = anchor_bdx
                                best_anchor_box_icx = prev_best   
                                icx_2_best_anchor_box[icx] = best_anchor_box_icx
                            sorted_icx_to_box = sorted(icx_2_best_anchor_box, 
                                       key=lambda x: predictions[ibx,x,icx_2_best_anchor_box[x]][0].item(), reverse=True)
                            retained_cells = sorted_icx_to_box[:5]
                            objects_detected = []
                            for icx in retained_cells:
                                pred_vec = predictions[ibx,icx, icx_2_best_anchor_box[icx]]
                                class_labels_predi  = pred_vec[-3:]                        
                                if torch.all(class_labels_predi < 0.05): 
                                    predicted_class_label = None
                                else:
                                    best_predicted_class_index = (class_labels_predi == 
                                                                      class_labels_predi.max()).nonzero().squeeze().item()
                                    predicted_class_label = self.rpg.class_labels[best_predicted_class_index]
                                    objects_detected.append(predicted_class_label)
                            print("[batch image=%d]  objects found in descending probability order: " % ibx, objects_detected)
                        logger = logging.getLogger()
                        old_level = logger.level
                        logger.setLevel(100)
                        plt.figure(figsize=[15,4])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.show()
                        logger.setLevel(old_level)

    

    ###%%%
    #######################################################################################################
    ###################  Start Definition of Inner Class RPN (Region Proposal Network)  ###################
    class RPN(nn.Module):             
        """
        I have not yet mentioned this class in the documentation page for this module
        because its implementation is not finished.
        """
        def __init__(self, rpg):
            super(RegionProposalGenerator.RPN, self).__init__()
            self.rpg = rpg
#            self.dataserver_train = None
#            self.dataserver_test =  None
            self.train_dataloader = None
            self.test_dataloader = None

        def show_sample_images_from_dataset(self, rpg):
            data = next(iter(self.train_dataloader))    
            real_batch = data[0]
            first_im = real_batch[0]
            self.rpg.display_tensor_as_image(torchvision.utils.make_grid(real_batch, padding=2, pad_value=1, normalize=True))

        def set_dataloaders(self, train=False, test=False):
            if train:
                dataserver_train = RegionProposalGenerator.PurdueDrEvalDataset(self.rpg, "train", 
                                                                                      dataroot_train=self.rpg.dataroot_train)
                self.train_dataloader = torch.utils.data.DataLoader(dataserver_train, self.rpg.batch_size, 
                                                                                                 shuffle=True, num_workers=4)
            if test:
                dataserver_test = RegionProposalGenerator.PurdueDrEvalDataset(self.rpg, "test", 
                                                                                        dataroot_test=self.rpg.dataroot_test)
                self.test_dataloader = torch.utils.data.DataLoader(dataserver_test, self.rpg.batch_size, shuffle=False, 
                                                                                                               num_workers=4)

        def check_dataloader(self, train=False, test=False):
            if train:      
                dataloader = self.train_dataloader
            if test:
                dataloader = self.test_dataloader
            for i, data in enumerate(dataloader):          
                im_tensor,mask_tensor,bbox_tensor, image_label = data
                for idx in range(im_tensor.shape[0]):
                    self.rpg.display_tensor_as_image( im_tensor[idx], "batch number: %d" % i)
                    print("\n\nbbox tensor: ", bbox_tensor[idx])
                    self.rpg.display_tensor_as_image( mask_tensor[idx], "batch number: %d" % i)
                im_tensor   = im_tensor.to(self.rpg.device)
                mask_tensor = mask_tensor.type(torch.FloatTensor)
                mask_tensor = mask_tensor.to(self.rpg.device)                 
                bbox_tensor = bbox_tensor.to(self.rpg.device)

        class SkipBlockDN(nn.Module):
            def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                super(RegionProposalGenerator.RPN.SkipBlockDN, self).__init__()
                self.downsample = downsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convo1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.bn2 = nn.BatchNorm2d(out_ch)
                if downsample:
                    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
            def forward(self, x):
                identity = x                                     
                out = self.convo1(x)                              
                out = self.bn1(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo2(out)                              
                    out = self.bn2(out)                              
                    out = torch.nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
                return out

        class SkipBlockUP(nn.Module):
            def __init__(self, in_ch, out_ch, upsample=False, skip_connections=True):
                super(RegionProposalGenerator.RPN.SkipBlockUP, self).__init__()
                self.upsample = upsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convoT1 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
                self.convoT2 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.bn2 = nn.BatchNorm2d(out_ch)
                if upsample:
                    self.upsampler = nn.ConvTranspose2d(in_ch, out_ch, 1, stride=2, dilation=2, output_padding=1, padding=0)

            def forward(self, x):
                identity = x                                     
                out = self.convoT1(x)                              
                out = self.bn1(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convoT2(out)                              
                    out = self.bn2(out)                              
                    out = torch.nn.functional.relu(out)
                if self.upsample:
                    out = self.upsampler(out)
                    identity = self.upsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out += identity[:,self.out_ch:,:,:]
                return out

        class mUnet_for_RPN(nn.Module):
            def __init__(self, skip_connections=True, depth=16):
                super(RegionProposalGenerator.RPN.mUnet_for_RPN, self).__init__()
                self.depth = depth // 2
                self.conv_in = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                ##  For the DN arm of the U:
                self.bn1DN  = nn.BatchNorm2d(64)
                self.bn2DN  = nn.BatchNorm2d(128)
                self.skip64DN_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64DN_arr.append(RegionProposalGenerator.RPN.SkipBlockDN(64, 64, skip_connections=skip_connections))
                self.skip64dsDN = RegionProposalGenerator.RPN.SkipBlockDN(64, 64, downsample=True, skip_connections=skip_connections)
                self.skip64to128DN = RegionProposalGenerator.RPN.SkipBlockDN(64, 128, skip_connections=skip_connections )
                self.skip128DN_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128DN_arr.append(RegionProposalGenerator.RPN.SkipBlockDN(128, 128, skip_connections=skip_connections))
                self.skip128dsDN = RegionProposalGenerator.RPN.SkipBlockDN(128,128, downsample=True, skip_connections=skip_connections)

                ##  For the UP arm of the U:
                self.bn1UP  = nn.BatchNorm2d(128)
                self.bn2UP  = nn.BatchNorm2d(64)
                self.skip64UP_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64UP_arr.append(RegionProposalGenerator.RPN.SkipBlockUP(64, 64, skip_connections=skip_connections))
                self.skip64usUP = RegionProposalGenerator.RPN.SkipBlockUP(64, 64, upsample=True, skip_connections=skip_connections)
                self.skip128to64UP = RegionProposalGenerator.RPN.SkipBlockUP(128, 64, skip_connections=skip_connections )
                self.skip128UP_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128UP_arr.append(RegionProposalGenerator.RPN.SkipBlockUP(128, 128, skip_connections=skip_connections))
                self.skip128usUP = RegionProposalGenerator.RPN.SkipBlockUP(128,128,
                                            upsample=True, skip_connections=skip_connections)
                self.conv_out = nn.ConvTranspose2d(64, 2, 3, stride=2,dilation=2,output_padding=1,padding=2)

            def forward(self, x):
                x = self.pool(torch.nn.functional.relu(self.conv_in(x)))          
                for i,skip64 in enumerate(self.skip64DN_arr[:self.depth//4]):
                    x = skip64(x)                
                num_channels_to_save1 = x.shape[1] // 2
                save_for_upside_1 = x[:,:num_channels_to_save1,:,:].clone()
                x = self.skip64dsDN(x)
                for i,skip64 in enumerate(self.skip64DN_arr[self.depth//4:]):
                    x = skip64(x)                
                x = self.bn1DN(x)
                num_channels_to_save2 = x.shape[1] // 2
                save_for_upside_2 = x[:,:num_channels_to_save2,:,:].clone()
                x = self.skip64to128DN(x)
                for i,skip128 in enumerate(self.skip128DN_arr[:self.depth//4]):
                    x = skip128(x)                
                x = self.bn2DN(x)
                num_channels_to_save3 = x.shape[1] // 2
                save_for_upside_3 = x[:,:num_channels_to_save3,:,:].clone()
                for i,skip128 in enumerate(self.skip128DN_arr[self.depth//4:]):
                    x = skip128(x)                
                x = self.skip128dsDN(x)

                ## Coming up from the bottom of U on the other side:
                x = self.skip128usUP(x)          
                for i,skip128 in enumerate(self.skip128UP_arr[:self.depth//4]):
                    x = skip128(x)                
                x[:,:num_channels_to_save3,:,:] =  save_for_upside_3
                x = self.bn1UP(x)
                for i,skip128 in enumerate(self.skip128UP_arr[:self.depth//4]):
                    x = skip128(x)                
                x = self.skip128to64UP(x)
                for i,skip64 in enumerate(self.skip64UP_arr[self.depth//4:]):
                    x = skip64(x)                
                x[:,:num_channels_to_save2,:,:] =  save_for_upside_2
                x = self.bn2UP(x)
                x = self.skip64usUP(x)
                for i,skip64 in enumerate(self.skip64UP_arr[:self.depth//4]):
                    x = skip64(x)                
                x[:,:num_channels_to_save1,:,:] =  save_for_upside_1
                x = self.conv_out(x)
                return x

        class SegmentationLoss(nn.Module):
            def __init__(self, batch_size):
                super(RegionProposalGenerator.RPN.SegmentationLoss, self).__init__()
                self.batch_size = batch_size
            def forward(self, output, mask_tensor):
                composite_loss = torch.zeros(1,self.batch_size)
                mask_based_loss = torch.zeros(1,5)
                for idx in range(self.batch_size):
                    outputh = output[idx,0,:,:]
                    for mask_layer_idx in range(mask_tensor.shape[0]):
                        mask = mask_tensor[idx,mask_layer_idx,:,:]
                        element_wise = (outputh - mask)**2                   
                        mask_based_loss[0,mask_layer_idx] = torch.mean(element_wise)
                    composite_loss[0,idx] = torch.sum(mask_based_loss)
                return torch.sum(composite_loss) / self.batch_size


        def run_code_for_training_RPN(self, net):        
            filename_for_out1 = "performance_numbers_" + str(self.rpg.epochs) + ".txt"
            FILE1 = open(filename_for_out1, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.rpg.device)
            criterion1 = nn.MSELoss()
            optimizer = optim.SGD(net.parameters(), lr=self.rpg.learning_rate, momentum=self.rpg.momentum)
            print("\n\nStarting training loop...\n\n")
            start_time = time.perf_counter()
            Loss_tally = []
            composite_mask_tensor = torch.zeros(self.rpg.batch_size, 2, self.rpg.image_size[0], self.rpg.image_size[1])
            for epoch in range(self.rpg.epochs):  
                print("")
                running_loss_segmentation = 0.0
                for i, data in enumerate(self.train_dataloader):   
                    im_tensor,mask_tensor,bbox_tensor, image_label = data
                    im_tensor   = im_tensor.to(self.rpg.device)
                    mask_tensor = mask_tensor.type(torch.FloatTensor)
                    mask_tensor = mask_tensor.to(self.rpg.device)                 
                    bbox_tensor = bbox_tensor.to(self.rpg.device)
                    im_tensor_masked = im_tensor * mask_tensor
                    mask_tensor_complement = 1 - mask_tensor
                    composite_mask_tensor[:,0,:,:] = mask_tensor[:,0,:,:]
                    composite_mask_tensor[:,1,:,:] = mask_tensor_complement[:,0,:,:]
                    composite_mask_tensor = composite_mask_tensor.to(self.rpg.device)
                    optimizer.zero_grad()
                    output = net(im_tensor_masked) 
                    segmentation_loss = criterion1(output, composite_mask_tensor)  
                    segmentation_loss.backward()
                    optimizer.step()
                    running_loss_segmentation += segmentation_loss.item()    
                    if i%100==99:    
                        current_time = time.perf_counter()
                        elapsed_time = current_time - start_time 
                        avg_loss_segmentation = running_loss_segmentation / float(100)
                        print("[epoch:%d/%d, iter=%4d  elapsed_time=%5d secs]      mean MSE loss: %7.4f" % 
                                                         (epoch+1,self.rpg.epochs, i+1, elapsed_time, avg_loss_segmentation))
                        Loss_tally.append(running_loss_segmentation)
                        FILE1.write("%.3f\n" % avg_loss_segmentation)
                        FILE1.flush()
                        running_loss_segmentation = 0.0
            print("\nFinished Training\n")
            plt.figure(figsize=(10,5))
            plt.title("RPN Training: Loss vs. Iterations")
            plt.plot(Loss_tally)
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig("rpn_training_loss.png")
            plt.show()
            self.save_RPN_model(net)
            return net


        def save_RPN_model(self, model):
            '''
            Save the trained RPN model to a disk file
            '''
            torch.save(model.state_dict(), self.rpg.path_saved_RPN_model)

        def run_code_for_testing_RPN(self, net):
            net.load_state_dict(torch.load(self.rpg.path_saved_RPN_model))
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    im_tensor,mask_tensor,bbox_tensor, image_label = data
                    if self.rpg.debug_test and i % 50 == 0:
                        print("\n\n\n\nShowing output for test batch %d: " % (i+1))
                        outputs = net(im_tensor)                           #### Network Output
                        outputs = outputs[:,0,:,:]
                        outputs_smoothed = outputs.clone().detach().cpu()
                        display_tensor = torch.zeros(self.rpg.batch_size,3, 4 * self.rpg.image_size[0],
                                                                           self.rpg.image_size[1], dtype=float).cpu()
                        display_tensor[:,:,:128,:]     =  im_tensor
                        display_tensor[:,:,128:256,:]  =  mask_tensor
                        display_tensor[:,:,256:384,:]  =  torch.unsqueeze(outputs,1)
                        display_tensor[:,:,384:512,:]  =  torch.unsqueeze(outputs_smoothed,1)
                        for idx in range(self.rpg.batch_size):
                            bb_tensor = bbox_tensor[idx]
                            i1 = int(bb_tensor[1]) + 384
                            i2 = int(bb_tensor[3]) + 384
                            j1 = int(bb_tensor[0])
                            j2 = int(bb_tensor[2])
                            display_tensor[idx,0,i1:i2,j1] = 1.0
                            display_tensor[idx,0,i1:i2,j2] = 1.0
                            display_tensor[idx,0,i1,j1:j2] = 1.0
                            display_tensor[idx,0,i2,j1:j2] = 1.0
                        display_tensor[:,:,126:129,:] = 1.0
                        display_tensor[:,:,254:257,:] = 1.0
                        display_tensor[:,:,382:385,:] = 1.0
                        plt.imshow(np.transpose(torchvision.utils.make_grid(display_tensor, normalize=False, 
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.show()

        def test_rpn_model_on_one_image(self, image, rpn_model=None):
            if rpn_model is None:
                rpn_model = self.mUnet_for_RPN(skip_connections=True, depth=16)
                rpn_model.load_state_dict(torch.load(self.rpg.path_saved_RPN_model))
            resize_and_normalize = tvt.Compose( [tvt.Resize((128,128)), 
                                                 tvt.ToTensor(), 
                                                 tvt.Normalize(mean=[0.5], std=[0.5]) ] )
            display_tensor = torch.zeros(1,3,self.rpg.image_size[0], 2*self.rpg.image_size[1], dtype=float).cpu()
            with torch.no_grad():
                image = Image.open(image)
                im_tensor = resize_and_normalize(image)
                display_tensor[:,:,:,:128] = im_tensor
                im_tensor = torch.unsqueeze(im_tensor,0)
                output = rpn_model(im_tensor)     

                output = output[:,0,:,:]
                output[output <= 0.0] = 0
                output[output > 0.0] = 1.0
                display_tensor[:,:,:,128:] = output
                plt.imshow(np.transpose(torchvision.utils.make_grid(display_tensor, normalize=False, 
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                plt.show()


    ###%%%
    #######################################################################################################    
    ################################ Class based utility functions #######################################

    def graying_resizing_binarizing(self, image_file, polarity=1, area_threshold=0, min_brightness_level=100):
        '''
        This is a demonstration of some of the more basic and commonly used image
        transformations from the torchvision.transformations module.  The large comments
        blocks are meant to serve as tutorial introduction to the syntax used for invoking
        these transformations.  The transformations shown can be used for converting a
        color image into a grayscale image, for resizing an image, for converting a
        PIL.Image into a tensor and a tensor back into an PIL.Image object, and so on.
        '''
        if os.path.isfile(image_file):
            im_pil = Image.open(image_file)
        else:
            sys.exit("the image file %s does not exist --- aborting" % image_file)
        self.displayImage6(im_pil, "input_image")

        ###  The next three lines of code that follow are three examples of calls to the
        ###  constructor of the torchvision.tranforms.Compose class whose contract, as its
        ###  name implies, is to compose a sequence of transformations to be applied to an
        ###  image.  The instance of Compose constructed in line (A) has only one
        ###  transformation in it, which would resize an image to a 64x64 array of pixels.
        ###  On the other hand, the instance constructed in line (B) includes two
        ###  transformations: the first transformation is for converting an image from
        ###  "RGB" to gray scale, and the second for resizing an image as before to an
        ###  array of 64x64 pixels.  The instance of Compose constructed in line (C)
        ###  incorporates a sequence of five transformations. It invoked on a color image,
        ###  it will convert the image into grayscale, then resize it to an array of 64x64
        ###  pixels, convert the array to a tensor, normalize the array so that its mean
        ###  and the standard deviation both equal 0.5, and, finally, convert the tensor
        ###  into a PIL image object.  
        ###
        ###  A most important thing to note here is that each of the instances returned in
        ###  lines (A), (B), and (C) is a callable object, meaning that the instance can
        ###  be called directly, with the image to which the transformation are to be
        ###  applied, as the argument to the instance.
        ###
        ###  Note that in the Compose instance constructed in line (C), we had to
        ###  interpose the "ToTensor" transformation between the Resize and the Normalize
        ###  transformations because the Resize transformation returns an Image object that
        ###  cannot be normalized directly.  That is, the Normalize transformation is
        ###  meant for the normalization of tensors --- it takes a tensor as its input and
        ###  returns a tensor at its output.  If you want the final result of the sequence
        ###  of transformations in line (C) to return an Image, then you would also like
        ###  to call the ToPILImage transformation as shown.
        ###  
        resize_xform = tvt.Compose( [ tvt.Resize((64,64)) ] )                               ## (A)

        gray_and_resize = tvt.Compose( [tvt.Grayscale(num_output_channels = 1),  
                                        tvt.Resize((64,64)) ] )                             ## (B)

        gray_resize_normalize = tvt.Compose( [tvt.Grayscale(num_output_channels = 1), 
                                              tvt.Resize((64,64)), 
                                              tvt.ToTensor(), 
                                              tvt.Normalize(mean=[0.5], std=[0.5]), 
                                              tvt.ToPILImage() ] )                          ## (C)

        ###  As explained in the previous comment block, the three statements shown above
        ###  are merely calls to the constructor of the Compose class for the creation of
        ###  instances.  As also mentioned previously, these instances are designed to be
        ###  callable; that is, they can be treated like function objects for actually
        ###  applying the transformations to a given image.  This is shown in the lines of
        ###  code that follow.
        ###
        ###  Applying the resize_xform of line (A) to an image:
        img = resize_xform( im_pil )
        self.displayImage6(img, "output_of_resize_xform")

        ###  Applying gray_and_resize of line (B) to an image:
        img = gray_and_resize( im_pil )
        self.displayImage6(img, "output_of_gray_and_resize")

        ###  Applying gray_resize_normalize of line (C) to an image:
        img = gray_resize_normalize( im_pil )
        self.displayImage6(img, "output_of_gray_resize_normalize")

        ###  Demonstrating the ToTensor transformation all by itself.  As in earlier
        ###  examples, first construct a callable instance of Compose and then invoke it
        ###  on the image which must of type PIL.Image.
        img_tensor = tvt.Compose([tvt.ToTensor()])
        img_data = img_tensor(img)
        print("\nshape of the img_data tensor: %s" % str(img_data.shape))               ##  (1,64,64)
        print("\n\n\nimg_tensor: %s" % str(img_data))
                           #
                           #  tensor([[[0.9333, 0.9569, 0.9647,  ..., 0.6745, 0.5882, 0.5569],
                           #           [0.8392, 0.8392, 0.7922,  ..., 0.6275, 0.6980, 0.7922],
                           #           [0.9255, 0.9176, 0.8157,  ..., 0.9725, 0.9725, 0.9882],
                           #           ...,
                           #           [0.4431, 0.4745, 0.5882,  ..., 0.6588, 0.7373, 0.6667],
                           #           [0.4431, 0.5098, 0.5725,  ..., 0.4667, 0.5255, 0.5412],
                           #           [0.5098, 0.5490, 0.5255,  ..., 0.4980, 0.6118, 0.5804]]])
                           #               

        ###  With the image in its 1x64x64 numeric tensor representation, we can apply a
        ###  comparison operator to the individual elements of the tensor to threshold the
        ###  the image data.  Since the pixel values in a grayscale image (we have
        ###  grayscale because of an earlier transformation to the originally color image)
        ###  are between 0 and 255 and since the normalization is going to convert these
        ###  numbers into floating point numbers between 0.0 and 1.0, the thresholding
        ###  operation applied below is going to set to FALSE all pixel values that are
        ###  below 128 and to TRUE all pixel values that are above 128.
        img_data = img_data > 0.5                                                           ## (D)
        print("\n\n\nimg_data: %s" % str(img_data))
                           #
                           #  tensor([[[ True,  True,  True,  ...,  True,  True,  True],
                           #           [ True,  True,  True,  ...,  True,  True,  True],
                           #           [ True,  True,  True,  ...,  True,  True,  True],
                           #           ...,
                           #           [False, False,  True,  ...,  True,  True,  True],
                           #           [False,  True,  True,  ..., False,  True,  True],
                           #           [ True,  True,  True,  ..., False,  True,  True]]])

        ###  In order to visualize the thresholding effect achieved above, we need to
        ###  convert the Boolean pixel values back into numbers, which we can do by
        ###  calling float() on the output image tensor as shown below:
        img_data = img_data.float()                                                         ## (E)
        ###  Now we need to construct a Compose instance with the ToPILImage
        ###  transformation at its heart.  This we can do by:
        to_image_xform = tvt.Compose([tvt.ToPILImage()])                                    ## (F)
        ###  Invoking the callable to_image_xform instance on the tensor returned by the
        ###  call in line (E) gives us the desired PIL.Image object that can be
        ###  visualized.
        img = to_image_xform(img_data)
        self.displayImage6(img, "after_thresholding")

    def display_tensor_as_image2(self, tensor, title=""):
        '''
        This method converts the argument tensor into a photo image that you can display
        in your terminal screen. It can convert tensors of three different shapes
        into images: (3,H,W), (1,H,W), and (H,W), where H, for height, stands for the
        number of pixels in the vertical direction and W, for width, for the same
        along the horizontal direction.  When the first element of the shape is 3,
        that means that the tensor represents a color image in which each pixel in
        the (H,W) plane has three values for the three color channels.  On the other
        hand, when the first element is 1, that stands for a tensor that will be
        shown as a grayscale image.  And when the shape is just (H,W), that is
        automatically taken to be for a grayscale image.
        '''
        tensor_range = (torch.min(tensor).item(), torch.max(tensor).item())
        if tensor_range == (-1.0,1.0):
            ##  The tensors must be between 0.0 and 1.0 for the display:
            print("\n\n\nimage un-normalization called")
            tensor = tensor/2.0 + 0.5     # unnormalize
        plt.figure(title)
        ###  The call to plt.imshow() shown below needs a numpy array. We must also
        ###  transpose the array so that the number of channels (the same thing as the
        ###  number of color planes) is in the last element.  For a tensor, it would be in
        ###  the first element.
        if tensor.shape[0] == 3 and len(tensor.shape) == 3:
#            plt.imshow( tensor.numpy().transpose(1,2,0) )
            plt.imshow( tensor.numpy().transpose(1,2,0) )
        ###  If the grayscale image was produced by calling torchvision.transform's
        ###  ".ToPILImage()", and the result converted to a tensor, the tensor shape will
        ###  again have three elements in it, however the first element that stands for
        ###  the number of channels will now be 1
        elif tensor.shape[0] == 1 and len(tensor.shape) == 3:
            tensor = tensor[0,:,:]
            plt.imshow( tensor.numpy(), cmap = 'gray' )
        ###  For any one color channel extracted from the tensor representation of a color
        ###  image, the shape of the tensor will be (W,H):
        elif len(tensor.shape) == 2:
            plt.imshow( tensor.numpy(), cmap = 'gray' )
        else:
            sys.exit("\n\n\nfrom 'display_tensor_as_image2()': tensor for image is ill formed -- aborting")
        plt.show()


    def display_tensor_as_image(self, tensor, title=""):
        '''
        This method converts the argument tensor into a photo image that you can display
        in your terminal screen. It can convert tensors of three different shapes
        into images: (3,H,W), (1,H,W), and (H,W), where H, for height, stands for the
        number of pixels in the vertical direction and W, for width, for the same
        along the horizontal direction.  When the first element of the shape is 3,
        that means that the tensor represents a color image in which each pixel in
        the (H,W) plane has three values for the three color channels.  On the other
        hand, when the first element is 1, that stands for a tensor that will be
        shown as a grayscale image.  And when the shape is just (H,W), that is
        automatically taken to be for a grayscale image.
        '''
        print("\n\n\ndisplay_tensor_as_image() called with a tensor of type: %s" % tensor.type())
                                                                                  ##  torch.FloatTensor
        ###  The 'plt' in the following statement stands for the plotting module 
        ###  matplotlib.pyplot.  See the module import declarations at the beginning of
        ###  this module.
        plt.figure(title)

        ###  The call to plt.imshow() shown below needs a numpy array. We must also
        ###  transpose the array so that the number of channels (the same thing as the
        ###  number of color planes) is in the last element.  For a tensor, it would be in
        ###  the first element.
        if tensor.shape[0] == 3 and len(tensor.shape) == 3:
            plt.imshow( tensor.numpy().transpose(1,2,0) )

        ###  If the grayscale image was produced by calling torchvision.transform's
        ###  ".ToPILImage()", and the result converted to a tensor, the tensor shape will
        ###  again have three elements in it, however the first element that stands for
        ###  the number of channels will now be 1
        elif tensor.shape[0] == 1 and len(tensor.shape) == 3:
            tensor = tensor[0,:,:]
            plt.imshow( tensor.numpy(), cmap = 'gray' )

        ###  For any one color channel extracted from the tensor representation of a color
        ###  image, the shape of the tensor will be (W,H):
        elif len(tensor.shape) == 2:
            plt.imshow( tensor.numpy(), cmap = 'gray' )
        else:
            sys.exit("\n\n\ntensor for image is ill formed -- aborting")
        plt.show()
            
    def accessing_one_color_plane(self, image_file, n):
        '''
        This method shows how can access the n-th color plane of the argument color image.
        '''
        if os.path.isfile(image_file):
            im_pil = Image.open(image_file)
        else:
            sys.exit("the image file %s does not exist --- aborting" % image_file)

        ###  In order to access the color planes individually, it is best to first convert
        ###  the image into a tensor of shape 3xWxH where 3 is for the three color planes,
        ###  W for the width of the image in pixels, and H for the height of the image in
        ###  pixels. To accomplish this PIL.Image to tensor conversion, we first need to
        ###  construct an instance of the ToTensor class by calling its constructor.
        ###  Since the resulting instance will be a callable object, we can treat it like
        ###  a function object and invoke it directly as shown below while supplying the
        ###  image to it as its argument.
        image_to_tensor_converter = tvt.ToTensor()
        image_as_tensor = image_to_tensor_converter(im_pil)

        ###  IT IS VERY IMPORTANT TO REALIZE that while the pixels in the original color
        ###  image are one-byte integers, with values between 0 and 255 for each of the
        ###  color channels, after the image is turned into a tensor, the three values at
        ###  each pixel are convereted into a floating point number between 0.0 and 1.0.
        print("\n\n\nimage as tensor: %s" % str(image_as_tensor))
                            #
                            #  tensor([[[0.4588, 0.4588, 0.4627,  ..., 0.2667, 0.2627, 0.2549],    r-plane
                            #           [0.4588, 0.4627, 0.4667,  ..., 0.2784, 0.2745, 0.2667],
                            #           [0.4588, 0.4667, 0.4745,  ..., 0.2784, 0.2745, 0.2667],
                            #           ...,
                            #           [0.2078, 0.2235, 0.2392,  ..., 0.2941, 0.2627, 0.2392],
                            #           [0.2118, 0.2314, 0.2431,  ..., 0.2902, 0.2706, 0.2549],
                            #           [0.2235, 0.2392, 0.2471,  ..., 0.2706, 0.2588, 0.2510]],
                            # 
                            #          [[0.4784, 0.4784, 0.4824,  ..., 0.2902, 0.2863, 0.2784],    g-plane
                            #           [0.4745, 0.4784, 0.4824,  ..., 0.3020, 0.2980, 0.2902],
                            #           [0.4824, 0.4902, 0.4980,  ..., 0.3020, 0.2980, 0.2902],
                            #           ...,
                            #           [0.2510, 0.2667, 0.2824,  ..., 0.3529, 0.3216, 0.2980],
                            #           [0.2549, 0.2745, 0.2863,  ..., 0.3373, 0.3176, 0.3020],
                            #           [0.2667, 0.2824, 0.2902,  ..., 0.3098, 0.2980, 0.2902]],
                            # 
                            #          [[0.2275, 0.2275, 0.2314,  ..., 0.1490, 0.1529, 0.1451],    b-plane
                            #           [0.2353, 0.2392, 0.2431,  ..., 0.1608, 0.1569, 0.1490],
                            #           [0.2392, 0.2471, 0.2549,  ..., 0.1529, 0.1490, 0.1490],
                            #           ...,
                            #           [0.1176, 0.1333, 0.1490,  ..., 0.2000, 0.1686, 0.1451],
                            #           [0.1216, 0.1412, 0.1529,  ..., 0.1882, 0.1686, 0.1529],
                            #           [0.1333, 0.1490, 0.1569,  ..., 0.1647, 0.1529, 0.1451]]])
                            # 
        ###  Two different ways of checking the type of the tensor.  The second call is more
        ###  informative
        print("\n\n\nType of image_as_tensor: %s" % type(image_as_tensor))       ## <class 'torch.Tensor'>
        print("\n[More informative]  Type of image_as_tensor: %s" % image_as_tensor.type())    
                                                                                 ## <class 'torch.FloatTensor'>
        print("\n\n\nShape of image_as_tensor: %s" % str(image_as_tensor.shape)) ## (3, 366, 320)

        ###  The following function will automatically re-convert the 0.0 to 1.0 floating
        ###  point values for at the pixels into the integer one-byte representations for
        ###  displaying the image.
        self.display_tensor_as_image(image_as_tensor,"color image in 'accessing each color plane method'")

        ###  n=0 means the R channel, n=1 the G channel, and n=2 the B channel
        channel_image = image_as_tensor[n]
        print("\n\n\nchannel image: %s" % str(channel_image))
                            # tensor([[0.4588, 0.4588, 0.4627,  ..., 0.2667, 0.2627, 0.2549],
                            #         [0.4588, 0.4627, 0.4667,  ..., 0.2784, 0.2745, 0.2667],
                            #         [0.4588, 0.4667, 0.4745,  ..., 0.2784, 0.2745, 0.2667],
                            #         ...,
                            #         [0.2078, 0.2235, 0.2392,  ..., 0.2941, 0.2627, 0.2392],
                            #         [0.2118, 0.2314, 0.2431,  ..., 0.2902, 0.2706, 0.2549],
                            #         [0.2235, 0.2392, 0.2471,  ..., 0.2706, 0.2588, 0.2510]]) 

        self.display_tensor_as_image(channel_image, "showing just the designated channel" )

        ###   In the statement shown below, the coefficients (0.4, 0.4, 0.2) are a measure
        ###   of how sensitive the human visual system is to the three different color
        ###   channels.  Index 0 is for R, index 1 for G, and the index 2 for B.
        ###
        ###   Note that these weights are predicated on the pixel values being
        ###   respresented by floating-point numbers between 0.0 and 1.0 (as opposed
        ###   to the more commonly used one-byte integers).
        gray_tensor = 0.4 * image_as_tensor[0]  +   0.4 * image_as_tensor[1]   + 0.2 * image_as_tensor[2]
        self.display_tensor_as_image(gray_tensor, "showing the grayscale version")


    def extract_data_pixels_in_bb(self, image_file, bounding_box):
        '''
        Mainly used for testing
        '''
        im_arr  =  np.asarray(Image.open(image_file))
        height,width,_ = im_arr.shape
        hmin,hmax = bounding_box[0],bounding_box[2]
        wmin,wmax = bounding_box[1],bounding_box[3]
        im_arr_portion = im_arr[hmin:hmax,wmin:wmax,:]
        return im_arr_portion

    def working_with_hsv_color_space(self, image_file, test=False):
        ''' 
        Shows color image conversion to HSV
        '''
        if os.path.isfile(image_file):
            im_pil = Image.open(image_file)
        else:
            sys.exit("the image file %s does not exist --- aborting" % image_file)

        ###   Get the HsV representation of the PIL Image object by invoking
        ###   "convert('HSV')" on it as shown below:
        hsv_image = im_pil.convert('HSV')
        hsv_arr = np.asarray(hsv_image)
        np.save("hsv_arr.npy", hsv_arr)
        image_to_tensor_converter = tvt.ToTensor()
        hsv_image_as_tensor = image_to_tensor_converter( hsv_image )
        ###   The index "1" as the last argument means that we want the three images
        ###   to be concatenated horizontally (meaning, along the 'width' dimension
        ###   as opposed to the 'height' dimension).  If you change that value to
        ###   "0", you will see the three images lined up vertically.
        if test is False:
            self.display_tensor_as_image(torch.cat((hsv_image_as_tensor[0], hsv_image_as_tensor[1], 
                                           hsv_image_as_tensor[2] ),1), "displaying the HSV channels separately")


    def histogramming_the_image(self, image_file):
        '''
        PyTorch based experiments with histogramming the grayscale and the color values in an
        image
        '''
        if os.path.isfile(image_file):
            im_pil = Image.open(image_file)
        else:
            sys.exit("the image file %s does not exist --- aborting" % image_file)

        image_to_tensor_converter = tvt.ToTensor()
        color_image_as_tensor = image_to_tensor_converter( im_pil )

        ###   Let's first plot the histogram of the grayscale version of the image:
        gray_tensor = 0.4 * color_image_as_tensor[0]  +   0.4 * color_image_as_tensor[1]   + 0.2 * color_image_as_tensor[2]
        hist_gray = torch.histc(gray_tensor, bins = 10, min = 0.0, max = 1.0)
        hist_gray = hist_gray.div( hist_gray.sum() )

        fig = plt.figure("histogram of the grayscale")
        ax = fig.add_subplot(111)
        ax.bar( np.linspace(1.0, 10.0, num = 10), hist_gray.numpy(), color='black' )
        plt.show()

        ###   We will now plot separately the histogram for each color channel
        ###
        r_tensor = color_image_as_tensor[0]
        g_tensor = color_image_as_tensor[1]
        b_tensor = color_image_as_tensor[2]
        
        ###  Computing the hist for each color channel separately
        hist_r = torch.histc(r_tensor, bins = 10, min = 0.0, max = 1.0)
        hist_g = torch.histc(g_tensor, bins = 10, min = 0.0, max = 1.0)
        hist_b = torch.histc(b_tensor, bins = 10, min = 0.0, max = 1.0)
        
        ###  Normalizing the channel based hists so that the bin counts in each sum to 1.
        hist_r = hist_r.div(hist_r.sum())
        hist_g = hist_g.div(hist_g.sum())
        hist_b = hist_b.div(hist_b.sum())
        
        ### Displaying the channel histograms separately in one figure:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True)
        fig.title = "histogramming the color components separately"
        ax1.bar(np.linspace(1.0, 10.0, num = 10), hist_r.numpy(), color='r')
        ax2.bar(np.linspace(1.0, 10.0, num = 10), hist_g.numpy(), color='g')
        ax3.bar(np.linspace(1.0, 10.0, num = 10), hist_b.numpy(), color='b')
        plt.show();


    def displaying_and_histogramming_images_in_batch1(self, dir_name, batch_size):
        '''
        This method is the first of three such methods in this module for illustrating the
        functionality of matplotlib for simultaneously displaying multiple images and
        the results obtained on them in gridded arrangements.  In the implementation
        shown below, the core idea in this method is to call
        "plt.subplots(2,batch_size)" to create 'batch_size' number of subplot
        objects, called "axes", in the form of a '2xbatch_size' array. We use the first
        row of this grid to display each image in its own subplot object.  And we use
        the second row the grid to display the histogram of the corresponding image
        in the first row.
        '''
        fig, axes = plt.subplots(2,batch_size)
        image_files = glob.glob(dir_name + '/*.jpg')[:batch_size]
        images = list(map(Image.open, image_files))
        images = [tvt.Grayscale()(x) for x in images]
        images = [tvt.Resize((64,64), Image.ANTIALIAS)(x) for x in images]
        im_tensors = [tvt.ToTensor()(x) for x in images]
        im_tensors = [tvt.Normalize(mean=[0.5], std=[0.5])(x) for x in im_tensors]

        for j in range(batch_size):
            axes[0,j].imshow(im_tensors[j][0,:,:].numpy(), cmap='gray')

        hists = [torch.histc(x,  bins=10) for x in im_tensors]
        total_counts = list(map(sum, hists)) 
        hists_normed = [hists[i] / total_counts[i] for i in range(len(hists))]
        for j in range(batch_size):
            axes[1,j].bar(np.linspace(1.0, 10.0, num = 10), hists_normed[j].numpy())  
            axes[1,j].set_yticks([])
        plt.show()

    def displaying_and_histogramming_images_in_batch2(self, dir_name, batch_size):
        '''
        I now show a second approach to display multiple images and their corresponding
        histograms in a gridded display.  Unlike in the previous implementation of
        this method, now we do not call on "plt.subplots()" to create a grid
        structure for displaying the images.  On the other hand, we now call on
        "torchvision.utils.make_grid()" to construct a grid for us.  The grid is
        created by giving an argument like "nrow=4" to it.  When using this method,
        an important thing to keep in mind is that the first argument to make_grip()
        must be a tensor of shape "(B, C, H, W)" where B stands for batch_size, C for
        channels (3 for color, 1 for gray), and (H,W) for the height and width of the
        image. What that means in our example is that we need to synthesize a tensor
        of shape "(8,1,64,64)" in order to be able to call the "make_grid()"
        function. Note that the object returned by the call to make_grid() is a
        tensor unto itself.  For the example shown, if we had called
        "print(grid.shape)" on the "grid" returned by "make_grid()", the answer would
        be "torch.Size([3, 158, 306])" which, after it is converted into a numpy
        array, can be construed by a plotting function as a color image of size
        158x306.
        '''
        image_files = glob.glob(dir_name + '/*.jpg')[:batch_size]
        images = list(map(Image.open, image_files))
        images = [tvt.Grayscale()(x) for x in images]
        images = [tvt.Resize((64,64), Image.ANTIALIAS)(x) for x in images]
        im_tensors = [tvt.ToTensor()(x) for x in images]
        im_tensors = [tvt.Normalize(mean=[0.5], std=[0.5])(x) for x in im_tensors]
        IM_Tensor = torch.zeros(batch_size,1,64,64, dtype=float)
        for i in range(batch_size):
            IM_Tensor[i,0,:,:] = im_tensors[i][0,:,:]
        # for the display:
        grid = tutils.make_grid(IM_Tensor, nrow=4, padding=10, normalize=True)
        npgrid = grid.cpu().numpy()
        plt.imshow(np.transpose(npgrid, (1,2,0)), interpolation='nearest')
        plt.show()

        hists = [torch.histc(x,  bins=10) for x in im_tensors]
        total_counts = list(map(sum, hists)) 
        hists_normed = [hists[i] / total_counts[i] for i in range(len(hists))]
        fig, axes = plt.subplots(nrows=2, ncols=4, sharey = True)    

        for i in range(2):
            for j in range(batch_size // 2):
                k = i * (batch_size//2) + j
                axes[i,j].bar(np.linspace(1.0, 10.0, num = 10), hists_normed[k].numpy())  
        plt.show();

    def displaying_and_histogramming_images_in_batch3(self, dir_name, batch_size):
        '''
        The core idea here is to illustrate two things: (1) The syntax used for the
        'singular' version of the subplot function "plt.subplot()" --- although I'll
        be doing so by actually calling "fig.add_subplot()".  And (2) How you can put
        together multiple multi-image plots by creating multiple Figure objects.
        Figure is the top-level container of plots in matplotlib.  In the 
        implementation shown below, the key statements are: 

            fig1 = plt.figure(1)    
            axis = fig1.add_subplot(241)              
                                                                                                                      
        Calling "add_subplot()" on a Figure object returns an "axis" object.  The
        word "axis" is a misnomer for what should really be called a "subplot".
        Subsequently, you can call display functions lime "imshow()", "bar()", etc.,
        on the axis object to display an individual plot in a gridded arrangement.

        The argument "241" in the first call to "add_subplot()" means that your
        larger goal is to create a 2x4 display of plots and that you are supplying
        the 1st plot for that grid.  Similarly, the argument "242" in the next call
        to "add_subplot()" means that for your goal of creating a 2x4 gridded
        arrangement of plots, you are now supplying the second plot.  Along the same
        lines, the argument "248" toward the end of the code block that you are now
        supplying the 8th plot for the 2x4 arrangement of plots.

        Note how we create a second Figure object in the second major code block.  We
        use it to display the histograms for each of the images shown in the first
        Figure object.  The two Figure containers will be shown in two separate
        windows on your laptop screen.
        '''
        image_files = glob.glob(dir_name + '/*.jpg')[:batch_size]
        images = list(map(Image.open, image_files))
        images = [tvt.Grayscale()(x) for x in images]
        images = [tvt.Resize((64,64), Image.ANTIALIAS)(x) for x in images]
        im_tensors = [tvt.ToTensor()(x) for x in images]
        im_tensors = [tvt.Normalize(mean=[0.5], std=[0.5])(x) for x in im_tensors]

        # Let's make a Figure for the 8 images:
        fig1 = plt.figure(1)
        axis = fig1.add_subplot(241)
        axis.imshow(im_tensors[0][0,:,:].numpy(), cmap='gray') 
        axis = fig1.add_subplot(242)
        axis.imshow(im_tensors[1][0,:,:].numpy(), cmap='gray') 
        axis = fig1.add_subplot(243)
        axis.imshow(im_tensors[2][0,:,:].numpy(), cmap='gray') 
        axis = fig1.add_subplot(244)
        axis.imshow(im_tensors[3][0,:,:].numpy(), cmap='gray') 
        axis = fig1.add_subplot(245)
        axis.imshow(im_tensors[4][0,:,:].numpy(), cmap='gray') 
        axis = fig1.add_subplot(246)
        axis.imshow(im_tensors[5][0,:,:].numpy(), cmap='gray') 
        axis = fig1.add_subplot(247)
        axis.imshow(im_tensors[6][0,:,:].numpy(), cmap='gray') 
        axis = fig1.add_subplot(248)
        axis.imshow(im_tensors[7][0,:,:].numpy(), cmap='gray') 

        # Now let's make a second figure for the 8 corresponding histograms:
        hists = [torch.histc(x,  bins=10) for x in im_tensors]
        total_counts = list(map(sum, hists)) 
        hists_normed = [hists[i] / total_counts[i] for i in range(len(hists))]
        fig2 = plt.figure(2)
        axis = fig2.add_subplot(241)
        axis.bar(np.linspace(1.0, 10.0, num = 10), hists_normed[0].numpy())
        axis.set_yticks([])
        axis = fig2.add_subplot(242)
        axis.bar(np.linspace(1.0, 10.0, num = 10), hists_normed[1].numpy())
        axis.set_yticks([])
        axis = fig2.add_subplot(243)
        axis.bar(np.linspace(1.0, 10.0, num = 10), hists_normed[2].numpy())
        axis.set_yticks([])
        axis = fig2.add_subplot(244)
        axis.bar(np.linspace(1.0, 10.0, num = 10), hists_normed[3].numpy())
        axis.set_yticks([])
        axis = fig2.add_subplot(245)
        axis.bar(np.linspace(1.0, 10.0, num = 10), hists_normed[4].numpy())
        axis.set_yticks([])
        axis = fig2.add_subplot(246)
        axis.bar(np.linspace(1.0, 10.0, num = 10), hists_normed[5].numpy())
        axis.set_yticks([])
        axis = fig2.add_subplot(247)
        axis.bar(np.linspace(1.0, 10.0, num = 10), hists_normed[6].numpy())
        axis.set_yticks([])
        axis = fig2.add_subplot(248)
        axis.bar(np.linspace(1.0, 10.0, num = 10), hists_normed[7].numpy())
        axis.set_yticks([])
        plt.show()

    def histogramming_and_thresholding(self, image_file):
        '''
        PyTorch based experiments with histogramming and thresholding
        '''
        if os.path.isfile(image_file):
            im_pil = Image.open(image_file)
        else:
            sys.exit("the image file %s does not exist --- aborting" % image_file)

        image_to_tensor_converter = tvt.ToTensor()
        ###   Note that "self.original_im" is a PIL Image object
        color_image_as_tensor = image_to_tensor_converter( im_pil )
        print("\n\n\nshape of the image tensor: %s" % str(color_image_as_tensor.shape))
        print("\n\n\ndisplaying the original color image")
        self.display_tensor_as_image(color_image_as_tensor, "original color image")
        ###   Let's first plot the histogram of the grayscale version of the image:
        gray_tensor = 0.4 * color_image_as_tensor[0]  +   0.4 * color_image_as_tensor[1]   + 0.2 * color_image_as_tensor[2]
        print("\n\n\ndisplaying the grayscale version of the color image")
        self.display_tensor_as_image(gray_tensor, "grayscale version of color image")
#        hist_gray = torch.histc(gray_tensor, bins = 255, min = 0.0, max = 256.0)
        hist_gray = torch.histc(gray_tensor, bins = 255, min = 0.0, max = 1.0)
        fig = plt.figure("plot of the histogram")
        ax = fig.add_subplot(111)
        ax.bar( np.linspace(1.0, 256, num = 255), hist_gray.numpy(), color='black' )
        print("\n\n\ndisplaying the histogram of the graylevels")
        plt.show()
        print("\n\n\nNumber of pixels in the histogram: %s" % str(hist_gray.sum()))
        print("\n\n\nhist_gray: %s" % str(hist_gray))
        prob = hist_gray.div( hist_gray.sum() )
        cumulative_prob = prob.cumsum(0)           ##  this gives us a cumulative probability distribution
        print("\n\n\ncumulative_probability: %s" % str(cumulative_prob))
        print("\n\n\nnumber of bins in the cumulative prob: %s" % str(len(cumulative_prob)))       ## 255

        ###  For the rest of the implementation of the Otsu algo, the fact that the
        ###  histogram of the gray levels was calculated with the grayscale values scaled
        ###  to floating point numbers between 0 and 1 by the tensor representation IS NOT
        ###  NOT NOT NOT an issue.  That is because cumulative_prob is an array of 256
        ###  numbers, which each number corresponding one of 256 gray levels.
        hist = prob
        cumu_hist = cumulative_prob
        sigmaBsquared = {k : None for k in range(255)}
        for k in range(255):
             ###   Notice calling ".item()" on one-element tensors to extract the number being
             ###   held by them:
            omega0 = cumu_hist[k].item()
            omega1 = 1 - omega0
            if omega0 > 0 and omega1 > 0:
                mu0 = (1.0/omega0) * sum([i * hist[i].item() for i in range(0,k+1)])     
                mu1 = (1.0/omega1) * sum([i * hist[i].item() for i in range(k+1,255)])      
                sigmaBsquared[k] = omega0 * omega1 * (mu1 - mu0) ** 2
        sigmaBsquared = {k : sigmaBsquared[k] for k in range(255) if sigmaBsquared[k] is not None}

        sorted_thresholds = sorted(sigmaBsquared.items(), key=lambda x: x[1], reverse=True)
        print("\nThe threshold discovered by Otsu: %d" % sorted_thresholds[0][0])
        otsu_threshold = sorted_thresholds[0][0]
        thresholded_gray_image_as_tensor =  torch.clamp( gray_tensor, min=(otsu_threshold / float(256) ) )
#        thresholded_gray_image_as_tensor =  torch.clone(gray_tensor)
        tensor_shape = thresholded_gray_image_as_tensor.shape
        for i in range(tensor_shape[0]):
            for j in range(tensor_shape[1]):
                if thresholded_gray_image_as_tensor[i,j] < (otsu_threshold / float(256)):
                    thresholded_gray_image_as_tensor[i,j] = 0.0
        print("\n\n\nDisplaying the Otsu thresholded image")
        self.display_tensor_as_image(thresholded_gray_image_as_tensor, "otsu thresholded version")


    def convolutions_with_pytorch(self, image_file, kernel):
        '''
        Using torch.nn.functional.conv2d() for demonstrating a single image convolution with
        a specified kernel
        '''
        if os.path.isfile(image_file):
            im_pil = Image.open(image_file)
        else:
            sys.exit("the image file %s does not exist --- aborting" % image_file)

        image_to_tensor_converter = tvt.ToTensor()
        color_image_as_tensor = image_to_tensor_converter( im_pil )
        gray_tensor = 0.4 * color_image_as_tensor[0]  +   0.4 * color_image_as_tensor[1]   + 0.2 * color_image_as_tensor[2] 
        self.display_tensor_as_image(gray_tensor, "grayscale version for demonstrating convolution")

        gray_section = gray_tensor[100:200, 100:250]
        self.display_tensor_as_image(gray_section, "showing a 100x150 section of the grayscale image")

        ###  converting the convolutional kernel into a tensor:
        op = torch.Tensor( kernel )

        ###  The input to the "nn.functional.conv2d()" requires a tensor of shape
        ###
        ###       (batch_size, num_input_channels, image_height, image_width)
        ###
        ###  Since we are dealing with a single image for demonstrating how you can carry
        ###  out a convolution, in our case batch_size=1.  Additionally, the convolution
        ###  demo is on a grayscale image, so we must set num_input_channels=1.
        input = torch.Tensor(1, 1, gray_section.shape[0], gray_section.shape[1])

        ###   The second arg to the "nn.functional.conv2d()" invoked below is the kernel
        ###   you want to use for the convolution.  This kernel must have the same shape
        ###   as the input image.  So the first two shape parameters are the same as for
        ###   the input image.
        kernel = torch.Tensor(1, 1, op.shape[0], op.shape[1])

        print("\n\n\nshape of input: %s" % str(input.shape))
        print("\n\n\nshape of kernel: %s" % str(kernel.shape))  

        ###   Now we must stuff the gray image where it belongs in the input tensor:
        input[0,0,:,:] = gray_section

        ###   Finally, we must stuff the kernel where it belongs in the kernel tensor:
        kernel[0,0,:,:] = op
        output = nn.functional.conv2d( input, kernel, stride=1, padding=1 )
        print(output)

        ###   In order to respect the input conditions required by the display function,
        ###   we now extract the convolved output from the result of the convolution:
        output = output[0,0,:,:]
        self.display_tensor_as_image( output, "the result of convolution on the image section" )


    def gaussian_smooth(self, pil_grayscale_image):
        '''
        This method smooths an image with a Gaussian of specified sigma.
        '''
        sigma = self.sigma
        width,height = pil_grayscale_image.size
#        gray_image = self.data_im
        gray_image_as_array = np.zeros((height, width), dtype="float")
        for i in range(0, height):
            for j in range(0, width):
                gray_image_as_array[(i,j)] = pil_grayscale_image.getpixel((j,i))
        self.gray_image_as_array = gray_image_as_array
        smoothing_op = _gaussian(sigma)
        smoothed_image_array = _convolution_2D(gray_image_as_array, smoothing_op)
#        self._display_and_save_array_as_image( smoothed_image_array, "_gaussian_smoothed__" + str(sigma) )
        height,width = smoothed_image_array.shape
        maxVal = smoothed_image_array.max()
        minVal = smoothed_image_array.min()
        newimage = Image.new("L", (width,height), (0,))
        for i in range(0, height):
            for j in range(0, width):
                displayVal = int( (smoothed_image_array[(i,j)] - minVal) * (255/(maxVal-minVal)) )
                newimage.putpixel((j,i), displayVal)
        self.displayImage3(newimage, "Gaussian Smoother: close window when done viewing")
#        image_name = "smoothed" + "_" + self.data_im_name
        image_name = "smoothed.png"
        newimage.save(image_name)        
        return newimage


    def visualize_segmentation_in_pseudocolor(self, pixel_blobs, color_map, label=""):
        '''
        Assigns a random color to each blob in the output of an image segmentation algorithm
        '''
        height,width = self.im_array.shape
        colorized_mask_image = Image.new("RGB", (width,height), (0,0,0))
        for blob_idx in sorted(pixel_blobs, key=lambda x: len(pixel_blobs[x]), reverse=True):
            for (i,j) in pixel_blobs[blob_idx]:
                colorized_mask_image.putpixel((j,i), color_map[blob_idx])
        seg_mask_image = colorized_mask_image.resize((width*self.image_size_reduction_factor,
                                          height*self.image_size_reduction_factor), Image.ANTIALIAS)
        self.displayImage6(seg_mask_image, label + "_segmentation")

    def visualize_segmentation_with_mean_gray(self, pixel_blobs, label=""):
        '''
        Assigns the mean color to each each blob in the output of an image segmentation algorithm
        '''
        height,width = self.im_array.shape
        gray_mask_image = Image.new("L", (width,height), (0))
        for blob_idx in sorted(pixel_blobs, key=lambda x: len(pixel_blobs[x]), reverse=True):
            pixel_blob = pixel_blobs[blob_idx]
            pixel_vals = np.array([self.im_array[pixel] for pixel in pixel_blob])
            gray_mean = int(np.mean(pixel_vals))
            for (i,j) in pixel_blobs[blob_idx]:
                gray_mask_image.putpixel((j,i), gray_mean)
        seg_mask_image = gray_mask_image.resize((width*self.image_size_reduction_factor,
                                        height*self.image_size_reduction_factor), Image.ANTIALIAS)
        self.displayImage6(seg_mask_image, label)

    def repair_blobs(self, merged_blobs, color_map, all_pairwise_similarities):
        '''
        The goal here to do a final clean-up of the blob by merging tiny pixel blobs with
        an immediate neighbor, etc.  Such a cleanup requires adjacency info regarding the
        blobs in order to figure out which larger blob to merge a small blob with.
        '''
        pairwise_adjacency  =  all_pairwise_similarities['adjacency']
        pairwise_color_homogeneity_val  =  all_pairwise_similarities['color_homogeneity']
        pairwise_gray_var_comp =  all_pairwise_similarities['gray_var']
        pairwise_texture_comp = all_pairwise_similarities['texture']

        singleton_blobs = [blob_id for blob_id in merged_blobs if len(merged_blobs[blob_id]) == 1]
        sorted_blobs = sorted(merged_blobs, key=lambda x: len(merged_blobs[x]))
        for blob_id in singleton_blobs:
            if blob_id not in merged_blobs: continue
            for blb_id in sorted_blobs:            
                if blb_id == blob_id: continue
                if blb_id not in merged_blobs: continue
                if blb_id > blob_id:
                    pair_label = "%d,%d" % (blb_id,blob_id)
                else:
                    pair_label = "%d,%d" % (blob_id,blb_id)                    
                if blb_id in merged_blobs and blob_id in merged_blobs and pairwise_adjacency[pair_label] is True:
                    merged_blobs[blb_id] += merged_blobs[blob_id]
                    del merged_blobs[blob_id]
        sorted_blobs = sorted(merged_blobs, key=lambda x: len(merged_blobs[x]))
        for blob_id in sorted_blobs:
            if len(merged_blobs[blob_id]) > 200: continue
            neighboring_blobs = []          
            for blb_id in sorted_blobs:
                if blb_id == blob_id: continue
                if blb_id > blob_id:
                    pair_label = "%d,%d" % (blb_id,blob_id)
                else:
                    pair_label = "%d,%d" % (blob_id,blb_id)                    

                if ( (pairwise_adjacency[pair_label] is True) and 
                     (pairwise_color_homogeneity_val[pair_label] < self.color_homogeneity_thresh) and 
                     (pairwise_gray_var_comp[pair_label] < self.gray_var_thresh) and 
                     (pairwise_texture_comp[pair_label] < self.texture_homogeneity_thresh) ):
                    neighboring_blobs.append(blb_id)
            if self.debug: 
                print("\n\n\nneighboring_blobs for blob %d: %s" % (blob_id, str(neighboring_blobs)))
            if len(neighboring_blobs) == 1 and len(merged_blobs[neighboring_blobs[0]]) > len(merged_blobs[blob_id]):
                merged_blobs[neighboring_blobs[0]] += merged_blobs[blob_id] 
                del merged_blobs[blob_id]
        return merged_blobs,color_map                              


    def selective_search_for_region_proposals(self, graph, image_name):
        '''
        This method implements the Selective Search (SS) algorithm proposed by Uijlings,
        van de Sande, Gevers, and Smeulders for creating region proposals for object
        detection.  As mentioned elsewhere here, that algorithm sits on top of the graph
        based image segmentation algorithm that was proposed by Felzenszwalb and
        Huttenlocher.  The parameter 'pixel_blobs' required by the method presented here
        is supposed to be the pixel blobs produced by the Felzenszwalb and Huttenlocher
        algorithm.
        '''
        def are_two_blobs_adjacent(blob1, blob2):
            '''
            We say that two pixel blobs with no pixels in common are adjacent if at
            least one pixel in one block is 8-adjacent to any of the pixels in the other
            pixel blob.
            '''
            for pixel_u in blob1:
                for pixel_v in blob2:
                    if abs(pixel_u[0] - pixel_v[0])  <= 1 and abs(pixel_u[1] - pixel_v[1]) <= 1:
                        return True
            return False

        def estimate_lbp_texture(blob, im_array):
            '''
             This method implements the Local Binary Patterns (LBP) method of characterizing image
            textures. This algorithm, proposed by Ojala, Pietikainen, and Maenpaa
            generates a grayscale and rotationally invariant texture signature through
            what is referred to as an LBP histogram.  For a tutorial introduction to this
            method, see:
                   https://engineering.purdue.edu/kak/Tutorials/TextureAndColor.pdf
            The code presented below is borrowed from this tutorial.
            '''
            import BitVector
            height_coords = [p[0] for p in blob]
            width_coords  = [p[1] for p in blob]
            bb_height_min = min(height_coords)
            bb_height_max = max(height_coords)
            bb_width_min  = min(width_coords)
            bb_width_max  = max(width_coords) 
            ###  Create bounding box for each blob to make it more convenient to apply
            ###  the LBP logic to the blob:
            bb = [[0 for w in range(bb_width_max - bb_width_min + 1)] 
                              for h in range(bb_height_max - bb_height_min + 1)]
            for h in range(bb_height_max - bb_height_min + 1):
                for w in range(bb_width_max - bb_width_min + 1):
                    if (h+bb_height_min, w+bb_width_min) in blob:
                        bb[h][w] = im_array[h+bb_height_min,w+bb_width_min]
            if self.debug:
                print("\n\n\nbb: %s" % str(bb))
            R,P = 1,8
            rowmax,colmax = bb_height_max-bb_height_min+1 - R, bb_width_max - bb_width_min + 1 - R
            lbp_hist = {t:0 for t in range(P+2)}                                  
            ###  Visit each pixel and find the LBP vector at that pixel.            
            for h in range(rowmax):           
                for w in range(colmax):       
                    pattern = [] 
                    for p in range(P):                                               
                        #  We use the index k to point straight down and l to point to the 
                        #  right in a circular neighborhood around the point (i,j). And we 
                        #  use (del_k, del_l) as the offset from (i,j) to the point on the 
                        #  R-radius circle as p varies.
                        del_k,del_l = R*math.cos(2*math.pi*p/P), R*math.sin(2*math.pi*p/P)  
                        if abs(del_k) < 0.001: del_k = 0.0                                  
                        if abs(del_l) < 0.001: del_l = 0.0                                  
                        k, l =  h + del_k, w + del_l                                        
                        k_base,l_base = int(k),int(l)                                       
                        delta_k,delta_l = k-k_base,l-l_base                                 
                        if (delta_k < 0.001) and (delta_l < 0.001):                         
                            image_val_at_p = float(bb[k_base][l_base])                   
                        elif (delta_l < 0.001):                                             
                            image_val_at_p = (1 - delta_k) * bb[k_base][l_base] +  \
                                                          delta_k * bb[k_base+1][l_base] 
                        elif (delta_k < 0.001):                                             
                            image_val_at_p = (1 - delta_l) * bb[k_base][l_base] +  \
                                                          delta_l * bb[k_base][l_base+1] 
                        else:                                                               
                            image_val_at_p = (1-delta_k)*(1-delta_l)*bb[k_base][l_base] + \
                                             (1-delta_k)*delta_l*bb[k_base][l_base+1]  + \
                                             delta_k*delta_l*bb[k_base+1][l_base+1]  + \
                                             delta_k*(1-delta_l)*bb[k_base+1][l_base]   
                        if image_val_at_p >= bb[h][w]:                                  
                            pattern.append(1)                                              
                        else:                                                              
                            pattern.append(0)                                              
                    if self.debug:
                        print("pattern: %s" % pattern)                                         
                    bv =  BitVector.BitVector( bitlist = pattern )                         
                    intvals_for_circular_shifts  =  [int(bv << 1) for _ in range(P)]       
                    minbv = BitVector.BitVector( intVal = min(intvals_for_circular_shifts), size = P )   
                    if self.debug:
                        print("minbv: %s" % minbv)                                               
                    bvruns = minbv.runs()                                                    
                    encoding = None
                    if len(bvruns) > 2:                                                
                        lbp_hist[P+1] += 1                                             
                        encoding = P+1                                                 
                    elif len(bvruns) == 1 and bvruns[0][0] == '1':                     
                        lbp_hist[P] += 1                                               
                        encoding = P                                                   
                    elif len(bvruns) == 1 and bvruns[0][0] == '0':                     
                        lbp_hist[0] += 1                                               
                        encoding = 0                                                   
                    else:                                                              
                        lbp_hist[len(bvruns[1])] += 1                                  
                        encoding = len(bvruns[1])                                      
                    if self.debug:
                        print("encoding: %s" % encoding)                                   
            if self.debug:
                print("\nLBP Histogram: %s" % lbp_hist)                                    
            lbp_array = np.zeros(len(lbp_hist))
            for i in range(len(lbp_hist)): lbp_array[i] = lbp_hist[i]
            return lbp_array
            ###  End of Texture operator definition

        ###  BEGIN CODE FOR SELECTIVE-SEARCH BASED MERGING OF THE BLOBS
        ###  BUT FIRST WE COMPUTE UNARY AND BINARY ATTRIBUTES OF THE BLOBS.
        pixel_blobs,E = graph
        ###  We need access to the underlying image to fetch the pixel values for the blobs
        ###  in question:
        im_array_color  =    np.asarray(self.low_res_PIL_image_color)
        im_array_gray = self.im_array
        ###  Compute unary properties of blobs:
        color_mean_vals = {}
        gray_mean_vals = {}
        gray_vars = {}
        texture_vals = {}
        sorted_blobs = sorted(pixel_blobs, key=lambda x: len(pixel_blobs[x]), reverse=True)
        for blob_id in sorted_blobs:
            pixel_blob = pixel_blobs[blob_id]
#            pixel_vals_color = np.array([im_array_color[pixel] for pixel in pixel_blob])
            pixel_vals_color = [im_array_color[pixel[0],pixel[1],:].tolist() for pixel in pixel_blob]
            pixel_vals_gray =  np.array([im_array_gray[pixel] for pixel in pixel_blob])
            color_mean_vals[blob_id]  = [ float(sum([pix[j] for pix in pixel_vals_color])) / float(len(pixel_vals_color)) for j in range(3) ]
            gray_mean_vals[blob_id] =  np.mean(pixel_vals_gray)
            gray_vars[blob_id] = np.var(pixel_vals_gray)
            texture_vals[blob_id] = estimate_lbp_texture(pixel_blob, im_array_gray)
        if self.debug:
            print("\n\n\ncolor_mean_vals: %s" % str(color_mean_vals))
            print("\n\n\ngray_mean_vals: %s" % str(gray_mean_vals))
            print("\n\n\ngray_vars: %s" % str(gray_vars))
            print("\n\n\ntexture_vals: %s" % str(texture_vals))

        ###  Compute pairwise similarity scores:
        all_pairwise_similarities = {}
        pairwise_adjacency = {}
        pairwise_gray_homogeneity_val = {}
        pairwise_color_homogeneity_val = {}
        pairwise_gray_var_comp = {}
        pairwise_texture_comp   = {}
        all_pairwise_similarities['adjacency'] = pairwise_adjacency
        all_pairwise_similarities['color_homogeneity'] = pairwise_color_homogeneity_val
        all_pairwise_similarities['gray_var'] = pairwise_gray_var_comp
        all_pairwise_similarities['texture'] = pairwise_texture_comp

        for blob_id_1 in pixel_blobs:
            for blob_id_2 in pixel_blobs:
                if blob_id_1 > blob_id_2:
                    pair_id = str("%d,%d" % (blob_id_1,blob_id_2))
                    pairwise_adjacency[pair_id] = True if pair_id in E else False
                    pairwise_gray_homogeneity_val[pair_id] = abs(gray_mean_vals[blob_id_1]
                                                                  - gray_mean_vals[blob_id_2])
                    pairwise_color_homogeneity_val[pair_id] = [abs(color_mean_vals[blob_id_1][j] 
                                               - color_mean_vals[blob_id_2][j]) for j in range(3)]
                    pairwise_gray_var_comp[pair_id] = abs(gray_vars[blob_id_1] - gray_vars[blob_id_2])
                    pairwise_texture_comp[pair_id] =  np.linalg.norm(texture_vals[blob_id_1] - 
                                                                     texture_vals[blob_id_2])
        if self.debug:
            print("\n\n\npairwise_adjacency: %s" % str(pairwise_adjacency))
            print("\n\n\npairwise_gray_homogeneity_val: %s" % str(pairwise_gray_homogeneity_val))
            print("\n\n\npairwise_color_homogeneity_val: %s" % str(pairwise_color_homogeneity_val))
            print("\n\n\npairwise_gray_var_comp: %s" % str(pairwise_gray_var_comp))
            print("\n\n\npairwise_texture_comp: %s" % str(pairwise_texture_comp)) 

        ###  Initialize merged blocks
        merged_blobs = pixel_blobs
        if self.debug:
            print("\n\n\ninitial blobs: %s" % str(pixel_blobs))
        next_blob_id = max(merged_blobs.keys()) + 1
        ###  You have to be careful with the program flow in the blob merging block of
        ###  code shown below in order to deal with the fact that you are modifying the
        ###  blobs as you iterate through them.  You merge two blobs because they are
        ###  adjacent and because they are color and texture homogeneous.  However, when
        ###  you merge two blobs, the original blobs must be deleted from the blob
        ###  dictionary.  At the same time, you must compute the unary properties of the
        ###  newly formed blob and also estimate its pairwise properties with respect to
        ###  all the other blobs in the blob dictionary.
        ss_iterations = 0
        '''
        In this version, we will make only one pass through the 'while' loop shown below
        because, in the UPDATE of the PAIRWISE PROPERTIES, I have not yet included
        those pairs that involve the latest LATEST NEW blob vis-a-vis the other older
        NEWLY DISCOVERED blobs.  In any case, my experience has shown that you need
        just one pass for the images in the Examples directory.  However, it is
        possible that, for complex imagery, you may need multiple (even an
        indeterminate number of) passes through the blob merging code shown below.
        '''
        while ss_iterations < 1:
            sorted_up_blobs = sorted(merged_blobs, key=lambda x: len(merged_blobs[x]))
            sorted_down_blobs = sorted(merged_blobs, key=lambda x: len(merged_blobs[x]), reverse=True)
            for blob_id_1 in sorted_up_blobs:
                if blob_id_1 not in merged_blobs: continue
                for blob_id_2 in sorted_down_blobs[:-1]:        # the largest blob is typically background
                    if blob_id_2 not in merged_blobs: continue
                    if blob_id_1 not in merged_blobs: break
                    if blob_id_1 > blob_id_2:
                        pair_id = "%d,%d" % (blob_id_1,blob_id_2)   
#                        if (pair_id in pairwise_adjacency) and (pairwise_adjacency[pair_id] is True):
                        if (pairwise_color_homogeneity_val[pair_id][0] < self.color_homogeneity_thresh[0])\
                           and                                                                   \
                           (pairwise_color_homogeneity_val[pair_id][1] < self.color_homogeneity_thresh[1])\
                           and                                                                   \
                           (pairwise_color_homogeneity_val[pair_id][2] < self.color_homogeneity_thresh[2])\
                           and                                                                    \
                           (pairwise_gray_var_comp[pair_id] < self.gray_var_thresh)               \
                           and                                                                    \
                           (pairwise_texture_comp[pair_id] < self.texture_homogeneity_thresh):          
                            if self.debug:
                                print("\n\n\nmerging blobs of id %d and %d" % (blob_id_1, blob_id_2))
                            new_merged_blob = merged_blobs[blob_id_1] +  merged_blobs[blob_id_2] 
                            merged_blobs[next_blob_id] = new_merged_blob
                            del merged_blobs[blob_id_1]
                            del merged_blobs[blob_id_2]
                            ###  We need to estimate the unary properties of the newly created
                            ###  blob:
                            pixel_vals_color = [im_array_color[pixel[0],pixel[1],:].tolist() for pixel in 
                                                                                    new_merged_blob]
                            pixel_vals_gray = np.array([im_array_gray[pixel] for pixel in new_merged_blob])
                            color_mean_vals[next_blob_id]  = [float(sum([pix[j] for pix in \
                                pixel_vals_color])) / float(len(pixel_vals_color)) for j in range(3)]
                            gray_mean_vals[next_blob_id] = np.mean(pixel_vals_gray)
                            gray_vars[next_blob_id] = np.var(pixel_vals_gray)
                            texture_vals[next_blob_id] = estimate_lbp_texture(new_merged_blob, im_array_gray)
                            ###  Now that we have merged two blobs, we need to create entries
                            ###  in pairwise dictionaries for entries related to this new blob
                            for blb_id in sorted_up_blobs:
                                if blb_id not in merged_blobs: continue
                                if next_blob_id > blb_id:
                                    pair_id = "%d,%d" % (next_blob_id, blb_id)
                                    pairwise_adjacency[pair_id] = \
                  True if are_two_blobs_adjacent(new_merged_blob, pixel_blobs[blb_id]) else False
                                    pairwise_color_homogeneity_val[pair_id] = \
                  [abs(color_mean_vals[next_blob_id][j]  - color_mean_vals[blb_id][j]) for j in range(3)]
                                    pairwise_gray_homogeneity_val[pair_id] = \
                                         abs(gray_mean_vals[next_blob_id] - gray_mean_vals[blb_id])
                                    pairwise_gray_var_comp[pair_id] = \
                                         abs(gray_vars[next_blob_id] - gray_vars[blb_id])
                                    pairwise_texture_comp[pair_id] =  \
                                np.linalg.norm(texture_vals[next_blob_id] - texture_vals[blb_id])
                    next_blob_id += 1                           
            ss_iterations += 1

        num_pixels_in_final_merged_blobs = sum( [len(blob) for _,blob in merged_blobs.items()] )
        print("\n\n\ntotal number of pixels in all merged blobs: %d" % num_pixels_in_final_merged_blobs)
        ###  color_map is a dictionary with blob_ids as keys and the assigned values the color
        ###  assigned to each blob for its visualization

        bounding_boxes = {}
        retained_vertex_list = []
        total_pixels_in_output = 0
        color_map = {}
        for blob_idx in sorted(merged_blobs, key=lambda x: len(merged_blobs[x]), reverse=True)[:self.max_num_blobs_expected]:
            color_map[blob_idx] = (random.randint(0,255), random.randint(0,255),random.randint(0,255))
            all_pixels_in_blob = merged_blobs[blob_idx]
            total_pixels_in_output += len(all_pixels_in_blob)
            retained_vertex_list.append(blob_idx)
            height_coords = [p[0] for p in all_pixels_in_blob]
            width_coords  = [p[1] for p in all_pixels_in_blob]
            bb_height_min = min(height_coords)
            bb_height_max = max(height_coords)
            bb_width_min  = min(width_coords)
            bb_width_max  = max(width_coords)
            bounding_boxes[blob_idx] = [bb_height_min, bb_width_min, bb_height_max, bb_width_max]

        print("\n\n\nTotal number of pixels in output blobs: %d" % total_pixels_in_output)
        title = "selective_search_based_bounding_boxes"
        arr_height,arr_width = im_array_gray.shape
        colorized_mask_image = Image.new("RGB", (arr_width,arr_height), (0,0,0))
        for blob_idx in retained_vertex_list:
            for (i,j) in merged_blobs[blob_idx]:
                colorized_mask_image.putpixel((j,i), color_map[blob_idx])
        mw = Tkinter.Tk()
        winsize_w,winsize_h = None,None
        screen_width,screen_height = mw.winfo_screenwidth(),mw.winfo_screenheight()
        if screen_width <= screen_height:
            winsize_w = int(0.5 * screen_width)
            winsize_h = int(winsize_w * (arr_height * 1.0 / arr_width))            
        else:
            winsize_h = int(0.5 * screen_height)
            winsize_w = int(winsize_h * (arr_width * 1.0 / arr_height))
        scaled_image =  colorized_mask_image.copy().resize((winsize_w,winsize_h), Image.ANTIALIAS)
        mw.title(title) 
        mw.configure( height = winsize_h, width = winsize_w )         
        canvas = Tkinter.Canvas( mw,                         
                             height = winsize_h,            
                             width = winsize_w,             
                             cursor = "crosshair" )   
        canvas.pack(fill=BOTH, expand=True)
        frame = Tkinter.Frame(mw)                            
        frame.pack( side = 'bottom' )                             
        Tkinter.Button( frame,         
                text = 'Save',                                    
                command = lambda: canvas.postscript(file = title + ".eps") 
              ).pack( side = 'left' )                             
        Tkinter.Button( frame,                        
                text = 'Exit',                                    
                command = lambda: mw.destroy(),                    
              ).pack( side = 'right' )                            
        photo = ImageTk.PhotoImage( scaled_image )
        canvas.create_image(winsize_w//2,winsize_h//2,image=photo)
        scale_w = winsize_w / float(arr_width)
        scale_h = winsize_h / float(arr_height)
        for v in bounding_boxes:
            bb = bounding_boxes[v]
            canvas.create_rectangle( (bb[1]*scale_w,bb[0]*scale_h,(bb[3]+1)*scale_w,(bb[2]+1)*scale_h), width='3', outline='red' )
        canvas.update()
        mw.update()
        print("\n\n\nIterations used: %d" % self.iterations_used)
        print("\n\n\nNumber of region proposals: %d" % len(bounding_boxes))
        mw.mainloop()
        if os.path.isfile(title + ".eps"):
            Image.open(title + ".eps").save(title + ".png")
            os.remove(title + ".eps")
        retained_vertices = {}
        for u in retained_vertex_list:
            retained_vertices[u] = merged_blobs[u]
        return retained_vertices, color_map

        def are_two_blobs_color_homogeneous(blob1, blob2, image):
            color_in_1 = [image[pixel] for pixel in blob1]
            color_in_2 = [image[pixel] for pixel in blob2]
            mean_diff = abs(np.mean(color_in_1) - np.mean(color_in_2))
            var1 = np.var(color_in_1)
            var2 = np.var(color_in_2)
            if var1 < self.var_threshold and var2 < self.var_thresh and mean_diff < self.mean_diff_thresh:
                return True
            return False

        def are_two_blobs_texture_homogeneous(blob1, blob2, image):
            lbp_hist_1 = estimate_lbp_texture(blob1)
            lpb_hist_2 = estimate_lbp_texture(blob2)
            if np_norm( np.to_array(lbp_hist_1) - np.to_array(lbp_hist_2) ) < self.texture_thresh:
                return True
            return False               

    def graph_based_segmentation(self, image_name, num_blobs_wanted=None):
        '''
        This is an implementation of the Felzenszwalb and Huttenlocher algorithm for
        graph-based segmentation of images.  At the moment, it is limited to working
        on grayscale images.
        '''
        ###  image_name may be a file, in which case it needs to be opened, or directly
        ###  a PIL.Image object
        try:
            image_pil_color = Image.open(image_name)
        except:
            image_pil_color = image_name                           ### needed for the interactive mode
        width,height = image_pil_color.size
        kay = self.kay
        print("\n\n\nImage of WIDTH=%d  and  HEIGHT=%d   being processed by graph_based_segmentation" % (width,height))
        self.displayImage6(image_pil_color, "input_image -- size: width=%d height=%d" % (width,height))
        kay = self.kay
        input_image_gray = image_pil_color.copy().convert("L")      ## first convert the image to grayscale
        if self.sigma > 0:
            smoothed_image_gray = self.gaussian_smooth(input_image_gray) 
            ##  we do NOT need a smoothed version of the original color image
        else:
            smoothed_image_gray = input_image_gray
        image_size_reduction_factor = self.image_size_reduction_factor
        width_to = width // image_size_reduction_factor
        height_to = height // image_size_reduction_factor
        if self.image_normalization_required:
            gray_resized_normalized = tvt.Compose( [tvt.Grayscale(num_output_channels = 1), tvt.Resize((height_to,width_to)), tvt.ToTensor(), tvt.Normalize(mean=[0.5], std=[0.5]) ] )
            color_resized_normalized = tvt.Compose( [tvt.Resize((height_to,width_to)), tvt.ToTensor(), tvt.Normalize(mean=[0.5], std=[0.5]) ] )
            img_tensor_gray = gray_resized_normalized(smoothed_image_gray)
            ### we do NOT need a smoothed version of the color image:
            img_tensor_color = color_resized_normalized(image_pil_color)
            to_image_xform = tvt.Compose([tvt.ToPILImage()])
            low_res_PIL_image_gray = to_image_xform(img_tensor_gray)
            low_res_PIL_image_color = to_image_xform(img_tensor_color)
        else:
            low_res_PIL_image_gray = smoothed_image_gray.resize((width_to,height_to), Image.ANTIALIAS)
            low_res_PIL_image_color = image_pil_color.resize((width_to,height_to), Image.ANTIALIAS)
        self.low_res_PIL_image_gray = low_res_PIL_image_gray
        self.low_res_PIL_image_color = low_res_PIL_image_color
        self.displayImage6(low_res_PIL_image_gray, "low_res_version_of_original")
        ###   VERY IMPORTANT:  In PIL.Image, the first coordinate refers to the width-wise coordinate
        ###                    and the second coordinate to the height-wise coordinate pointing downwards
        ###   However:         In numpy and tensor based representations, the first coordinate is the
        ###                    height-wise coordinate and the second coordinate the width-wise coordinate.
        ###   Since the tensor operations cause the grayscale image to be represented
        ###   by a 3D array, with the first dimension set to the number of channels
        ###   (which would be 1 for a grayscale image), we need to ignore it:
        img_array = np.asarray(low_res_PIL_image_gray)
        self.im_array = img_array
        arr_height,arr_width = img_array.shape
        print("\n\n\nimage array size: height=%d  width=%d" % (arr_height,arr_width))

        initial_num_graph_vertices = arr_height * arr_width
        print("\n\n\nnumber of vertices in graph: %d" % initial_num_graph_vertices)
        initial_graph_vertices = {i : None for i in range(initial_num_graph_vertices)}
        for i in range(initial_num_graph_vertices):
            h,w  =  i // arr_width, i - (i // arr_width)*arr_width
            initial_graph_vertices[i] = [(h,w)]
      
        initial_graph_edges = {}
        MInt = {}
        for i in range(initial_num_graph_vertices):
            hi,wi =  initial_graph_vertices[i][0]
            for j in range(initial_num_graph_vertices):
                hj,wj =  initial_graph_vertices[j][0]
                if i > j:
                    if abs(hi - hj) <= 1 and abs(wi - wj) <= 1:
                        ###  In order to take care of the error report: "overflow encountered in 
                        ###  ubyte_scalars":
                        ###  Since pixels are stored as the uint8 datatype (which implies that
                        ###  their values are only expected to be between 0 and 255), any 
                        ###  arithmetic on them could violate that range.  So we must first convert
                        ###  into the int datatype:
                        initial_graph_edges["%d,%d" % (i,j) ]  =  abs(int(img_array[hi,wi]) - int(img_array[hj,wj]))
                        MInt[ "%d,%d" % (i,j) ] = kay

        ###   INTERNAL DIFFERENCE property at the initial vertices in the graph
        ###   Internal Difference is defined as the max edge weight between the pixels in the pixel
        ###   blob represented by a graph vertex.
        Int_prop = {v : 0.0 for v in initial_graph_vertices}
        MInt_prop = {v : kay for v in initial_graph_vertices}   
        if self.debug:
            print("\n\n\ninitial graph_vertices: %s" % str(sorted(initial_graph_vertices.items())))
            print("\n\n\nnumber of vertices in initial graph: %d" % len(initial_graph_vertices))
            print("\n\n\ninitial graph_edges: %s" % str(sorted(initial_graph_edges.items())))
            print("\n\n\nnumber of edges in initial graph: %d" % len(initial_graph_edges))
            print("\n\n\ninitial MInt: %s" % str(sorted(MInt.items())))
            print("\n\n\nnumber of edges in initial MInt: %d" % len(MInt))

        initial_graph = (copy.deepcopy(initial_graph_vertices), copy.deepcopy(initial_graph_edges))

        def find_all_connections_for_a_vertex(vert, graph):
            vertices = graph[0]
            edges    = graph[1]
            print("pixels in vertex %d: %s" % (vert, str(vertices[vert])))
            connected_verts_in_graph = []
            for edge in edges:
                end1,end2 = int(edge[:edge.find(',')]), int(edge[edge.find(',')+1 :])                 
                if vert == end1:
                    connected_verts_in_graph.append(end2)
                elif vert == end2:
                    connected_verts_in_graph.append(end1)
            return connected_verts_in_graph

        index_for_new_vertex = len(initial_graph_vertices)
        master_iteration_index = 0
        self.iterations_terminated = False
        ###   graph = (V,E) with both V and E as dictionaries.
        ###   NOTE: The edges E in the graph stand for 'Dif(C1,C2)' in F&H
        def seg_gen( graph, MInt, index_for_new_vertex, master_iteration_index, Int_prop, MInt_prop, kay ):
            print("\n\n\n=========================== Starting iteration %d ==========================" % master_iteration_index)
            V,E = graph
            if num_blobs_wanted is not None and len(initial_graph[0]) > num_blobs_wanted:
                if num_blobs_wanted is not None and len(V) <= num_blobs_wanted: return graph
            if self.debug:
                print("\n\n\nV: %s" % str(V))
                print("\n\n\nE: %s" % str(E))
                print("\n\n\nMInt: %s" % str(MInt))
            max_iterations = self.max_iterations
            print("\n\n\nNumber of region proposals at the current level of merging: %d" % len(V))
            if len(E) == 0:
                print("\n\n\nThe graph has no edges left")
                return graph
            sorted_vals_and_edges = list( sorted( (v,k) for k,v in E.items() ) )
            sorted_edges = [x[1] for x in sorted_vals_and_edges]
            print("\n\n\n[Iter Index: %d] Dissimilarity value associated with the most similar edge: %s" % (master_iteration_index, str(sorted_vals_and_edges[0])))
            print("\nOne dot represents 100 possible merge operations in the graph representation of the image\n")
            edge_counter = 0

            ###  You have to be careful when debugging the code in the following 'for' loop.  The
            ###  problem is that the sorted edge list is made from the original edge list which is
            ###  modified by the code in the 'for' loop.  Let's say that the edge 'u,v' is a good
            ###  candidate for the merging of the pixel blobs corresponding to u and v.  After the
            ###  'for' loop has merged these two blobs corresponding to these two vertices, the 'u'
            ###  and 'v' vertices in the graph do not exist and must be deleted.  Deleting these two
            ###  vertices requires that we must also delete from E all the other edges that connect
            ###  with either u and v.  So if you are not careful, it is possible that in the next
            ###  go-around in the 'for' loop you will run into one of those edges as the next
            ###  candidate for the merging of two vertices.
            for edge in sorted_edges:
                if edge not in E: continue
                edge_counter += 1
                if edge_counter % 100 == 0: 
                    sys.stdout.write(". ")
                    sys.stdout.flush()
                ###  This is the fundamental condition for merging the pixel blobs corresponding to
                ###  two different vertices: The 'Diff' edge weight, which is represented by the
                ###  edge weight E[edge], must be LESS than the minimum of the Internal component
                ###  edge weight, the minimum being over the two vertices for the two pixel blobs.
                if E[edge] >  MInt[edge]: 
                    del E[edge]
                    del MInt[edge]
                    continue 
                ###  Let us now find the identities of the vertices of the edge whose two vertices
                ###  are the best candidates for the merging of the two pixel blobs.
                vert1,vert2 = int(edge[:edge.find(',')]), int(edge[edge.find(',')+1 :])
                if self.debug:
                    print("\n\n\n[Iter Index: %d] The least dissimilar two vertices in the graph are: %s and %s" % 
                                                                              (master_iteration_index, vert1, vert2))
                ###   Since we want to go through all the edges in 'sorted_edges" WHILE we are
                ###   deleting the vertices that are merged and the edges that are no longer
                ###   relevant because of vertex deletion, we need to be careful going forward:
                if (vert1 not in V) or (vert2 not in V): continue
                affected_edges = []
                for edg in E:
                    end1,end2 = int(edg[:edg.find(',')]), int(edg[edg.find(',')+1 :])                
                    if (vert1 == end1) or (vert1 == end2) or (vert2 == end1) or (vert2 == end2):
                        affected_edges.append(edg)
                if self.debug:
                    print("\n\n\naffected edges to be deleted: %s" % str(affected_edges))
                for edg in affected_edges:
                    del E[edg]
                    del MInt[edg]
                merged_blob = V[vert1] + V[vert2]
#                change_flag = True
                if self.debug:
                    print("\n\n\nAdded vertex %d to V" % index_for_new_vertex)
                V[index_for_new_vertex] = merged_blob
                if self.debug:
                    print("\n\n\n[Iter Index: %d] index for new vertex: %d   and the merged blob: %s" % 
                                                   (master_iteration_index, index_for_new_vertex, str(merged_blob)))
                ###   We will now calculate the Int (Internal Difference) and MInt property to be
                ###   to be associated with the newly created vertex in the graph:
                within_blob_edge_weights = []
                for u1 in merged_blob:
                    i = u1[0] * arr_width + u1[1]
                    for u2 in merged_blob:
                        j = u2[0] * arr_width + u2[1]
                        if i > j:
                            ij_key = "%d,%d" % (i,j)
                            if ij_key in initial_graph_edges:
                                within_blob_edge_weights.append( initial_graph_edges[ ij_key ] )
                Int_prop[index_for_new_vertex] = max(within_blob_edge_weights) 
                MInt_prop[index_for_new_vertex] = Int_prop[index_for_new_vertex] + kay / float(len(merged_blob))
                ###   Now we must calculate the new graph edges formed by the connections between the newly
                ###   formed node and all other nodes.  However, we first must delete the two nodes that
                ###   we just merged:
                del V[vert1] 
                del V[vert2]
                del Int_prop[vert1]
                del Int_prop[vert2]
                del MInt_prop[vert1]
                del MInt_prop[vert2]
                if self.debug:
                    print("\n\n\nThe modified vertices: %s" % str(V))
                for v in sorted(V):
                    if v == index_for_new_vertex: continue
                    ###   we need to store the edge weights for the pixel-to-pixel edges
                    ###   in the initial graph with one pixel in the newly constructed
                    ###   blob and other in a target blob
                    pixels_in_v = V[v]
                    for u_pixel in merged_blob:
                        i = u_pixel[0] * arr_width + u_pixel[1]
                        inter_blob_edge_weights = []
                        for v_pixel in pixels_in_v:
                            j = v_pixel[0] * arr_width + v_pixel[1]
                            if i > j: 
                                ij_key = "%d,%d" % (i,j)
                            else:
                                ij_key = "%d,%d" % (j,i)
                            if ij_key in initial_graph_edges:
                                inter_blob_edge_weights.append( initial_graph_edges[ij_key ] )
                        if len(inter_blob_edge_weights) > 0:
                            uv_key = str("%d,%d" % (index_for_new_vertex,v))
                            E[uv_key] = min(inter_blob_edge_weights)                        
                            MInt[uv_key] = min( MInt_prop[index_for_new_vertex], MInt_prop[v] )
                if self.debug:
                    print("\n\n\nAt the bottom of for-loop for edges ---   E: %s" % str(E))
                    print("\n\nMInt: %s" % str(MInt))
                index_for_new_vertex = index_for_new_vertex + 1   
#                if change_flag is False: break
    
            new_graph = (copy.deepcopy(V), copy.deepcopy(E))
            MInt = copy.deepcopy(MInt)
            if self.debug:
                print("\n\n\nnew graph at end of iteration: %s" % str(new_graph))
            if master_iteration_index == max_iterations:
                return new_graph
            else:
                self.iterations_used = master_iteration_index
                master_iteration_index += 1
                if self.iterations_terminated:
                    return new_graph
                else:
                    return seg_gen(new_graph, MInt, index_for_new_vertex, master_iteration_index, Int_prop, MInt_prop, kay)  
        segmented_graph = seg_gen(initial_graph, MInt, index_for_new_vertex, master_iteration_index, Int_prop, MInt_prop, kay)
        if self.debug:
            print("\n\n\nsegmented_graph: %s" % str(segmented_graph))
        bounding_boxes = {}
        total_pixels_in_output = 0
        retained_vertex_list = []
        for vertex in sorted(segmented_graph[0]):
            all_pixels_in_blob = segmented_graph[0][vertex]
            total_pixels_in_output += len(all_pixels_in_blob)
            if len(all_pixels_in_blob) > self.min_size_for_graph_based_blobs:
                print("\n\n\npixels in blob indexed %d: %s" % (vertex, str(segmented_graph[0][vertex])))
                retained_vertex_list.append(vertex)
                height_coords = [p[0] for p in all_pixels_in_blob]
                width_coords  = [p[1] for p in all_pixels_in_blob]
                bb_height_min = min(height_coords)
                bb_height_max = max(height_coords)
                bb_width_min  = min(width_coords)
                bb_width_max  = max(width_coords)
                """
                if (abs(bb_width_max - bb_width_min) <= 2 or abs(bb_height_max - bb_height_min) <= 2): continue
                if abs(bb_width_max - bb_width_min) < 0.1 * abs(bb_height_max - bb_height_min): continue
                if abs(bb_height_max - bb_height_min)  <  0.1 * abs(bb_width_max - bb_width_min): continue
                """
                bounding_boxes[vertex] = [bb_height_min, bb_width_min, bb_height_max, bb_width_max]

        print("\n\n\nTotal number of pixels in output blobs: %d" % total_pixels_in_output)
        title = "graph_based_bounding_boxes"
        mw = Tkinter.Tk()
        winsize_w,winsize_h = None,None
        screen_width,screen_height = mw.winfo_screenwidth(),mw.winfo_screenheight()
        if screen_width <= screen_height:
            winsize_w = int(0.5 * screen_width)
            winsize_h = int(winsize_w * (arr_height * 1.0 / arr_width))            
        else:
            winsize_h = int(0.5 * screen_height)
            winsize_w = int(winsize_h * (arr_width * 1.0 / arr_height))
        scaled_image =  image_pil_color.copy().resize((winsize_w,winsize_h), Image.ANTIALIAS)
        mw.title(title) 
        mw.configure( height = winsize_h, width = winsize_w )         
        canvas = Tkinter.Canvas( mw,                         
                             height = winsize_h,            
                             width = winsize_w,             
                             cursor = "crosshair" )   
        canvas.pack(fill=BOTH, expand=True)
        frame = Tkinter.Frame(mw)                            
        frame.pack( side = 'bottom' )                             
        Tkinter.Button( frame,         
                text = 'Save',                                    
                command = lambda: canvas.postscript(file = title + ".eps") 
              ).pack( side = 'left' )                             
        Tkinter.Button( frame,                        
                text = 'Exit',                                    
                command = lambda: mw.destroy(),                    
              ).pack( side = 'right' )                            
        photo = ImageTk.PhotoImage( scaled_image )
        canvas.create_image(winsize_w//2,winsize_h//2,image=photo)
        scale_w = winsize_w / float(arr_width)
        scale_h = winsize_h / float(arr_height)
        for v in bounding_boxes:
            bb = bounding_boxes[v]
            print("\n\n\nFor region proposal with ID %d, the bounding box is: %s" % (v, str(bb)))
            canvas.create_rectangle( (bb[1]*scale_w,bb[0]*scale_h,(bb[3]+1)*scale_w,(bb[2]+1)*scale_h), width='3', outline='red' )
        canvas.update()
        mw.update()
        print("\n\n\nIterations used: %d" % self.iterations_used)
        print("\n\n\nNumber of region proposals: %d" % len(bounding_boxes))
        mw.mainloop()
        if os.path.isfile(title + ".eps"):
            Image.open(title + ".eps").save(title + ".png")
            os.remove(title + ".eps")

        retained_vertices = {}
        retained_edges = {}
        for u in retained_vertex_list:
            retained_vertices[u] = segmented_graph[0][u]
            for v in retained_vertex_list:
                if u > v:
                    uv_label = "%d,%d"%(u,v)
                    if uv_label in segmented_graph[1]:
                        retained_edges[uv_label] = segmented_graph[1][uv_label]
        output_segmentation_graph = (retained_vertices, retained_edges)

        ###  color_map is a dictionary with blob_ids as keys and the assigned values the color
        ###  assigned to each blob for its visualization
        color_map = {}
        for blob_idx in sorted(output_segmentation_graph[0], key=lambda x: len(output_segmentation_graph[0][x]), reverse=True):
            if blob_idx not in color_map:
                color_map[blob_idx] = (random.randint(0,255), random.randint(0,255),random.randint(0,255))
        return output_segmentation_graph, color_map

    def graph_based_segmentation_for_arrays(self, which_one):
        '''
        This method is provided to enable the user to play with small arrays when
        experimenting with graph-based logic for image segmentation.  At the moment, it
        provides three small arrays, one under the "which_one==1" option, one under the
        "which_one==2" option, and the last under the "which_one==3" option.
        '''
        print("\nExperiments with selective-search logic on made-up arrays")
        kay = self.kay
        if which_one == 1:
            img_array = np.zeros((20,24), dtype = np.float)             ## height=20  width=24
            arr_height,arr_width = img_array.shape
            print("\n\n\nimage array size: height=%d  width=%d" % (arr_height,arr_width))
            for h in range(arr_height):
                for w in range(arr_width):
                    if ((4 < h < 8) or (12 < h < 16)) and ((4 < w < 10) or (14 < w < 20)):
                        img_array[h,w] = 200
            print("\n\n\nimg_array:")
            print(img_array)
            image_pil = Image.fromarray(img_array.astype('uint8'), 'L')
            self.displayImage3(image_pil, "made-up image")
            image_pil.save("array1.png")
        elif which_one == 2:
            img_array = np.zeros((6,10), dtype = np.float)             ## height=6  width=10
            arr_height,arr_width = img_array.shape
            print("\n\n\nimage array size: height=%d  width=%d" % (arr_height,arr_width))
            for h in range(arr_height):
                for w in range(arr_width):
                    if (1 < h < 4) and ((1 < w < 4) or (6 < w < 9)):
                        img_array[h,w] = 128
            print("\n\n\nimg_array:")
            print(img_array)
            image_pil = Image.fromarray(img_array.astype('uint8'), 'L')
            self.displayImage3(image_pil, "made-up image")
            image_pil.save("array2.png")
        else:
            img_array = np.zeros((20,24), dtype = np.float)            ## height=20  width=24
            arr_height,arr_width = img_array.shape
            print("\n\n\nimage array size: height=%d  width=%d" % (arr_height,arr_width))
            for h in range(4,arr_height-4):
                for w in range(4,arr_width-4):
                    img_array[h,w] = 100
            for h in range(8,arr_height-8):
                for w in range(8,arr_width-8):
                    img_array[h,w] = 200
            print("\n\n\nimg_array:")
            print(img_array)
            image_pil = Image.fromarray(img_array.astype('uint8'), 'L')
            self.displayImage3(image_pil, "made-up image")
            image_pil.save("array3.png")

        initial_num_graph_vertices = arr_height * arr_width
        print("\n\n\nnumber of vertices in graph: %d" % initial_num_graph_vertices)
        initial_graph_vertices = {i : None for i in range(initial_num_graph_vertices)}
        for i in range(initial_num_graph_vertices):
            h,w  =  i // arr_width, i - (i // arr_width)*arr_width
            initial_graph_vertices[i] = [(h,w)]
      
        initial_graph_edges = {}
        MInt = {}
        for i in range(initial_num_graph_vertices):
            hi,wi =  initial_graph_vertices[i][0]
            for j in range(initial_num_graph_vertices):
                hj,wj =  initial_graph_vertices[j][0]
                if i > j:
                    if abs(hi - hj) <= 1 and abs(wi - wj) <= 1:
                        initial_graph_edges[ "%d,%d" % (i,j) ]  =  abs(img_array[hi,wi] - img_array[hj,wj])
                        MInt[ "%d,%d" % (i,j) ] = kay

        ###   INTERNAL DIFFERENCE property at the initial vertices in the graph
        ###   Internal Difference is defined as the max edge weight between the pixels in the pixel
        ###   blob represented by a graph vertex.
        Int_prop = {v : 0.0 for v in initial_graph_vertices}
        ###   MInt_prop at each vertex is the Int_prop plus the kay divided by the cardinality of the blob
        MInt_prop = {v : kay for v in initial_graph_vertices}   
        if self.debug:
            print("\n\n\ninitial graph_vertices: %s" % str(sorted(initial_graph_vertices.items())))
            print("\n\n\nnumber of vertices in initial graph: %d" % len(initial_graph_vertices))
            print("\n\n\ninitial graph_edges: %s" % str(sorted(initial_graph_edges.items())))
            print("\n\n\nnumber of edges in initial graph: %d" % len(initial_graph_edges))
            print("\n\n\ninitial MInt: %s" % str(sorted(MInt.items())))
            print("\n\n\nnumber of edges in initial MInt: %d" % len(MInt))

        initial_graph = (copy.deepcopy(initial_graph_vertices), copy.deepcopy(initial_graph_edges))

        def find_all_connections_for_a_vertex(vert, graph):
            vertices = graph[0]
            edges    = graph[1]
            print("pixels in vertex %d: %s" % (vert, str(vertices[vert])))
            connected_verts_in_graph = []
            for edge in edges:
                end1,end2 = int(edge[:edge.find(',')]), int(edge[edge.find(',')+1 :])                 
                if vert == end1:
                    connected_verts_in_graph.append(end2)
                elif vert == end2:
                    connected_verts_in_graph.append(end1)
            return connected_verts_in_graph

        index_for_new_vertex = len(initial_graph_vertices)
        master_iteration_index = 0
        self.iterations_terminated = False

        ###   graph = (V,E) with both V and E as dictionaries.
        ###   NOTE: The edges E in the graph stand for 'Dif(C1,C2)' in F&H
        def seg_gen( graph, MInt, index_for_new_vertex, master_iteration_index, Int_prop, MInt_prop, kay ):
            print("\n\n\n=========================== Starting iteration %d ========================== \n\n\n" % 
                                                                                             master_iteration_index)
            V,E = graph
            if self.debug:
                print("\n\n\nV: %s" % str(V))
                print("\n\n\nE: %s" % str(E))
                print("\n\n\nMInt: %s" % str(MInt))
            max_iterations = self.max_iterations
            print("\n\n\nNumber of region proposals at the current level of merging: %d" % len(V))
            if len(E) == 0:
                print("\n\n\nThe graph has no edges left")
                return graph
            sorted_vals_and_edges = sorted( (v,k) for k,v in E.iteritems() )
            sorted_edges = [x[1] for x in sorted_vals_and_edges]
            print("\n\n\n[Iter Index: %d] Dissimilarity value associated with the most similar edge: %s" % 
                                                              (master_iteration_index, str(sorted_vals_and_edges[0])))
            """    
            if sorted_vals_and_edges[0][0] > 0.5:
                print("\n\n\nIterations terminated at iteration index: %d" % master_iteration_index)
                self.iterations_terminated = True
                return graph
            """
#            change_flag = False
            print("\nOne dot represents TEN possible merge operations in the graph representation of the image\n")
            if self.debug:
                print("\n\n\nBefore entering the edge loop --- sorted_edges: %s" % str(sorted_edges))
                print("\n\n\nBefore entering the edge loop --- E: %s" % str(E))
                print("\n\n\nBefore entering the edge loop --- MInt: %s" % str(MInt))
                print("\n\n\nBefore entering the edge loop --- vertices: %s" % str(V))
            edge_counter = 0
            for edge in sorted_edges:
                if edge not in E: continue
                edge_counter += 1
                if edge_counter % 10 == 0: 
                    sys.stdout.write(". ") 
                    sys.stdout.flush()
                if edge not in MInt:
                    sys.exit("MInt does not have an entry for %s" % edge)
                if edge not in E:
                    sys.exit("\n\n\nE does not have an entry for %s" % edge)
                if E[edge] >  MInt[edge]: 
                    del E[edge]
                    del MInt[edge]
                    continue 
                vert1,vert2 = int(edge[:edge.find(',')]), int(edge[edge.find(',')+1 :])
                if self.debug:
                    print("\n\n\n[Iter Index: %d] The least dissimilar two vertices in the graph are: %s and %s" % 
                                                                                (master_iteration_index, vert1, vert2))
                ###   Since we want to go through all the edges in 'sorted_edges" WHILE we are
                ###   deleting the vertices that are merged and the edges that are no longer
                ###   relevant because of vertex deletion, we need to be careful going forward:
                if (vert1 not in V) or (vert2 not in V): continue
                affected_edges = []
                for edg in E:
                    end1,end2 = int(edg[:edg.find(',')]), int(edg[edg.find(',')+1 :])                
                    if (vert1 == end1) or (vert1 == end2) or (vert2 == end1) or (vert2 == end2):
                        affected_edges.append(edg)
                if self.debug:
                    print("\n\n\naffected edges to be deleted: %s" % str(affected_edges))
                for edg in affected_edges:
                    del E[edg]
                    del MInt[edg]
                merged_blob = V[vert1] + V[vert2]
#                change_flag = True
                if self.debug:
                    print("\n\n\nAdded vertex %d to V" % index_for_new_vertex)
                V[index_for_new_vertex] = merged_blob
                if self.debug:
                    print("\n\n\n[Iter Index: %d] index for new vertex: %d   and the merged blob: %s" % 
                                                     (master_iteration_index, index_for_new_vertex, str(merged_blob)))
                ###   We will now calculate the Int (Internal Difference) and MInt property to be
                ###   to be associated with the newly created vertex in the graph:
                within_blob_edge_weights = []
                for u1 in merged_blob:
                    i = u1[0] * arr_width + u1[1]
                    for u2 in merged_blob:
                        j = u2[0] * arr_width + u2[1]
                        if i > j:
                            ij_key = "%d,%d" % (i,j)
                            if ij_key in initial_graph_edges:
                                within_blob_edge_weights.append( initial_graph_edges[ ij_key ] )
                Int_prop[index_for_new_vertex] = max(within_blob_edge_weights) 
                MInt_prop[index_for_new_vertex] = Int_prop[index_for_new_vertex] + kay / float(len(merged_blob))
                ###   Now we must calculate the new graph edges formed by the connections between the newly
                ###   formed node and all other nodes.  However, we first must delete the two nodes that
                ###   we just merged:
                del V[vert1] 
                del V[vert2]
                del Int_prop[vert1]
                del Int_prop[vert2]
                del MInt_prop[vert1]
                del MInt_prop[vert2]
                if self.debug:
                    print("\n\n\nThe modified vertices: %s" % str(V))
                for v in sorted(V):
                    if v == index_for_new_vertex: continue
                    ###   we need to store the edge weights for the pixel-to-pixel edges
                    ###   in the initial graph with one pixel in the newly constructed
                    ###   blob and other in a target blob
                    pixels_in_v = V[v]
                    for u_pixel in merged_blob:
                        i = u_pixel[0] * arr_width + u_pixel[1]
                        inter_blob_edge_weights = []
                        for v_pixel in pixels_in_v:
                            j = v_pixel[0] * arr_width + v_pixel[1]
                            if i > j: 
                                ij_key = "%d,%d" % (i,j)
                            else:
                                ij_key = "%d,%d" % (j,i)
                            if ij_key in initial_graph_edges:
                                inter_blob_edge_weights.append( initial_graph_edges[ij_key ] )
                        if len(inter_blob_edge_weights) > 0:
                            uv_key = "%d,%d" % (index_for_new_vertex,v)
                            E[uv_key] = min(inter_blob_edge_weights)                        
                            MInt[uv_key] = min( MInt_prop[index_for_new_vertex], MInt_prop[v] )
                if self.debug:
                    print("\n\n\nAt the bottom of for-loop for edges ---   E: %s" % str(E))
                    print("\n\nMInt: %s" % str(MInt))
                index_for_new_vertex = index_for_new_vertex + 1   
#                if change_flag is False: break
            new_graph = (copy.deepcopy(V), copy.deepcopy(E))
            MInt = copy.deepcopy(MInt)
            if self.debug:
                print("\n\n\nnew graph at end of iteration: %s" % str(new_graph))
            if master_iteration_index == max_iterations:
                return new_graph
            else:
                self.iterations_used = master_iteration_index - 1
                master_iteration_index += 1
                if self.iterations_terminated:
                    return new_graph
                else:
                    return seg_gen(new_graph, MInt, index_for_new_vertex, master_iteration_index, Int_prop, MInt_prop, kay)  
        segmented_graph = seg_gen(initial_graph, MInt, index_for_new_vertex, master_iteration_index, Int_prop, MInt_prop, kay)
        if self.debug:
            print("\n\n\nsegmented_graph: %s" % str(segmented_graph))
        bounding_boxes = {}
        total_pixels_in_output = 0
        for vertex in sorted(segmented_graph[0]):
            all_pixels_in_blob = segmented_graph[0][vertex]
            total_pixels_in_output += len(all_pixels_in_blob)
            if len(all_pixels_in_blob) > 1:
                print("\n\n\npixels in blob indexed %d: %s" % (vertex, str(segmented_graph[0][vertex])))
                height_coords = [p[0] for p in all_pixels_in_blob]
                width_coords  = [p[1] for p in all_pixels_in_blob]
                bb_height_min = min(height_coords)
                bb_height_max = max(height_coords)
                bb_width_min  = min(width_coords)
                bb_width_max  = max(width_coords)
                """
                if (abs(bb_width_max - bb_width_min) <= 2 or abs(bb_height_max - bb_height_min) <= 2): continue
                if abs(bb_width_max - bb_width_min) < 0.1 * abs(bb_height_max - bb_height_min): continue
                if abs(bb_height_max - bb_height_min)  <  0.1 * abs(bb_width_max - bb_width_min): continue
                """
                bounding_boxes[vertex] = [bb_height_min, bb_width_min, bb_height_max, bb_width_max]

        print("\n\n\nTotal number of pixels in output blobs: %d" % total_pixels_in_output)
        title = "graph_based_bounding_boxes"
        mw = Tkinter.Tk()
        winsize_x,winsize_y = None,None
        screen_width,screen_height = mw.winfo_screenwidth(),mw.winfo_screenheight()
        if screen_width <= screen_height:
            winsize_w = int(0.5 * screen_width)
            winsize_h = int(winsize_w * (arr_height * 1.0 / arr_width))            
        else:
            winsize_h = int(0.5 * screen_height)
            winsize_w = int(winsize_h * (arr_width * 1.0 / arr_height))
        scaled_image =  image_pil.copy().resize((winsize_w,winsize_h), Image.ANTIALIAS)
        mw.title(title) 
        mw.configure( height = winsize_h, width = winsize_x )         
        canvas = Tkinter.Canvas( mw,                         
                             height = winsize_h,            
                             width = winsize_w,             
                             cursor = "crosshair" )   
        canvas.pack(fill=BOTH, expand=True)
        frame = Tkinter.Frame(mw)                            
        frame.pack( side = 'bottom' )                             
        Tkinter.Button( frame,         
                text = 'Save',                                    
                command = lambda: canvas.postscript(file = title + ".eps") 
              ).pack( side = 'left' )                             
        Tkinter.Button( frame,                        
                text = 'Exit',                                    
                command = lambda: mw.destroy(),                    
              ).pack( side = 'right' )                            
        photo = ImageTk.PhotoImage( scaled_image )
        canvas.create_image(winsize_w//2,winsize_h//2,image=photo)
        scale_w = winsize_w / float(arr_width)
        scale_h = winsize_h / float(arr_height)
        for v in bounding_boxes:
            bb = bounding_boxes[v]
            print("\n\n\nFor region proposal with ID %d, the bounding box is: %s" % (v, str(bb)))
            canvas.create_rectangle( (bb[1]*scale_w,bb[0]*scale_h,(bb[3]+1)*scale_w,(bb[2]+1)*scale_h), width='3', outline='red' )
        canvas.update()
        mw.update()
        print("\n\n\nIterations used: %d" % self.iterations_used)
        print("\n\n\nNumber of region proposals: %d" % len(bounding_boxes))
        mw.mainloop()
        if os.path.isfile(title + ".eps"):
            Image.open(title + ".eps").save(title + ".png")
            os.remove(title + ".eps")
        return segmented_graph[0]


    def extract_image_region_interactively_by_dragging_mouse(self, image_name):
        '''
        This is one method you can use to apply selective search algorithm to just a
        portion of your image.  This method extract the portion you want.  You click
        at the upper left corner of the rectangular portion of the image you are
        interested in and you then drag the mouse pointer to the lower right corner.
        Make sure that you click on "save" and "exit" after you have delineated the
        area.
        '''
        global delineator_image      ### global so that methods like _on_mouse_motion etc can access it
        global delineator_polygon    ###                  """
        print("Drag the mouse pointer to delineate the portion of the image you want to extract:")
        RegionProposalGenerator.image_portion_delineation_coords = []
        pil_image = Image.open(image_name).convert("L")
        RegionProposalGenerator.image_type = "L"
        image_portion_delineation_coords = []
        mw = Tkinter.Tk() 
        mw.title("Click and then drag the mouse pointer --- THEN CLICK SAVE and EXIT")
        width,height = pil_image.size

        screen_width,screen_height = mw.winfo_screenwidth(),mw.winfo_screenheight()
        if screen_width <= screen_height:
            winsize_x = int(0.5 * screen_width)
            winsize_y = int(winsize_x * (height * 1.0 / width))            
        else:
            winsize_y = int(0.5 * screen_height)
            winsize_x = int(winsize_y * (width * 1.0 / height))
        display_pil_image = pil_image.resize((winsize_x,winsize_y), Image.ANTIALIAS)
        scale_w = width / float(winsize_x)
        scale_h = height / float(winsize_y)
        delineator_image =  display_pil_image.copy()
        extracted_image =  display_pil_image.copy()
        self.extracted_image_portion_file_name = os.path.basename(image_name)
        mw.configure(height = winsize_y, width = winsize_x) 
        photo_image = ImageTk.PhotoImage(display_pil_image)
        canvasM = Tkinter.Canvas( mw,   
                                  width = winsize_x,
                                  height =  winsize_y,
                                  cursor = "crosshair" )  
        canvasM.pack( side = 'top' )   
        frame = Tkinter.Frame(mw)  
        frame.pack( side = 'bottom' ) 
        Tkinter.Button( frame, 
                        text = 'Save', 
                        command = lambda: RegionProposalGenerator.extracted_image.save(self.extracted_image_portion_file_name) 
                      ).pack( side = 'left' )  
        Tkinter.Button( frame,  
                        text = 'Exit',
                        command = lambda: mw.destroy()
                      ).pack( side = 'right' )  
        canvasM.bind("<ButtonPress-1>", lambda e: self._start_mouse_motion(e, delineator_image))
        canvasM.bind("<ButtonRelease-1>", lambda e:self._stop_mouse_motion(e, delineator_image))
        canvasM.bind("<B1-Motion>", lambda e: self._on_mouse_motion(e, delineator_image))
        canvasM.create_image( 0,0, anchor=NW, image=photo_image)
        canvasM.pack(fill=BOTH, expand=1)
        mw.mainloop()       
        self.displayImage3(RegionProposalGenerator.extracted_image, "extracted image -- close window when done viewing")
        extracted_image = RegionProposalGenerator.extracted_image
        width_ex, height_ex = extracted_image.size
        extracted_image = extracted_image.resize( (int(width_ex * scale_w), int(height_ex * scale_h)), Image.ANTIALIAS )
        self.displayImage6(extracted_image, "extracted image")
        return extracted_image


    def extract_image_region_interactively_through_mouse_clicks(self, image_file):
        '''
        This method allows a user to use a sequence of mouse clicks in order to specify a
        region of the input image that should be subject to further processing.  The
        mouse clicks taken together define a polygon. The method encloses the
        polygonal region by a minimum bounding rectangle, which then becomes the new
        input image for the rest of processing.
        '''
        global delineator_image
        global delineator_coordinates
        print("Click mouse in a clockwise fashion to specify the portion you want to extract:")
        RegionProposalGenerator.image_portion_delineation_coords = []

        if os.path.isfile(image_file):
            pil_image = Image.open(image_file).convert("L")
        else:
            sys.exit("the image file %s does not exist --- aborting" % image_file)
        RegionProposalGenerator.image_type = "L"
        mw = Tkinter.Tk() 
        mw.title("Place mouse clicks clockwise --- THEN CLICK SAVE and EXIT")
        width,height = pil_image.size
        screen_width,screen_height = mw.winfo_screenwidth(),mw.winfo_screenheight()
        if screen_width <= screen_height:
            winsize_x = int(0.5 * screen_width)
            winsize_y = int(winsize_x * (height * 1.0 / width))            
        else:
            winsize_y = int(0.5 * screen_height)
            winsize_x = int(winsize_y * (width * 1.0 / height))
        display_pil_image = pil_image.resize((winsize_x,winsize_y), Image.ANTIALIAS)
        scale_w = width / float(winsize_x)
        scale_h = height / float(winsize_y)
        delineator_image =  display_pil_image.copy()
        extracted_image =  display_pil_image.copy()
        self.extracted_image_portion_file_name = "image_portion_of_" + image_file
        mw.configure(height = winsize_y, width = winsize_x) 
        photo_image = ImageTk.PhotoImage(display_pil_image)
        canvasM = Tkinter.Canvas( mw,   
                                  width = width,
                                  height =  height,
                                  cursor = "crosshair" )  
        canvasM.pack( side = 'top' )   
        frame = Tkinter.Frame(mw)  
        frame.pack( side = 'bottom' ) 
        Tkinter.Button( frame, 
                        text = 'Save', 
                         command = RegionProposalGenerator._extract_and_save_image_portion_polygonal
                      ).pack( side = 'left' )  
        Tkinter.Button( frame,  
                        text = 'Exit',
                        command = lambda: mw.destroy()
                      ).pack( side = 'right' )  
        canvasM.bind("<Button-1>", lambda e: self._image_portion_delineator(e, delineator_image))
        canvasM.create_image( 0,0, anchor=NW, image=photo_image)
        canvasM.pack(fill=BOTH, expand=1)
        mw.mainloop()       
        self.displayImage3(RegionProposalGenerator.extracted_image, "extracted image -- close window when done viewing")
        extracted_image = RegionProposalGenerator.extracted_image
        width_ex, height_ex = extracted_image.size
        extracted_image = extracted_image.resize( (int(width_ex * scale_w), int(height_ex * scale_h)), Image.ANTIALIAS )
        self.displayImage6(extracted_image, "extracted image")
        return extracted_image


    def extract_rectangular_masked_segment_of_image(self, horiz_start, horiz_end, vert_start, vert_end):
        '''
        Keep in mind the following convention used in the PIL's Image class: the first
        coordinate in the args supplied to the getpixel() and putpixel() methods is for
        the horizontal axis (the x-axis, if you will) and the second coordinate for the
        vertical axis (the y-axis).  On the other hand, in the args supplied to the
        array and matrix processing functions, the first coordinate is for the row
        index (meaning the vertical) and the second coordinate for the column index
        (meaning the horizontal).  In what follows, I use the index 'i' with its
        positive direction going down for the vertical coordinate and the index 'j'
        with its positive direction going to the right as the horizontal coordinate. 
        The origin is at the upper left corner of the image.
        '''
        masked_image = self.original_im.copy()
        width,height = masked_image.size 
        mask_array = np.zeros((height, width), dtype="float")
        for i in range(0, height):
            for j in range(0, width):
                if (vert_start < i < vert_end) and (horiz_start < j < horiz_end):
                    mask_array[(i,j)] = 1
        self._display_and_save_array_as_image( mask_array, "_mask__" )
        for i in range(0, height):
            for j in range(0, width):
                if mask_array[(i,j)] == 0:
                    masked_image.putpixel((j,i), (0,0,0)) 
        self.displayImage3(masked_image, "a segment of the image")

    def displayImage(self, argimage, title=""):
        '''
        Displays the argument image.  The display stays on for the number of seconds
        that is the first argument in the call to tk.after() divided by 1000.
        '''
        width,height = argimage.size
        winsize_x,winsize_y = width,height
        if width > height:
            winsize_x = 600
            winsize_y = int(600.0 * (height * 1.0 / width))
        else:
            winsize_y = 600
            winsize_x = int(600.0 * (width * 1.0 / height))
        display_image = argimage.resize((winsize_x,winsize_y), Image.ANTIALIAS)
        tk = Tkinter.Tk()
        tk.title(title)   
        frame = Tkinter.Frame(tk, relief=RIDGE, borderwidth=2)
        frame.pack(fill=BOTH,expand=1)
        photo_image = ImageTk.PhotoImage( display_image )
        label = Tkinter.Label(frame, image=photo_image)
        label.pack(fill=X, expand=1)
        tk.after(1000, self._callback, tk)    # display will stay on for just one second
        tk.mainloop()

    def displayImage2(self, argimage, title=""):
        '''
        Displays the argument image.  The display stays on until the user closes the
        window.  If you want a display that automatically shuts off after a certain
        number of seconds, use the previous method displayImage().
        '''
        width,height = argimage.size
        winsize_x,winsize_y = width,height
        if width > height:
            winsize_x = 600
            winsize_y = int(600.0 * (height * 1.0 / width))
        else:
            winsize_y = 600
            winsize_x = int(600.0 * (width * 1.0 / height))
        display_image = argimage.resize((winsize_x,winsize_y), Image.ANTIALIAS)
        tk = Tkinter.Tk()
        tk.title(title)   
        frame = Tkinter.Frame(tk, relief=RIDGE, borderwidth=2)
        frame.pack(fill=BOTH,expand=1)
        photo_image = ImageTk.PhotoImage( display_image )
        label = Tkinter.Label(frame, image=photo_image)
        label.pack(fill=X, expand=1)
        tk.mainloop()

    def displayImage3(self, argimage, title=""):
        '''
        Displays the argument image (which must be of type Image) in its actual size.  The 
        display stays on until the user closes the window.  If you want a display that 
        automatically shuts off after a certain number of seconds, use the method 
        displayImage().
        '''
        width,height = argimage.size
        tk = Tkinter.Tk()
        winsize_x,winsize_y = None,None
        screen_width,screen_height = tk.winfo_screenwidth(),tk.winfo_screenheight()
        if screen_width <= screen_height:
            winsize_x = int(0.5 * screen_width)
            winsize_y = int(winsize_x * (height * 1.0 / width))            
        else:
            winsize_y = int(0.5 * screen_height)
            winsize_x = int(winsize_y * (width * 1.0 / height))
        display_image = argimage.resize((winsize_x,winsize_y), Image.ANTIALIAS)
        tk.title(title)   
        frame = Tkinter.Frame(tk, relief=RIDGE, borderwidth=2)
        frame.pack(fill=BOTH,expand=1)
        photo_image = ImageTk.PhotoImage( display_image )
        label = Tkinter.Label(frame, image=photo_image)
        label.pack(fill=X, expand=1)
        tk.mainloop()

    def displayImage4(self, argimage, title=""):
        '''
        Displays the argument image (which must be of type Image) in its actual size without 
        imposing the constraint that the larger dimension of the image be at most half the 
        corresponding screen dimension.
        '''
        width,height = argimage.size
        tk = Tkinter.Tk()
        tk.title(title)   
        frame = Tkinter.Frame(tk, relief=RIDGE, borderwidth=2)
        frame.pack(fill=BOTH,expand=1)
        photo_image = ImageTk.PhotoImage( argimage )
        label = Tkinter.Label(frame, image=photo_image)
        label.pack(fill=X, expand=1)
        tk.mainloop()

    def displayImage5(self, argimage, title=""):
        '''
        This does the same thing as displayImage4() except that it also provides for
        "save" and "exit" buttons.  This method displays the argument image with more 
        liberal sizing constraints than the previous methods.  This method is 
        recommended for showing a composite of all the segmented objects, with each
        object displayed separately.  Note that 'argimage' must be of type Image.
        '''
        width,height = argimage.size
        winsize_x,winsize_y = None,None
        mw = Tkinter.Tk()
        screen_width,screen_height = mw.winfo_screenwidth(),mw.winfo_screenheight()
        if screen_width <= screen_height:
            winsize_x = int(0.8 * screen_width)
            winsize_y = int(winsize_x * (height * 1.0 / width))            
        else:
            winsize_y = int(0.8 * screen_height)
            winsize_x = int(winsize_y * (width * 1.0 / height))
        mw.configure(height = winsize_y, width = winsize_x)         
        mw.title(title)   
        canvas = Tkinter.Canvas( mw,                         
                             height = winsize_y,
                             width = winsize_x,
                             cursor = "crosshair" )   
        canvas.pack( side = 'top' )                               
        frame = Tkinter.Frame(mw)                            
        frame.pack( side = 'bottom' )                             
        Tkinter.Button( frame,         
                text = 'Save',                                    
                command = lambda: canvas.postscript(file = title + ".eps") 
              ).pack( side = 'left' )                             
        Tkinter.Button( frame,                        
                text = 'Exit',                                    
                command = lambda: mw.destroy(),                    
              ).pack( side = 'right' )                            
        photo = ImageTk.PhotoImage(argimage.resize((winsize_x,winsize_y), Image.ANTIALIAS))
        canvas.create_image(winsize_x/2,winsize_y/2,image=photo)
        mw.mainloop()
        if os.path.isfile(title + ".eps"):
            Image.open(title + ".eps").save(title + ".png")
            os.remove(title + ".eps")

    def displayImage6(self, argimage, title=""):
        '''
        For the argimge which must be of type PIL.Image, this does the same thing as
        displayImage3() except that it also provides for "save" and "exit" buttons.
        '''
        width,height = argimage.size
        mw = Tkinter.Tk()
        winsize_x,winsize_y = None,None
        screen_width,screen_height = mw.winfo_screenwidth(),mw.winfo_screenheight()
        if screen_width <= screen_height:
            winsize_x = int(0.5 * screen_width)
            winsize_y = int(winsize_x * (height * 1.0 / width))            
        else:
            winsize_y = int(0.5 * screen_height)
            winsize_x = int(winsize_y * (width * 1.0 / height))
        display_image = argimage.resize((winsize_x,winsize_y), Image.ANTIALIAS)
        mw.title(title)   
        canvas = Tkinter.Canvas( mw,                         
                             height = winsize_y,
                             width = winsize_x,
                             cursor = "crosshair" )   
        canvas.pack( side = 'top' )                               
        frame = Tkinter.Frame(mw)                            
        frame.pack( side = 'bottom' )                             
        Tkinter.Button( frame,         
                text = 'Save',                                    
                command = lambda: canvas.postscript(file = title + ".eps") 
              ).pack( side = 'left' )                             
        Tkinter.Button( frame,                        
                text = 'Exit',                                    
                command = lambda: mw.destroy(),                    
              ).pack( side = 'right' )                            
        photo = ImageTk.PhotoImage(argimage.resize((winsize_x,winsize_y), Image.ANTIALIAS))
        canvas.create_image(winsize_x/2,winsize_y/2,image=photo)
        mw.mainloop()
        if os.path.isfile(title + ".eps"):
            Image.open(title + ".eps").save(title + ".png")
            os.remove(title + ".eps")

    @staticmethod    
    def _start_mouse_motion(evt, input_image):
        global delineator_image
        display_width, display_height = delineator_image.size
        canvasM = evt.widget   
        markX, markY = evt.x, evt.y   
        RegionProposalGenerator.image_portion_delineation_coords.append((markX,markY))
        print("Button pressed at: x=%s  y=%s\n" % (markX, markY)) 
        canvasM.create_oval( markX-5, markY-5, markX+5, markY+5, outline="red", fill="green", width = 2 )  

    @staticmethod    
    def _stop_mouse_motion(evt, input_image):
        global delineator_image
        display_width, display_height = delineator_image.size
        canvasM = evt.widget   
        markX, markY = evt.x, evt.y   
        RegionProposalGenerator.image_portion_delineation_coords.append((markX,markY))
        print("Button pressed at: x=%s  y=%s\n" % (markX, markY))
        points = RegionProposalGenerator.image_portion_delineation_coords
        canvasM.create_rectangle(points[0][0], points[0][1], points[-1][0], points[-1][1], outline="red", fill="green", width = 2 ) 
        RegionProposalGenerator.extracted_image = RegionProposalGenerator._extract_image_portion_rectangular()

    @staticmethod    
    def _on_mouse_motion(evt, input_image):
        global delineator_image
        display_width, display_height = delineator_image.size
        canvasM = evt.widget   
        markX, markY = evt.x, evt.y   
        RegionProposalGenerator.image_portion_delineation_coords.append((markX,markY))
        points = RegionProposalGenerator.image_portion_delineation_coords
        canvasM.create_rectangle(points[0][0], points[0][1], points[-1][0], points[-1][1], outline="red", fill="green", width = 2 ) 

    @staticmethod    
    def _image_portion_delineator(evt, input_image):
        global delineator_image
        display_width, display_height = delineator_image.size
        canvasM = evt.widget   
        markX, markY = evt.x, evt.y   
        RegionProposalGenerator.image_portion_delineation_coords.append((markX,markY))
        print("Button pressed at: x=%s  y=%s\n" % (markX, markY)) 
        canvasM.create_oval( markX-10, markY-10, markX+10, markY+10, outline="red", fill="green", width = 2 )  

    @staticmethod    
    def _extract_image_portion_rectangular():
        '''
        This extracts a rectangular region of the image as specified by dragging the mouse pointer
        from the upper left corner of the region to its lower right corner.  After extracting the
        region, it sets the 'original_im' and 'data_im' attributes of the RegionProposalGenerator
        instance to the region extracted.
        '''
        global delineator_image
        width,height = delineator_image.size
        polygon = RegionProposalGenerator.image_portion_delineation_coords
        extracted_width = polygon[-1][0] - polygon[0][0]
        extracted_height = polygon[-1][1] - polygon[0][1]
        extracted_image = Image.new(RegionProposalGenerator.image_type, (extracted_width,extracted_height), (0))
        for x in range(0, extracted_width):        
            for y in range(0, extracted_height):
                extracted_image.putpixel((x,y), delineator_image.getpixel((polygon[0][0]+x, polygon[0][1]+y)))
        return extracted_image

    @staticmethod    
    def _extract_and_save_image_portion_polygonal():
        '''
        This extracts a polygonal region of the image as specified by clicking the mouse in a clockwise
        direction.  After extracting the region, it sets the 'original_im' and 'data_im' attributes of 
        the RegionProposalGenerator instance to the minimum bounding rectangle portion of the original 
        image that encloses the polygonal --- with the pixels outside the polygonal area set to 0.
        '''
        global delineator_image
        width,height = delineator_image.size
        polygon = RegionProposalGenerator.image_portion_delineation_coords
        if len(polygon) <= 2:
            sys.exit("You need MORE THAN TWO mouse clicks (in a clockwise fashion) to extract a region --- aborting!")
        x_min,x_max = min([x for (x,y) in polygon]),max([x for (x,y) in polygon])
        y_min,y_max = min([y for (x,y) in polygon]),max([y for (x,y) in polygon]) 
        extracted_width = x_max - x_min
        extracted_height = y_max - y_min
        extracted_image = Image.new(RegionProposalGenerator.image_type, (extracted_width,extracted_height), (0))
        polygon = [(x - x_min, y - y_min) for (x,y) in polygon]
        for x in range(0, extracted_width):        
            for y in range(0, extracted_height):
                number_of_crossings = 0
                raster_line = (0,y,x,y)
                for l in range(0,len(polygon)-1):
                    line = (polygon[l][0],polygon[l][1],polygon[l+1][0],polygon[l+1][1])
                    if _line_intersection(raster_line, line):
                        number_of_crossings += 1
                last_line = (polygon[l+1][0],polygon[l+1][1],polygon[0][0],polygon[0][1])
                number_of_crossings += _line_intersection(raster_line, last_line)
                if number_of_crossings % 2 == 1:
                    extracted_image.putpixel((x,y), delineator_image.getpixel((x+x_min, y + y_min)))
        RegionProposalGenerator.extracted_image = extracted_image

#______________________  Private Methods of the RegionProposalGenerator Class  ________________

    def _callback(self,arg):                       ## needed in Test directory for flashing an
        arg.destroy()                              ##  image momentarily


#_________________________  End of RegionProposalGenerator Class Definition ___________________________



#______________________________    Test code follows    _________________________________

if __name__ == '__main__': 
    pass
