#!/usr/bin/env python

##  verify_with_torchnn.py

"""
The purpose of this script is just to verify that the results obtained with the 
following scripts:

       one_neuron_classifier.py
and

       multi_neuron_classifier.py

are not too bizarre --- in the sense that if "verify_with_torchnn.py" shows decreasing loss
with iterations during training, we would want to the above two scripts to do the same.

Obviously, the actual performance you would see with "verify_with_torchnn.py" is bound to
be superior to what you will get with the two handcrafted networks in the two scripts named
above.  That would be the case for the simple reason that those two scripts do not use any 
optimization at all during stochastic gradient descent.

In you are trying to verify the behavior of "one_neuron.py", make the call in line (A) below.

On the other hand, if you are trying verify the behavior of "multi_neuron.py", make the call
in line (B) below.
"""

import random
import numpy
import torch
import os

import sys
sys.path.append("..")

seed = 0           
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

from ComputationalGraphPrimer import *

cgp = ComputationalGraphPrimer(
               expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],   # Only used to determine the data dimensionality
               dataset_size = 5000,
               # learning_rate = 1e-6,              # For the multi-neuron option below
              # learning_rate = 1e-3,             # For the one-neuron option below
              learning_rate = 5 * 1e-2,         # Also for the one-neuron option below
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
      )


# cgp = ComputationalGraphPrimer(
#                num_layers = 3,
#                layers_config = [4,2,1],                         # num of nodes in each layer
#                expressions = ['xw=ap*xp+aq*xq+ar*xr+as*xs',
#                               'xz=bp*xp+bq*xq+br*xr+bs*xs',
#                               'xo=cp*xw+cq*xz'],
#                output_vars = ['xo'],
#                dataset_size = 5000,
#                learning_rate = 1e-3,
#                training_iterations = 40000,
#                batch_size = 8,
#                display_loss_how_often = 100,
# )

##  This call is needed for generating the training data:

# multilayer
# cgp.parse_multi_layer_expressions()
# training_data = cgp.gen_training_data()
# cgp.run_training_with_torchnn('multi_neuron', training_data)                 ## (B)

# one neuran
cgp.parse_expressions()                               
training_data = cgp.gen_training_data()
cgp.run_training_with_torchnn('one_neuron', training_data)                  ## (A)


