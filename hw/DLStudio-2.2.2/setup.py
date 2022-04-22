
### setup.py

from setuptools import setup, find_packages
import sys, os

setup(name='DLStudio',
      version='2.2.2',
      author='Avinash Kak',
      author_email='kak@purdue.edu',
      maintainer='Avinash Kak',
      maintainer_email='kak@purdue.edu',
      url='https://engineering.purdue.edu/kak/distDLS/DLStudio-2.2.2.html',
      download_url='https://engineering.purdue.edu/kak/distDLS/DLStudio-2.2.2.tar.gz',
      description='An educational module to make it easier to design experimental deep-learning networks in PyTorch',
      long_description='''

Consult the module API page at

      https://engineering.purdue.edu/kak/distDLS/DLStudio-2.2.2.html

for all information related to this module, including the information about
the latest changes to the code.  

::

      convo_layers_config = "1x[128,3,3,1]-MaxPool(2) 1x[16,5,5,1]-MaxPool(2)"
      fc_layers_config = [-1,1024,10]
      
      dls = DLStudio(
                        dataroot = "/home/kak/ImageDatasets/CIFAR-10/",
                        image_size = [32,32],
                        convo_layers_config = convo_layers_config,
                        fc_layers_config = fc_layers_config,
                        path_saved_model = "./saved_model",
                        momentum = 0.9,
                        learning_rate = 1e-3,
                        epochs = 2,
                        batch_size = 4,
                        classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck'),
                        use_gpu = True,
                        debug_train = 0,
                        debug_test = 1
                    )
      
      configs_for_all_convo_layers = dls.parse_config_string_for_convo_layers()
      convo_layers = dls.build_convo_layers2( configs_for_all_convo_layers )
      fc_layers = dls.build_fc_layers()
      model = dls.Net(convo_layers, fc_layers)
      dls.show_network_summary(model)
      dls.load_cifar_10_dataset()
      dls.run_code_for_training(model)
      dls.run_code_for_testing(model)

      
          ''',

      license='Python Software Foundation License',
      keywords='PyTorch programming',
      platforms='All platforms',
      classifiers=['Topic :: Scientific/Engineering', 'Programming Language :: Python :: 3.8'],
      packages=['DLStudio','AdversarialLearning','Seq2SeqLearning','DataPrediction','Transformers']
)
