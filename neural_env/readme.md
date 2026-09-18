# Neural enviroment README

## Overview
This folder contains most of the work during the thesis. There were creations, validations and debugging of all tested or created models.

## Structure
- **bio**: Contains images used for comparison with the BioID deepfake detector. Source: [BioID Website](https://www.bioid.com/)
- **data**: Cropped data from the dataset, divided into "Original" and "Deepfake" folders.
- **models**: Contains the trained models, including two best-performing models, which are saved there on storage medium.
- **siam_graphs**: Graphs for every tested configuration of the siamese architecture.
- **vgg_resnet_graphs**: Graphs for every tested configuration of single-input neural networks.
- **meta-graphs**: Script for generating graphs from output files from Metacentrum
- **neural_start**: The initial script used in this thesis. It begins with the installation of libraries into the environment and focuses on single-input neural networks. Also contains evaluations of models.
- **result.out**: Example output file from Metacentrum. This one was created using the ResNet50 architecture for the siamese model.
- **siamese**: Contains the implementation of the siamese neural network. Some parts are simplified to enable testing locally. Also contains evaluations of models.

## Usage Instructions
To use the scipts we recommend to use anaconda or at least create own enviroment.  
The most important library is TensorFlow to have installed.  
All others as numpy, PIL or cv2 should be installed within the python notebooks scripts.

For more informations, refer to the comments within the scripts.