# Application documentation

## Overview
This folder contains the main goal of this thesis: an application for deepfake detection and testing the resiliance of facial recognition algorithms against deepfakes. It is a console application and in this documentation, we will discuss how to install it and run it. At the end of the documentation, there are some examples demonstrating how to run this application.

In this folder, you can also find files from FaceFusion, which is a deepfake program. Here is the source code: https://github.com/facefusion/facefusion. Files that i did not created: facefusion folder .flake8, facefusion.ini, install.py, LICENSE.md, mypy.ini, run.py.  

## Instalation for GPU support
Before installing the environment, you need to decide whether you want to use GPU support for the TensorFlow library. If you decide to use GPU, you need to install several dependencies, including CUDA and cuDNN, according to the version of TensorFlow. You can see the requirements in the TensorFlow documentation.

Here for windows: https://www.tensorflow.org/install/source_windows  

Here for linux/macOS: https://www.tensorflow.org/install/source  

We had installed TensorFlow 2.10.0, CUDA 11.2 and cudNN 11.2-windows-x64-v8.1.1.33.

## Setting the enviroment
The setting of enviroment will be done by anaconda so we recommend to install it before. If done than the first step is to create enviroment and activate it:  

**conda create --name deep_app python=3.10**

**conda activate deep_app**

Now we need to install libraries:

**pip install -r requirements.txt**

After this step, the application should work. However, deepfake generation requires more dependencies. We recommend to check FaceFusion installation, where you will find all necessities here: https://docs.facefusion.io/installation. Only the first step is needed; the second step is already set up. The third step is optional, and the fourth step can be skipped. Then, as the fifth step suggests, you need to type:

**python install.py**

If everything is done correctly, you should be able to use all features of the application.

## Examples
If you want to see the description of application you can use argument '-h'/'--help'

**python app.py -h**

Here we provide some test examples of application:

**python app.py -m vgg Deep1_4.jpg**

**python app.py -m generate CFP_2.jpg vzor1.jpg**

**python app.py -m generate Origin_711.jpg vzor15.jpg -r ArcFace -v cos -c**

**python app.py -m siamese Origin_1001.jpg Deep4_1001.jpg**

**python app.py -m siamese CFP_1.jpg CFP_2.jpg -r DeepFace -v l2**

Remember that the vgg model uses only one input image, and the siamese model requires two images. For the siamese model, you should place the original/reference image first and then the potential deepfake. If they are swapped, your output of the siamese model will probably not be correct. For facial recognition algorithms, it should not be a problem if the images are swapped.