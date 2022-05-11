
This Greek Character Does Not Exist - Deployment Demonstration

By: Matthew Swindall

For: CSCI 7850 - Deep Learning - Term Project - MTSU - Fall 2021

Abstract --------------------------------------------------------------

Generative adversarial neural networks (GANs) are
increasingly utilized to increase sample size for datasets with
class imbalance or insufficient data for training. This is a boon for
challenging datasets such as those from crowdsourcing initiatives.
The AL ALL dataset is an ideal testing ground for such a
technique as the sub-samples range in size from 62 to over
46,000. Here, we use the PyTorch implementation of StyleGAN2
to double some of the smallest samples in AL ALL, and train
a classic CNN and a ResNet on the new dataset comparing the
results to the same models trained with the original data. Modest
per-character accuracy increases of 8% to 12% were achieved.
Further improvements may be possible by utilizing attention
based architectures.

-----------------------------------------------------------------------

The deployment consists of a helper library, deploy.py, as the backend with the Jupyter
notebook, deploy.ipynb, as the frontend. To run the demonstration, simply open the notebook
in an standard Anaconda environment with the necessary libraries. Further version and library 
information can be found below.

The demonstration consists of several instances of the deploy() function. This function takes 3
arguments: architecture, trained model, and image directory. The architecture options include 
"resnet" and "cnn" which are the 2 architectures used in my project. Trained model options include 
"all" and "synth", corresponding to the architecture trained on the AL_ALL or AL_SYNTH datasets.
The third argument is simply the path to a directory containing images. Images must be 70x70 pixels.
Two directories of sample images are included. The "real/" directory contains images from the 
open-source AL_PUB dataset available at https://data.cs.mtsu.edu/al-pub. The "fake/" directory contains
synthetic images from the AL_SYNTH dataset (not currently available to the public). Both sample 
image directories contain 5 images each of psi and xi. All possible options have been included in the
notebook. One only needs to uncomment the desired configuration and run the cell, or copy the 
function call to a separate cell.

!!! This Repository has been verified to work on the MTSU BIOSIM cluster after installing openCV !!! 


Python Enviroment: 
	Anaconda with Python 3.8.8

Necessary Python Libraries:
	numpy (usually installs as a dependency of TensorFlow)
	matplotlib
	opencv-python
	tensorflow 2.5.0 (theoretically any vesion of tensorflow 2 should work)
	
