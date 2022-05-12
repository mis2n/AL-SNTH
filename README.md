
<h1>Dataset Augmentation in Papyrology with Generative Models: A Study of Synthetic Ancient Greek Character Images</h1>

<h2>By: 	Matthew I. Swindall , Timothy Player , Ben Keener , Alex C. Williams ,James H. Brusuelas , Federica Nicolardi , Marzia D’Angelo ,
	Claudio Vergara , Michael McOsker and John F. Wallin</h2>

<h3>IJCAI-2022 Supplement - Catagorical Models</h3>

<h3>Classification Models and Inference Demonstration Jupyter Notebook</h3>


<h3>Abstract</h3>

Character recognition models rely substantially on image datasets that maintain a balance of class samples. However, achieving a balance of classes is particularly challenging for ancient manuscript contexts as character instances may be significantly limited. In this paper, we present findings from a study that assess the efficacy of using synthetically generated character instances to augment an existing dataset of ancient Greek character images for use in machine learning models. We complement our model exploration by engaging professional papyrologists to better understand the practical opportunities afforded by synthetic instances. Our results
suggest that synthetic instances improve model performance for limited character classes, and may have unexplored effects on character classes more generally. We also find that trained papyrologists are unable to distinguish between synthetic and non-synthetic images and regard synthetic instances as valuable assets for professional and educational contexts. We conclude by discussing the practical implications of our research.

<h3>               ________________________________________________________________________________</h3>

The models trained for our paper are made availabl here openly for public used. The inference deployment consists of a helper library, deploy.py, as the backend with the Jupyter notebook, deploy.ipynb, as the frontend. To run the demonstration, simply open the notebook in an standard Anaconda environment with the necessary libraries. Further version and library information can be found below.

The demonstration consists of several instances of the deploy() function. This function takes 3 arguments: architecture, trained model, and image directory. The architecture options include  "resnet" and "cnn" which are the 2 architectures used in my project. Trained model options include "all" and "synth", corresponding to the architecture trained on the AL_ALL or AL_SYNTH datasets. The third argument is simply the path to a directory containing images. Images must be 70x70 pixels.
Two directories of sample images are included. The "real/" directory contains images from the  open-source AL_PUB dataset available at https://data.cs.mtsu.edu/al-pub. The "fake/" directory contains synthetic images from the AL_SYNTH dataset (not currently available to the public). Both sample  image directories contain 5 images each of psi and xi. All possible options have been included in the notebook. One only needs to uncomment the desired configuration and run the cell, or copy the 
function call to a separate cell.

Python Enviroment: 
	Anaconda with Python 3.8.8

Necessary Python Libraries:
	numpy (usually installs as a dependency of TensorFlow)
	matplotlib
	opencv-python
	tensorflow 2.5.0 (theoretically any vesion of tensorflow 2 should work)
	
