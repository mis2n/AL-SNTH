'''
Model Inference Demonstration Functions

Dataset Augmentation in Papyrology with Generative Models: A Study of Synthetic Ancient Greek Character Images

IJCAI-2022 Supplement
'''



import os
import sys
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# This line hides version warnings
tf.get_logger().setLevel('ERROR')

# List of classes in datasets
classes = ['Alpha', 'Beta', 'Chi', 'Delta', 'Epsilon', 'Eta', 'Gamma', 'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Omega', 'Omicron', 'Phi', 'Pi', 'Psi' ,'Rho', 'Sigma', 'Tau', 'Theta', 'Upsilon', 'Xi', 'Zeta']

# Function reads in images from directory and converts to tensor
def prepdata(path):
    images = []
    for (root, dirs, files) in os.walk(path, topdown=True):
        for fname in files:
            images.append(str(os.path.join(root, fname)))
    x = []
    for image in images:
        im = cv2.imread(image, cv2.IMREAD_COLOR)
        x.append(im)
    X = np.array(x)
    return X, images

# Function funs model inference (predictions) and produces plot with image, prediction, and probablility
def predict(X, model, files):
    predictions = model.predict(X)
    for i in range(len(predictions)):
        print("Image File: " + files[i])
        label = int(np.argmax(predictions[i]))
        prob = (predictions[i][label]) * 100
        caption = "Predicted Class: " + classes[label] + f" Probability: {prob:.2f}%"
        plt.figure()
        plt.imshow(X[i])
        plt.title(caption)
        plt.show()

# Main deployment function loads requested model and calls predata() and predict() functions
def deploy(arch, mod, imdir):
    if arch == "resnet":
        if mod == "synth":
            model = load_model("models/resnet_al_synth/")
        if mod == "all":
            model = load_model("models/resnet_al_all/")
    if arch == "cnn": 
        if mod == "synth":
            model = load_model("models/cnn_al_synth/")
        if mod == "all":
            model = load_model("models/cnn_al_all/")
    X, files = prepdata(imdir)
    predict(X, model, files)
