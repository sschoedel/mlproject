# reCAPTCHA Image Classifier

## Overview

This repo contains all the code, data, and models necessary to train and test classifiers on the reCAPTCHA dataset. There are three classification algorithms used - Naive Bayes, Multi-Layer Perceptron (MLP), and Convolutional Neural Network (CNN). All three classifiers can be trained and run through the driver script `main.py`.

```
python main.py 

```


## Installing requirements

This project was validated using Python version 3.9.4 and a Windows operating system. All of the dependencies necessary for this project are listed in the file `requirements.txt`. To install these dependencies, run the command

```
pip install -r requirements.txt
```

You can choose to install these in an isolated virtual environment as well. 

We did not test this installation in a conda environment, nor did we validate completely using a different Python version. There are no guarantees that the code will run properly in such environments. 

## Dataset

The dataset used for this project was pulled from the following Git repository:
```
https://github.com/deathlyface/recaptcha-dataset
```

This dataset contains thousands of images for 11 different reCAPTCHA classes. The classes `Chimney`, `Motorcycle`, and `Mountain` all contained fewer than 100 images, so we elected to remove them from the dataset. The 8 remaining classes were split into a training and testing dataset were split with the ratio 70/30, and stored in this repository within the directories `Train` and `Test`.