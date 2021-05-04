# reCAPTCHA Image Classifier

## Description

This repo contains all the code, data, and models necessary to train and test classifiers on the reCAPTCHA dataset. There are three classification algorithms used - Naive Bayes, Multi-Layer Perceptron (MLP), and Convolutional Neural Network (CNN). 


## Installation

This project was validated using Python version 3.9.4 and a Windows operating system. All of the dependencies necessary for this project are listed in the file `requirements.txt`. To install these dependencies, run the command

```
pip install -r requirements.txt
```

You can choose to install these in an isolated virtual environment as well. 

We did not test this installation in a conda environment, nor did we validate completely using a different Python version. There are no guarantees that the code will run properly in such environments. 

## Usage

All three classifiers can be trained and run through the driver script `main.py`.

```
python main.py [options]
```

The options are listed below, and can be found by running `python main.py -h`.

```
python main.py -h
    -h  Help
    -a  Run all models (default)
    -n  Run only Naive Bayes
    -c  Run only CNN
    -m  Run only MLP
    -C  Run with train CNN (takes a few hours)
    -M  Run with train MLP (takes a few hours)
```

### Testing

By default, all models will be tested. If you would like to test individual models, use the flags `-n`, `-c`, and `-m`, to test the Naive Bayes, CNN, or MLP only, respectively.

There are CNN and MLP models that have already been trained on the dataset, in `cnn/trained_cnn.pth` and `mlp/final_trained_mlp.pt`. By default, these models will be used when testing. When training a new model using `-C` and `-M`, the model that was just trained will subsequently be tested. 

### Training

To train new models, then use the flags `-C` to train a new CNN and `-M` to train a new MLP. Once these models are trained, they will automatically be tested. Their weights will be stored within the `cnn\models` and the `mlp` directory, respectively. 

There is no specific flag to train and test Naive Bayes separately, since it is an algorithm that runs very quickly. For the same reason, there is no pretrained Naive Bayes model saved in this repository. 

## Dataset

The dataset used for this project was pulled from the following Git repository:
```
https://github.com/deathlyface/recaptcha-dataset
```

This dataset contains thousands of images for 11 different reCAPTCHA classes. The classes `Chimney`, `Motorcycle`, and `Mountain` all contained fewer than 100 images, so we elected to remove them from the dataset. The 8 remaining classes were split into a training and testing dataset were split with the ratio 70/30, and stored in this repository within the directories `Train` and `Test`.

## Contributing

### Paolo Fermin

Paolo was primarily responsible for the design and training of the Convolutional Neural Network. This includes presenting his model during the class presentation, and completing that portion of the final report. In addition, Paolo wrote the README documentation in the repository. 

### Nathan Moeliono

Nathan was primarily responsible for the design and training of the Naive Bayes algorithm. This includes presenting his model during the class presentation, and completing that portion of the final report. In addition, Nathan designed and structured the final presentation. 
