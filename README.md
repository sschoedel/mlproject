# reCAPTCHA Image Classifier

## Description

This repo contains all the code, data, and models necessary to train and test classifiers on the reCAPTCHA dataset. There are three classification algorithms used - Naive Bayes, Multi-Layer Perceptron (MLP), and Convolutional Neural Network (CNN). 


## Installation

This project was validated using Python version 3.9.4 and a Windows operating system. All of the dependencies necessary for this project are listed in the file `requirements.txt`. To install these dependencies, run the command

```
pip install -r requirements.txt
```

In an isolated virtual environment.

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
    -G  Run CNN on GPU (use only with NVIDIA GPU with high memory)
```

### Testing

By default, all models will be tested. If you would like to test individual models, use the flags `-n`, `-c`, and `-m`, to test the Naive Bayes, CNN, or MLP only, respectively.

There are CNN and MLP models that have already been trained on the dataset, in `cnn/trained_cnn.pth` and `mlp/final_trained_mlp.pt`. By default, these models will be used when testing. When training a new model using `-C` and `-M`, the model that was just trained will subsequently be tested. 

### Training

To train new models, then use the flags `-C` to train a new CNN and `-M` to train a new MLP. Once these models are trained, they will automatically be tested. Their weights will be stored within the `cnn\models` and the `mlp` directory, respectively. 

There is no specific flag to train and test Naive Bayes separately, since it is an algorithm that runs very quickly. For the same reason, there is no pretrained Naive Bayes model saved in this repository. 

### A Note on GPUs

It is commonly known that CNNs run much faster on GPU resources, due to the spatial nature of the layers. However, the CNN used in this project was trained in a Google Colaboratory notebook, which leverages powerful GPU compute resources hosted by Google. These resources are not typically available on consumer GPUs. Thus, by default, the CNN will be trained and tested on the CPU. 

If you choose to use the `-G` flag and train on GPUs, the project team members occasionally ran into the problem of running out of GPU memory. This is because simply loading the trained CNN architecture consumes just over 1GB of GPU memory, and the machine may have other processes consuming GPU memory. If these issues occur when running the models, simply remove the `-G` flag and train on the CPU instead. 

## Dataset

The dataset used for this project was pulled from the following Git repository:
```
https://github.com/deathlyface/recaptcha-dataset
```

This dataset contains thousands of images for 11 different reCAPTCHA classes. The classes `Chimney`, `Motorcycle`, and `Mountain` all contained fewer than 100 images, so we elected to remove them from the dataset. The 8 remaining classes were split into 70% training images and 30% testing images and stored in this repository within the directories `Train` and `Test`.

## Contributing

### Nathan Moeliono

Nathan designed and trained the Naive Bayes algorithm as well as presented it during the presentation and in the final report. Nathan also designed and structured the final presentation. 

### Paolo Fermin

Paolo was primarily responsible for the design and training of the Convolutional Neural Network. Most of this was completed in a Google Colaboratory environment. In addition, Paolo presented his model during the class presentation, and completing that portion of the final report. Finally, Paolo wrote the README documentation in the repository. 

### Sam Schoedel

Sam was responsible for designing and training the multi-layer perceptron as well as laying out and editing the report. Responsibilities included presenting MLP results in the class presentation and writing the MLP portion of the report as well as summarizing the results in the conclusion.
