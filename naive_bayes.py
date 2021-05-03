import cv2 as cv
import numpy as np
import os
import time

IMG_SZ = 120
NUM_GROUPS = 4
LAPLACE = 1

TRAIN_PATH = './Train/'
TEST_PATH = './Test/'

def train_naive_bayes(classes, num):  

    NUM_GROUPS = num

    # initialize model
    model = {}
    # initialize conditional probability matrix and priors matrix
    conditionals = []
    priors = []
    total_files = 0

    # iterate over every class
    for dir_name in classes:    

        arr = np.zeros((IMG_SZ * IMG_SZ, NUM_GROUPS))
        num_files = 0

        # iterate over every image in this class
        for file_name in os.listdir(TRAIN_PATH + dir_name):   
            img = cv.imread(TRAIN_PATH + dir_name + "/" + file_name, 0)

            # skip over the image if it is not if size 120x120 (rare)
            if img.shape[0] != IMG_SZ:
                continue

            flattened = img.flatten()
            flattened = (flattened / 255 * (NUM_GROUPS - 1)).astype(int)
            arr[np.arange(flattened.size), flattened] += 1
            num_files += 1

        # apply Laplace smoothing
        arr = (arr + LAPLACE) / (num_files + (LAPLACE * len(classes)))
        conditionals.append(arr)
        priors.append(num_files)
        total_files += num_files

    # save results to returned model
    conditionals = np.array(conditionals)
    priors = np.array(priors)
    priors = priors / total_files
    model['conditionals'] = conditionals
    model['priors'] = priors

    return model

def test_naive_bayes(model, classes, img_path, dir_i, num): 

    NUM_GROUPS = num
    # read and flatten image
    img = cv.imread(img_path, 0)     
    flattened = img.flatten()
    flattened = (flattened / 255 * (NUM_GROUPS - 1)).astype(int)

    max_num = -999999999
    max_class = 0
    # calculate the predicted class
    for i in range(len(classes)):
        num = np.sum(np.log(model['conditionals'][i][np.arange(flattened.size), flattened]))
        num += np.log(model['priors'][dir_i])
        if num > max_num:
            max_class = i
            max_num = num
    return max_class
    

