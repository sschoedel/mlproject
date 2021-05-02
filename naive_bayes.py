import cv2 as cv
import numpy as np
import os
import time

# take first 10 images to greyscale and get average value 0-255 for each class
# divide into 4 discrete clasees, 0-64, 65-128, etc..
# create probabiity table
# conditional = probability of pixel_1 being in class X with value low (0/4)
# prior = probability of that class existing (same for everything)
# for every pixel, get probability it is of that class


# 14,400 x 4 x c array
# for each class:
    # get the images from the testing folder
    # intizliae 14,400 x 4 length array to hold values
    # for each image:
        # Turn greyscal
        # flatten
        # traverse pixels:
            # for each pixexl, get discrete value and 
            # increment value in array by 1
    # divide by number of images
    # add to humongous array

# initalize results
# true_positives
# true_negatives
# false_positives
# false_negatives

# (isTrue, image, class, model)

# array of length num_classes, init to 1,1,1...
# grayscale, flatten image
# for each class:    
    # for each pixel:
        # look at conditional array and multiply by 
        # [class][pixel][0-64]
# take the largest of this thing

    # get the true images (0-250)
    # for each other class:
        # get the false images (250-255)
    # --- logic here needs to be joined together


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


# ---------- testing below ----------

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
    

