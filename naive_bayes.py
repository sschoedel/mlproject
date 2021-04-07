import cv2 as cv
import numpy as np

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
    
arr = np.zeros((10000, 4))
img = cv.imread('./Test/Bicycle Test/Bicycle Test Test 0.png', 0)
flattened = img.flatten()
flattened = (flattened / 255 * 4).astype(int)
encoded =  np.zeros((flattened.size, flattened.max() +1))
encoded[np.arange(flattened.size), flattened] = 1
arr += encoded

