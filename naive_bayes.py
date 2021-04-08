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

dir_names = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney', 'Crosswalk', 'Hydrant', 'Motorcycle', 'Mountain', 'Palm', 'Traffic Light']
train_path = './Train/'
test_path = './Test/'

res_arr = []

for dir_name in dir_names:    
    arr = np.zeros((IMG_SZ * IMG_SZ, NUM_GROUPS))
    # max_files = 50 # remove later
    num_files = 0
    for file_name in os.listdir(train_path + dir_name):   
        img = cv.imread(train_path + dir_name + "/" + file_name, 0)
        if img.shape[0] != IMG_SZ:
            continue
        flattened = img.flatten()
        flattened = (flattened / 255 * (NUM_GROUPS - 1)).astype(int)
        arr[np.arange(flattened.size), flattened] += 1
        num_files += 1
        # if num_files == max_files: # remove later
          #  break
    arr = (arr + LAPLACE) / (num_files + (LAPLACE * len(dir_names)))
    res_arr.append(arr)

res_arr = np.array(res_arr)


# ---------- testing below ----------

print("TESTING STARTING ------------------")

dir_i = 0
correct = 0
incorrect = 0
for dir_name in dir_names:    
    arr = np.zeros((IMG_SZ * IMG_SZ, NUM_GROUPS))
    for file_name in os.listdir(test_path + dir_name):  
        img = cv.imread(test_path + dir_name + "/" + file_name, 0)     
        flattened = img.flatten()
        flattened = (flattened / 255 * (NUM_GROUPS - 1)).astype(int)

        
        max_num = -999999999
        max_class = 0
        for i in range(len(dir_names)):
            num = np.sum(np.log(res_arr[i][np.arange(flattened.size), flattened]))
            if num > max_num:
                max_class = i
                max_num = num
        if max_class == dir_i:
            correct += 1
        else:
            incorrect += 1

    dir_i += 1

print("RESULTS: ---------")
print(correct)
print(incorrect)
print(correct / (correct + incorrect))
    

