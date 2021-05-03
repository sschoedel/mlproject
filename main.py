import numpy as np
import os
from naive_bayes import train_naive_bayes, test_naive_bayes
from cnn.trainer import test_cnn
from sklearn.metrics import classification_report
import sys, getopt
from cnn.res_cnn import ResNet
import time

# path constants
classes = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Crosswalk', 'Hydrant', 'Palm', 'Traffic Light']
TRAIN_PATH = './Train/'
TEST_PATH = './Test/'

def main(argv):

    # option flags
    train_cnn_flag = False

    # read in command args
    try:
        opts, args = getopt.getopt(argv, "hc")
    except getopt.GetoptError:
      print("main.py -h for help")
      sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("-h\tHelp")
            print("-c\tTrain CNN (takes a couple hours)")
            sys.exit()
        elif opt == "-c":
            train_cnn_flag = True

    test_images_truth = []
    # traverse the test image directories to get the ground truth
    dir_i = 0
    for dir_name in classes:  
        for file_name in os.listdir(TEST_PATH + dir_name):
            test_images_truth.append(dir_i)
        dir_i += 1
    test_images_truth = np.array(test_images_truth)

    possible_num_groups = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    print("======= Running Naive Bayes ========")
    start_time_nb = time.time()
    print("-- Begin training NB --")
    naive_bayes_model = train_naive_bayes(classes, 128)

    run_time_nb = time.time() - start_time_nb   
    print("Naive Bayes training time: " + str(run_time_nb) + " seconds")

    predictions = []    
    start_time_nb = time.time()
    print("-- Begin testing NB --")    
    dir_i = 0
    for dir_name in classes:  
        for file_name in os.listdir(TEST_PATH + dir_name):
            img_path = TEST_PATH + dir_name + "/" + file_name
            res = test_naive_bayes(naive_bayes_model, classes, img_path, dir_i, 128)
            predictions.append(res)
        dir_i += 1
    predictions = np.array(predictions)

    # Output results of Naive Bayes
    print("Naive Bayes Report:")
    print(classification_report(test_images_truth, predictions, digits=3))
    
    run_time_nb = time.time() - start_time_nb   
    print("Naive Bayes testing time: " + str(run_time_nb) + " seconds")

    '''

    print("======= Running CNN ========")

    #print("-- begin training CNN --")
    #naive_bayes_model = train_naive_bayes(classes)

    print("-- Begin testing CNN--")
    predictions = test_cnn()

    # Output results of Naive Bayes
    print("CNN Report:")
    print(classification_report(test_images_truth, predictions, digits=3))
    '''


    

if __name__ == "__main__":
    main(sys.argv[1:])
