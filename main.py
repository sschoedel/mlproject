import numpy as np
import os
from naive_bayes import train_naive_bayes, test_naive_bayes
from cnn.tester import test_cnn
from cnn.trainer import train_cnn
from mlp.mlp_torch import Test_MLP
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
    train_mlp_flag = False
    train_nb_flag = False

    # read in command args
    try:
        opts, args = getopt.getopt(argv, "hcm")
    except getopt.GetoptError:
      print("main.py -h for help")
      sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("-h\tHelp")
            print("-c\tTrain CNN (takes a few hours)")
            print("-m\tTrain MLP (takes a few hours)")
            sys.exit()
        elif opt == "-c":
            train_cnn_flag = True
        elif opt == "-m":
            train_mlp_flag = True
        elif opt == "-n":
            train_nb_flag = True

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

    if train_cnn_flag:
        # train and test a new model
        print("-- Begin training CNN --")
        cnn_path = train_cnn()

        print("-- Begin testing CNN--")
        labels, predictions = test_cnn(model_path=cnn_path)
    else:
        # if training was not run, use the default model
        labels, predictions = test_cnn()    
    
    # Output results of CNN
    print("CNN Report:")
    print(classification_report(labels, predictions, digits=3))


    mlp = Test_MLP()
    mlp.load_data()

    if train_mlp_flag:
        print("-- Begin training MLP --")
        best_mlp = mlp.find_optimal_model()
        print("-- Begin testing MLP --")
        mlp.print_classification_report(mlp=best_mlp)
    else:
        print("-- Begin testing with pre-trained MLP model --")
        mlp.print_classification_report(model_path="mlp/final_trained_mlp.pt")

if __name__ == "__main__":
    main(sys.argv[1:])
