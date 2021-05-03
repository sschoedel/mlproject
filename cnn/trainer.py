from cnn.res_cnn import ResNet
import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_cnn():
    # set constants
    WORKERS = 0
    EPOCHS = 30
    BATCH_SIZE = 4
    PRINT_INTERVAL = 100

    # train on GPU, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    net = ResNet()
    #net = Net()
    print(net)
    # uncomment if training on a gpu
    #net.to(device)

    # specify what transformations to apply on each image
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor(),
        transforms.CenterCrop(100)
    ])

    # turn image datasets into pytorch dataloaders
    # separate training and testing datasets
    trainset = datasets.ImageFolder(os.path.join(os.getcwd(), 'Train'), transform=transform)
    print(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=WORKERS, pin_memory=True)

    testset = datasets.ImageFolder(os.path.join(os.getcwd(), 'Test'), transform=transform)
    print(testset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=WORKERS, pin_memory=True)

    # store dataloaders in dict for readability
    dataloaders = {
        'train': trainloader,
        'val': testloader
    }

    # define optimizer and loss functions
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    model_dir = os.path.join(os.getcwd(), 'cnn', 'models')
    i = 0
    while os.path.exists(os.path.join(model_dir, str(i))):
        i += 1
    model_dir = os.path.join(model_dir, str(i))
    model_path = os.path.join(model_dir, 'trained_cnn')

    # create a tensorboard log
    writer = SummaryWriter(model_dir)

    print(f'training model number {i}')

    # start the best loss as a very large number
    best_loss = 9999
    tolerance = 0.1

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        for phase in ['train', 'val']:
            running_loss = 0.0
            if phase == 'train': 
                # set model to training mode
                net.train(True)
            else:
                net.train(False)

            for i, data in enumerate(dataloaders[phase], 0):
                # get the inputs; data is a list of [inputs, labels]
                # send the inputs to the gpu
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # only update the weights in training, not validation
                if phase == 'train':         
                    loss.backward()
                    optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % PRINT_INTERVAL == PRINT_INTERVAL-1:    # print every 1000 mini-batches
                    # ...log the running loss
                    writer.add_scalar(f'{phase} loss',
                                    running_loss / PRINT_INTERVAL,
                                    epoch * len(dataloaders[phase]) + i)

                    print(f'[{epoch+1}, {i+1}] {phase} loss: {running_loss/PRINT_INTERVAL:0.3f}')

                    # save new best model if validation loss decreases 
                    if phase == 'val' and running_loss/PRINT_INTERVAL < (best_loss - tolerance):
                        torch.save(net, os.path.join(model_dir, 'best_cnn'))
                        print(f'saving new best model with val loss {running_loss/PRINT_INTERVAL}')
                        best_loss = running_loss/PRINT_INTERVAL

                    running_loss = 0.0

    print('Finished Training')

    # save final model in the same folder as this script
    model_path = os.path.join(model_dir, 'trained_cnn')
    torch.save(net, model_path)
    print(f'Saving model as {model_path}')

    return model_path
