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

# set constants
WORKERS = 0
EPOCHS = 10
BATCH_SIZE = 4

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
    #Rescale(120) 
    #RandomCrop(100)
])

model_dir = os.path.join(os.getcwd(), 'cnn')


def test_cnn():
    # test the final model
    testset = datasets.ImageFolder(os.path.join(os.getcwd(), '', 'Test'), transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=WORKERS)
    final_model = ResNet()
    final_model.load_state_dict(torch.load(os.path.join(model_dir, 'trained_cnn.zip')))
    final_model.eval()
    
    predictions = np.array([])
    with torch.no_grad():
        for data in testloader:
            #images, labels = data[0].to(device), data[1].to(device)
            images, labels = data
            outputs = final_model(images)
            _, predicted = torch.max(outputs.data, 1)
            np.concatenate(predictions, predicted)
    return predictions

