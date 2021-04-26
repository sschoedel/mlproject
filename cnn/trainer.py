from torch_cnn import Net
from res_cnn import ResNet
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
print(net)

# specify what transformations to apply on each image
transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),
    transforms.CenterCrop(100)
    #Rescale(120) 
    #RandomCrop(100)
])

# load the datasets
trainset = datasets.ImageFolder(os.path.join(os.getcwd(), '..', 'Train'), transform=transform)
print(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=WORKERS)

testset = datasets.ImageFolder(os.path.join(os.getcwd(), '..', 'Test'), transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=WORKERS)

# set loss and optimizer functions
# use crossentropyloss for multi-class classification
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

PRINT_INTERVAL = 1000

model_dir = os.path.join(os.getcwd(), 'models')
i = 0
while os.path.exists(os.path.join(model_dir, str(i))):
    i += 1
model_dir = os.path.join(model_dir, str(i))

# create a tensorboard log
writer = SummaryWriter(model_dir)

for epoch in range(EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # replace the line above with this line if training on a gpu:
        #inputs, labels = data.to(device)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % PRINT_INTERVAL == PRINT_INTERVAL-1:    # print every 1000 mini-batches
            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / PRINT_INTERVAL,
                            epoch * len(trainloader) + i)

            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / PRINT_INTERVAL))
            running_loss = 0.0

print('Finished Training')

# save model in the same folder as this script
model_path = os.path.join(model_dir, 'trained_cnn')
torch.save(net, model_path)
print(f'Saving model as {model_path}')

# test the model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100*correct/total}')