from torch_cnn import Net
import torch
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformations import Rescale, RandomCrop

batch_size = 4

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),
    transforms.CenterCrop(100)
    #Rescale(120) 
    #RandomCrop(100)
])

net = Net()
print(net)

trainset = datasets.ImageFolder(os.path.join('..', 'Train'), transform=transform)
print(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

def imshow(batch):
    if batch.ndim == 4:
        for img in batch:         
            img = img / 2 + 0.5     # unnormalize
            npimg = img.numpy()
            plt.imshow(npimg)
            plt.show()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')