import torch.nn as nn
import torch.nn.functional as F
import torch

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        first_channels = 16
        self.conv1 = nn.Conv2d(3, first_channels, 7)
        self.bn1 = nn.BatchNorm2d(first_channels)
        self.firstpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # the number of inner channels
        inner1 = 32
        self.block1 = nn.Sequential(
            nn.Conv2d(first_channels, inner1, 3),
            nn.BatchNorm2d(inner1),
            nn.ReLU(),
            nn.Conv2d(inner1, first_channels, 3, padding=2),
            nn.BatchNorm2d(first_channels)
        )

        inner2 = 64
        self.block2 = nn.Sequential(
            nn.Conv2d(first_channels, inner2, 3),
            nn.BatchNorm2d(inner2),
            nn.ReLU(),
            nn.Conv2d(inner2, first_channels, 3, padding=2),
            nn.BatchNorm2d(first_channels)
        )

        inner3 = 128
        self.block3 = nn.Sequential(
            nn.Conv2d(first_channels, inner3, 3),
            nn.BatchNorm2d(inner3),
            nn.ReLU(),
            nn.Conv2d(inner3, first_channels, 3, padding=2),
            nn.BatchNorm2d(first_channels)
        )

        inner4 = 256
        self.block4 = nn.Sequential(
            nn.Conv2d(first_channels, inner4, 3),
            nn.BatchNorm2d(inner4),
            nn.ReLU(),
            nn.Conv2d(inner4, first_channels, 3, padding=2),
            nn.BatchNorm2d(first_channels)
        )

        inner4 = 512
        self.block4 = nn.Sequential(
            nn.Conv2d(first_channels, inner4, 3),
            nn.BatchNorm2d(inner4),
            nn.ReLU(),
            nn.Conv2d(inner4, first_channels, 3, padding=2),
            nn.BatchNorm2d(first_channels)
        )

        self.fc = nn.Sequential(
            nn.Linear(141376, 2048),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(128, 8)
        )



    def forward(self, x):

        # apply a large convolutional filter over the image first
        x = F.relu(self.bn1(self.conv1(x)))

        # save the input as residual
        residual1 = x

        x = self.block1(x)
        x += residual1  # add input to the output of block 1
        x = F.relu(x)

        residual2 = x  #store output of first block

        x = self.block2(x)  # run through second block  
        x += residual2    # add input to block2 to the output of block 2
        x = F.relu(x)

        residual3 = x

        x = self.block3(x)
        x += residual3
        x = F.relu(x)

        residual4 = x

        x = self.block4(x)
        x += residual4
        x = F.relu(x)

        x = x.view(-1, 141376)
        x = self.fc(x)  # pytorch crossentropy loss doesn't need softmax output
        return x
        