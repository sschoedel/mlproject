import torch
import torch.nn as nn
import torch.nn.functional as F

# use https://madebyollin.github.io/convnet-calculator/ to calculate size of cnn layers

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # note that this pooling layer follows every conv layer
        self.pool = nn.MaxPool2d(2, 2)  # output: 6x48x48

        self.conv1 = nn.Conv2d(3, 6, 3) # output: 6x96x96
        self.bn1 = nn.BatchNorm2d(6)    # normalize
        self.conv2 = nn.Conv2d(6, 16, 5) # output: 16x44x44
        self.conv3 = nn.Conv2d(16, 32, 5) 
        # note that there is another maxpooling layer 
        self.fc1 = nn.Linear(2592, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # first argument should be batch size, and -1 will let pytorch figure out other dimensions
        x = x.view(-1, 2592)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
