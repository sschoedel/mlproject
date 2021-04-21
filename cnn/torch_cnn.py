import torch
import torch.nn as nn
import torch.nn.functional as F

# use https://madebyollin.github.io/convnet-calculator/ to calculate size of cnn layers

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # output: 6x96x96
        self.pool = nn.MaxPool2d(2, 2)  # output: 6x48x48
        self.conv2 = nn.Conv2d(6, 16, 5) # output: 16x44x44
        # note that there is another maxpooling layer here with output 16x22x22
        self.fc1 = nn.Linear(7744, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # first argument should be batch size, and -1 will let pytorch figure out other dimensions
        x = x.view(4, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
