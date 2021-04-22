class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 6, 3),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=2),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(30000, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 8),
            nn.ReLU()
        )

    def forward(self, x):
        # save the input as residual
        residual1 = x
        x = self.block1(x)

        # add input to the output of block 1
        x += residual1 

        x = x.view(-1, 30000)
        x = F.softmax(self.fc(x), dim=1)
        return x
        