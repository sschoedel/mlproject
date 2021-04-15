from torch_cnn import Net
import torch
from torchvision import datasets
import os

net = Net()
print(net)

trainset = datasets.ImageFolder(os.path.join('..', 'Train'))
print(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()