from torch_cnn import Net
from torchvision import datasets

net = Net()
print(net)

train_images = datasets.ImageFolder()