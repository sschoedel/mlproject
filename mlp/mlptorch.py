import os
import numpy as np
import torch
import time
import torch.optim as optim
import torch.nn as nn
from torch_mlp import MLP
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
  
EPOCHS=20
MIN_EPOCHS=10
MAX_EPOCHS=50
EPOCH_CHANGE=5

ALL_EPOCHS=np.arange(MIN_EPOCHS, MAX_EPOCHS+EPOCH_CHANGE, EPOCH_CHANGE)
# print(f"All epochs to test: {ALL_EPOCHS}.")

# Specify what transformations to apply to each image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(100)
    # transforms.Grayscale(num_output_channels=1)
])


def train_mlp():
  # Set fixed random number seed
  torch.manual_seed(42)
  
  start = time.time()
  
  print("Importing training dataset")
  trainset = datasets.ImageFolder(os.path.join(os.getcwd(), '..', 'Train'), transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1)
  print("Importing testing dataset")
  testset = datasets.ImageFolder(os.path.join(os.getcwd(), '..', 'Test'), transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True, num_workers=1)
  print("DataLoader finished")
  print(f"Importing datasets took {time.time() - start} seconds.")
  
  # Initialize the MLP
  mlp = MLP()
  
  # Define the loss function and optimizer
  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

  prev_test_correct = 0
  # print(f"Training and testing with {epochs} epochs.")
  with tqdm(total=EPOCHS*len(trainloader), desc="Training...") as progress_bar:
    # Run training loop
    for epoch in range(EPOCHS):
      current_loss = 0.0
      
      # Iterate over the DataLoader for training data
      for i, data in enumerate(trainloader, 0):
        inputs, targets = data
        optimizer.zero_grad()
        outputs = mlp(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
            
        progress_bar.update(1)
          
    progress_bar.close()

    # Process is complete.
    print('Training process has finished.')
  
  # if test_correct > prev_test_correct:
  # prev_test_correct = test_correct
  # print(f"New best model, saving...")
  # torch.save(mlp.state_dict(), "trained_mlp.pt")
  # print("Model saved to 'trained_mlp.pt'.")
  
  # save model in the same folder as this script
  model_dir = os.path.join(os.getcwd(), 'models')
  model_path = os.path.join(model_dir, 'trained_mlp')
  torch.save(mlp, model_path)
  print(f'Saving MLP model as {model_path}')

# Accepts 100x100x3 RGB image
def test_mlp(data, label):
  
  mlp = torch.load(os.path.join(os.getcwd(), 'trained_mlp'))
  mlp.eval()

  print("Importing training dataset")
  trainset = datasets.ImageFolder(os.path.join(os.getcwd(), '..', 'Train'), transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1)
  print("Importing testing dataset")
  testset = datasets.ImageFolder(os.path.join(os.getcwd(), '..', 'Test'), transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True, num_workers=1)
  print("DataLoader finished")
  print(f"Importing datasets took {time.time() - start} seconds.")
  
  # Evaluate model on train dataset
  correct = 0
  total = 0
  with torch.no_grad():
      for data in trainloader:
          images, labels = data
          outputs = mlp(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  
  print(f'Accuracy of the network on train images: {100*correct/total}')
  
  # Evaluate model on test dataset
  correct = 0
  total = 0
  with torch.no_grad():
      for data in testloader:
          images, labels = data
          outputs = mlp(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          
  test_correct = correct/total
  
  print(f'Accuracy of the network on test images: {100*test_correct}')
  