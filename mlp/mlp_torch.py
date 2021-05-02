import os
import numpy as np
import torch
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from tqdm import tqdm
import csv


# MLPs to test

class MLP(nn.Module):
  def __init__(self, layers_):
	super().__init__()
	self.layers = layers_

  def forward(self, x):
	return self.layers(x)


class MLP_Torch:
	
	def __init__(self):
		pass


	self.layers = nn.Sequential(
		nn.Flatten(),
		nn.Linear(30000, 120),
		nn.ReLU(),
		nn.Linear(120, 84),
		nn.ReLU(),
		nn.Linear(84, 8)
	)
 
	def find_optimal_params(self, structures=None):
		"""
		Train model on every combination of given structures and lambda values
		to find the best combination of model architecture and lambda value hyper-parameter.
		:param structure: MLP architectures to train with
  		"""
    
		# Set default structures to test if not given
		if structures == None:
			structures = [[100,10], [200,50], [7200,7200], [500,100,50], [5000,1000,500]]
   

    
	
	def train_mlp():
		EPOCHS=40
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

		headers = ['epochs', 'test accuracy', 'train accuracy']
		results_test = []
		results_train = []
		epochs = []
		
		prev_test_correct = 0
		# print(f"Training and testing with {epochs} epochs.")
		with tqdm(total=EPOCHS*len(trainloader), desc="Training...") as progress_bar:
			# Run training loop
			for epoch in range(1,EPOCHS+1):
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
					
				results_test.append(test_mlp(testloader, mlp))
				results_train.append(test_mlp(trainloader, mlp))
				epochs.append(epoch)
			
			progress_bar.close()

			# Process is complete.
			print('Training process has finished.')
		accuracies = np.array([epochs, results_test, results_train])
		accuracies = np.transpose(accuracies)
		with open('epoch_results.csv', 'w') as f:
			write = csv.writer(f)
			write.writerow(headers)
			write.writerows(accuracies)
		
		# if test_correct > prev_test_correct:
		# prev_test_correct = test_correct
		# print(f"New best model, saving...")
		# torch.save(mlp.state_dict(), "trained_mlp.pt")
		# print("Model saved to 'trained_mlp.pt'.")
		
		# save model in the same folder as this script
		# model_dir = os.path.join(os.getcwd(), 'models')
		# model_path = os.path.join(model_dir, 'trained_mlp.pt')
		# torch.save(mlp.state_dict(), model_path)
		# print(f'Saving MLP model as {model_path}')


	def test_mlp():
		
		mlp = MLP()
		mlp.load_state_dict(torch.load('trained_mlp.pt'))
		mlp.eval()

		# Specify what transformations to apply to each image
		transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.CenterCrop(100)
				# transforms.Grayscale(num_output_channels=1)
		])
		
		print("Importing training dataset")
		trainset = datasets.ImageFolder(os.path.join(os.getcwd(), '..', 'Train'), transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1)
		print("Importing testing dataset")
		testset = datasets.ImageFolder(os.path.join(os.getcwd(), '..', 'Test'), transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True, num_workers=1)
		print("DataLoader finished")

		# # Evaluate model on train dataset
		# correct = 0
		# total = 0
		# with torch.no_grad():
		#     for data in trainloader:
		#         images, labels = data
		#         outputs = mlp(images)
		#         _, predicted = torch.max(outputs.data, 1)
		#         total += labels.size(0)
		#         correct += (predicted == labels).sum().item()
		
		# # print(f'Accuracy of the network on train images: {100*correct/total}')
		
		# Evaluate model on test dataset
		
		test_images_truth = np.array(0)
		predictions = np.array(0)
		
		correct = 0
		total = 0
		with torch.no_grad():
				for data in testloader:
						images, labels = data
						outputs = mlp(images)
						_, predicted = torch.max(outputs.data, 1)
						predictions = np.append(predictions, predicted)
						test_images_truth = np.append(test_images_truth, labels)
						total += labels.size(0)
						correct += (predicted == labels).sum().item()
						
		test_correct = correct/total
		
		test_images_truth.ravel()
		predictions.ravel()
		
		# print(f'Accuracy of the network on test images: {100*test_correct}')
		
		print(classification_report(test_images_truth, predictions, digits=3))
		
		return test_correct