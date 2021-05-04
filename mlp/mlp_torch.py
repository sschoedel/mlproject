import os
import numpy as np
import torch
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from tqdm import tqdm
import csv


class MLP(nn.Module):
	def __init__(self, layers_=nn.Sequential(
					nn.Flatten(),
					nn.Linear(30000, 120),
					nn.ReLU(),
					nn.Linear(120, 84),
					nn.ReLU(),
					nn.Linear(84, 8) )):
		super().__init__()
		self.layers = layers_

	def forward(self, x):
		return self.layers(x)


class Test_MLP:
	final_params = 0
 
	trainset = 0
	trainloader = 0
	testset = 0
	testloader = 0
 
	def cross_validate(self, mlp, all_data, all_labels, folds, params):
		"""
		Performs cross validation with random splits
		
		:param all_data: 30,000 x n data matrix
		:type all_data: numpy ndarray
		:param all_labels: n x 1 label vector
		:type all_labels: numpy array
		:param folds: number of folds to run of validation
		:type folds: int
		:param params: auxiliary variables for training algorithm (e.g., regularization parameters)
		:type params: dict
		:return: tuple containing the average score and the learned models from each fold
		:rtype: tuple
		"""

		scores = np.zeros(folds)

		d, n = all_data.shape

		indices = np.array(range(n), dtype=int)

		# pad indices to make it divide evenly by folds
		examples_per_fold = int(np.ceil(n / folds))
		ideal_length = int(examples_per_fold * folds)
		# use -1 as an indicator of an invalid index
		indices = np.append(indices, -np.ones(ideal_length - indices.size, dtype=int))
		assert indices.size == ideal_length

		indices = indices.reshape((examples_per_fold, folds))

		models = []

		for i in range(folds):
			train_indices = np.delete(indices, i, 1).ravel()
			# remove invalid padded indices
			train_indices = train_indices[train_indices >= 0]
			val_indices = indices[:, i].ravel()
			# remove invalid padded indices
			val_indices = val_indices[val_indices >= 0]

			val_data = all_data[:, val_indices]
			val_labels = all_labels[val_indices]

			train_data = all_data[:, train_indices]
			train_labels = all_labels[train_indices]
   
			# Create mlp to train
			mlp = MLP(params['num_hidden_units'])
   
			cross_val_message = "Training cross validation fold " + str(i) + "."
			test_score, best_epoch = self.train(mlp, trainset, trainloader, epochs=params['epochs'], message=cross_val_message)

			# Test on all data
			predictions = test_mlp(testset, testloader, mlp=mlp, model_name='trained_mlp.pt')
			if isinstance(predictions, tuple):
				predictions = predictions[0]

			scores[i] = np.mean(predictions == val_labels)

		score = np.mean(scores)
	
		print(f"Average crossval score with structure {params['num_hidden_units']}: {score}")

		return score

	def find_optimal_model(self, structures=None, learning_rates=None, save_path='mlp/trained_mlp.pt'):
		"""
		Train model on every combination of given structures and lambda values
		to find the best combination of model architecture and lambda value hyper-parameter.
		:param structures: MLP architectures to train with
		:type structures: 
  		"""
	
		# Set default structures to test if not given
		if structures == None:
			structures_nodes = [[120,84], [200,28], [150,42], [80, 80]]
			num_structures = 4
			structures = [nn.Sequential(
					nn.Flatten(),
					nn.Linear(30000, structures_nodes[s][0]),
					nn.ReLU(),
					nn.Linear(structures_nodes[s][0], structures_nodes[s][1]),
					nn.ReLU(),
					nn.Linear(structures_nodes[s][1], 8) )
					for s in range(num_structures)]
   
		# Set default learning rates to test if not given
		if learning_rates == None:
			learning_rates = [.1, .01, .001]
   
		# Initialize best score to 0
		best_score = 0

		# Set cross validation parameters
		num_folds = 4
  
		# Initialize parameters
		params = {'epochs': 25}
		best_params = 0
		
		# Train on full dataset for each model architecture
		start_all = time.time()
		for i in range(len(structures)):
			for j in range(len(learning_rates)):
				structure = structures[i]

				params['num_hidden_layers'] = structures[i]
				params['lambda_value'] = learning_rates[j]
				
				# Cross validate with given structure and lambda values
				start_single = time.time()
				# cv_score = cross_validate(self.train_data, self.train_labels, num_folds, params)
	
				mlp = MLP(structure)
				train_msg = "Training with structure " + str(i+1) + " of " + str(len(structures)) + " and learning rate " + str(params['lambda_value']) + " (" + str(j+1) + " of " + str(len(learning_rates)) + ")."
				score, epoch_used = self.train(mlp, learning_rate=params['lambda_value'], EPOCHS=params['epochs'], message=train_msg)
    
				# Update params if best score
				if score > best_score:
					best_score = score
					best_params = copy.copy(params)
					best_params['epochs'] = epoch_used
		
		print(f"Total time: {time.time() - start_all} seconds.")
		print(f"Best model with structure {best_params['num_hidden_layers']} and learning rate {best_params['lambda_value']}.")
  
		# Train again on best structure and epochs and save model
		mlp_best = MLP(best_params['num_hidden_layers'])
		train_msg = "Training with best parameters..."
		_, _ = self.train(mlp_best, learning_rate=best_params['lambda_value'], EPOCHS=best_params['epochs'], message=train_msg)
		torch.save(mlp.state_dict(), save_path)
		print(f"Model trained with best parameters saved to {save_path}")

		self.final_params = best_params
		return mlp_best

	
	# Import all data
	def load_data(self):	

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
		self.trainset = datasets.ImageFolder(os.path.join(os.getcwd(), 'Train'), transform=transform)
		self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=10, shuffle=True, num_workers=1)
		print("Importing testing dataset")
		self.testset = datasets.ImageFolder(os.path.join(os.getcwd(), 'Test'), transform=transform)
		self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=10, shuffle=True, num_workers=1)
		print(f"Importing datasets took {time.time() - start} seconds.")
  
 
	def write_epochs_train_mlp(self, trainset, testset):
		EPOCHS=40
		MIN_EPOCHS=10
		MAX_EPOCHS=50
		EPOCH_CHANGE=5

		ALL_EPOCHS=np.arange(MIN_EPOCHS, MAX_EPOCHS+EPOCH_CHANGE, EPOCH_CHANGE)
		
		# Initialize the MLP
		mlp = MLP()

		headers = ['epochs', 'test accuracy', 'train accuracy']
		results_test = []
		results_train = []
		epochs = []
		
		prev_test_correct = 0
		# print(f"Training and testing with {epochs} epochs.")
		with tqdm(total=EPOCHS*len(trainloader), desc="Training...") as progress_bar:
			# Run training loop
			for epoch in range(1,EPOCHS+1):
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
   
	def train(self, mlp, learning_rate=1e-4, EPOCHS=30, message="Training..."):
		
		# Define the loss function and optimizer
		loss_function = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
  
		results_test = []
		results_train = []
		epochs = []
  
		# Initialize best results
		best_test = 0
		best_epoch = 1
		best_loss = 9999
		tolerance = 0.1
		PRINT_INTERVAL=1000
		with tqdm(total=EPOCHS*len(self.trainloader), desc=message) as progress_bar:
			# Run training loop
			for epoch in range(EPOCHS):
				# Iterate over the DataLoader for training data
				for i, data in enumerate(self.trainloader, 0):
					inputs, targets = data
					optimizer.zero_grad()
					outputs = mlp(inputs) # Forward pass
					loss = loss_function(outputs, targets) # Compute loss
					loss.backward() # Backpropagate errors and update weights
					optimizer.step()
							
					progress_bar.update(1)
					
				# # Track results for each epoch
				# test_result = self.test_mlp(self.testloader, mlp)
				# # train_result = self.test_mlp(self.trainloader, mlp)
				# if test_result > best_test:
				# 	best_test = test_result
				# 	best_epoch = epoch

			progress_bar.close()

		return best_test, best_epoch


	def test_mlp(self, dataloader, mlp, model_name=None):
		# # Load model from file if none are provided
		# if mlp == None:
		# 	mlp = MLP()
		# 	mlp.load_state_dict(torch.load(model_name), strict=False)
		# 	mlp.eval()
		
		# Evaluate model on test dataset
		correct = 0
		total = 0
		with torch.no_grad():
			for data in dataloader:
				images, labels = data
				outputs = mlp(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
						
		test_correct = correct/total
		
		return test_correct

	def print_classification_report(self, mlp=None, model_path=None):
		
		if mlp == None:
			mlp = MLP()
			mlp.load_state_dict(torch.load(model_path), strict=False)
			mlp.eval()

		# Specify what transformations to apply to each image
		transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.CenterCrop(100)
		])
		
		print("Importing training dataset")
		trainset = datasets.ImageFolder(os.path.join(os.getcwd(), 'Train'), transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1)
		print("Importing testing dataset")
		testset = datasets.ImageFolder(os.path.join(os.getcwd(), 'Test'), transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True, num_workers=1)
		print("DataLoader finished")
  
		test_images_truth = np.array(0)
		predictions = np.array(0)
		
		correct = 0
		total = 0
		with torch.no_grad():
			for data in tqdm(testloader, desc="Testing..."):
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
  
		print("MLP Report:")
		print(classification_report(test_images_truth, predictions, digits=3))