from __future__ import division
import h5py
import cv2
import numpy as np
import copy
import os
import time
import json
from scipy.special import softmax
from sklearn.metrics import classification_report
from tqdm import tqdm


class MLP_Scratch:
	train_data = 0
	test_data = 0
	train_labels = 0
	test_labels = 0

	cur_model = 0
	final_params = 0
 
	def __init__(self, dir_names_):
		self.dir_names = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Crosswalk', 'Hydrant', 'Palm', 'Traffic Light']
		self.NUM_CLASSES = len(dir_names_)
 

	def cross_validate(self, a, trainer, predictor, all_data, all_labels, folds, params):
		"""
		Performs cross validation with random splits
		
		:param trainer: function that trains a model from data with the template 
				model = function(all_data, all_labels, params)
		:type trainer: function
		:param predictor: function that predicts a label from a single data point
					label = function(data, model), or a tuple that contains the prediction in its 0th entry
		:type predictor: function
		:param all_data: d x n data matrix
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

			model = trainer(train_data, train_labels, params)

			predictions = predictor(val_data, model)
			if isinstance(predictions, tuple):
				predictions = predictions[0]

			models.append(model)

			scores[i] = np.mean(predictions == val_labels)

		score = np.mean(scores)
	
		print("Average crossval score with structure %s and lambda = %f: %f" % (repr(params['num_hidden_units']), params['lambda'], score))

		return score, models


	# Activation functions (logistic and relu)
	def logistic(self, x):
		"""
		Sigmoid activation function that returns the activation value and a matrix of the
					derivatives of the elementwise sigmoid operation.

		:param x: ndarray of inputs to be sent to the sigmoid function
		:type x: ndarray
		:return: tuple of (1) activation outputs and (2) the gradients of the
					sigmoid function, both ndarrays of the same shape as x
		:rtype: tuple
		"""
		y = 1 / (1 + np.exp(-x))
		grad = y * (1 - y)

		return y, grad

	def relu(self, x):
		# t3 = time.time()
		y = np.maximum(0,x)
		# print(f"activation max time: {time.time() -t3}")
		# t4 = time.time()
		grad = np.where(x<0, 0, 1)
		# print(f"grad time: {time.time() -t4}")
		# print(f"total relu time: {time.time() - t3}")
		return y, grad


	# Loss function
	def nll(self, score, labels):
		"""
		Compute the loss function as the negative log-likelihood of the labels by
									interpreting scores as probabilities.
		:param score: 8xn vector of positive-class probabilities
		:type score: array
		:param labels: length-n array of labels in {0-8}
		:type labels: array
		:return: tuple containing (1) the scalar negative log-likelihood (NLL) and (2) the length-n gradient of the NLL with respect to the scores.
		:rtype: tuple
		"""
		labels08 = np.copy(labels) # create a copy of labels in {0-8}
		labels08[labels08 < 0] = 0
		score[score == 0] = 10**-10
		score[score == 1] = 1-10**-10

		# Convert labels to a one hot encoding
		labels_onehot = np.eye(self.NUM_CLASSES)[labels08]
		labels_onehot_bool = np.array(labels_onehot, dtype=bool)
		true_scores = score[labels_onehot_bool.T]

		# Compute objective
		objective = np.sum(-np.log(true_scores))
	
		# gradient = np.zeros((new_weights.shape))
		# for i in range(num_classes):
		# 	sum = np.zeros(data.shape[0])
		# 	for j, example in enumerate(np.transpose(data)):
		# 		indicator = 0
		# 		if i == labels[j]:
		# 			indicator = 1
		# 		sum += example * (P_mat[i, j] - indicator)
			# gradient[:,i] = sum
	
	
		# Compute gradient
		grad = np.divide((score - labels08), (score * (1 - score)))
		grad = np.nan_to_num(grad)
	
		# grad = np.divide(1, (1-true_scores))
		# print(f"grad.shape: {grad.shape}")
		# print(f"NUM NANS in GRAD:")
		# print(np.count_nonzero(np.isnan(grad)))
	
		return objective, grad


	def mlp_predict(self, data, model, pre_augmented=False):
		"""
		Predict binary-class labels for a batch of test points

		:param data: ndarray of shape (14400, n), where each column is a data example
		:type data: ndarray
		:param model: learned model from mlp_train
		:type model: dict
		:return: array of +1 or -1 labels
		:rtype: array
		"""
		start = time.time()

		# n = data.shape[1]

		weights = model['weights']
		activation_function = model['activation_function']

		num_layers = len(weights)
		# print(f"weights.shape: {len(weights), len(weights[0])}")

		activation_derivatives = []
		# store all activations of neurons. Start with the data augmented with an all-ones feature
		activations = []
		if pre_augmented:
			activations = copy.copy(data)
		else:
			activations = [np.vstack((data, np.ones((1, data.shape[1]))))]
		
		# print(f"data.shape: {data.shape}")
		start2 = time.time()
		for layer in range(num_layers):
			# Compute activations and activation derivatives for layer 
			# (this is taking the most time due to the large input layer size)
			new_activations, activation_derivative = activation_function(weights[layer].dot(activations[layer]))
			activations.append(new_activations)
			activation_derivatives.append(activation_derivative)

		# Select between each class
		scores = softmax(activations[-1], axis=0)
		labels = np.argmax(scores, axis=0)

		return labels, scores, activations, activation_derivatives


	def mlp_objective(self, model, data, labels, loss_function):
		"""
		Compute the objective and gradient of multi-layer perceptron

		:param model: dict containing the current model 'weights'
		:type model: dict
		:param data: ndarray of shape (f, n), where each column is a data example
		:type data: ndarray
		:param labels: length-n array of labels in {+1, -1}
		:type labels: array
		:param loss_function: function that evaluates a vector of estimated
								positive-class probabilities against a vector of
								ground-truth labels. Returns a tuple containing the
								loss and its gradient.
		:type loss_function: function
		:return: tuple containing (1) the scalar loss objective value and (2) the
									list of gradients for each weight matrix
		:rtype: tuple
		"""
		n = labels.size

		# forward propagation
		weights = model['weights']
		num_layers = len(weights)
	
		# print(f"Calculating scores, activations, and activation_derivatives (num_layers: {num_layers})")
		
		predict_time = time.time()
		# Forward pass
		predicted_labels, scores, activations, activation_derivatives = mlp_predict(data, model, pre_augmented=True)

		start2 = time.time()

		# Back propagation
		layer_delta = [None] * num_layers
		layer_gradients = [None] * num_layers

		# Compute objective value and gradient of the loss function
		objective, gradient = loss_function(scores, labels)
		gradient = np.transpose(gradient)
		layer_delta[-1] = gradient.reshape((8, -1))

		# Back-propagate error to previous layers
		for i in reversed(range(num_layers - 1)):
			layer_delta[i] = weights[i+1].T.dot(layer_delta[i+1] * activation_derivatives[i+1])
	
		layer_gradient_computation_time = time.time()
	
		# Use layer_delta to compute gradients for each layer
		for i in range(num_layers):
			todot = layer_delta[i] * activation_derivatives[i]
			layer_gradients[i] = (todot).dot(activations[i].T)
	
		# print(f"layer_gradient_computation_time: {time.time()-layer_gradient_computation_time}")

		return objective, layer_gradients


	def mlp_train(self, data, labels, params, model=None):
		"""
		Train a multi-layered perceptron (MLP) with gradient descent and
		back-propagation.

		:param data: ndarray of shape (8, n), where each column is a data example
		:type data: ndarray
		:param labels: array of length n whose entries range from 1-8
		:type labels: array
		:param params: dictionary containing model parameters, most importantly
						'num_hidden_units', which is a list
						containing the number of hidden units in each hidden layer.
		:type params: dict
		:return: learned MLP model containing 'weights' list of each layer's weight
				matrix
		:rtype: dict
		"""
		# input dim = 14400
		input_dim, num_train = data.shape

		if not model:
			# no initial model was provided. Initialize model based on params
			model = dict()
			num_hidden_units = params['num_hidden_units']
			model['num_hidden_units'] = num_hidden_units
			# store weight matrices for each model layer in a list
			model['weights'] = list()
			# create input layer
			model['weights'].append(0.1 * np.random.randn(num_hidden_units[0], input_dim + 1))
			# create intermediate layers
			for layer in range(1, len(num_hidden_units)):
				model['weights'].append(0.1 * np.random.randn(num_hidden_units[layer], num_hidden_units[layer-1]))
			# create output layer
			model['weights'].append(0.1 * np.random.randn(self.NUM_CLASSES, num_hidden_units[-1]))
			model['activation_function'] = params['activation_function']

		loss_function = params['loss_function']
		num_layers = len(model['weights'])
		lam = params['lambda']
	
		# Begin training
		objective = np.zeros(params['max_iter'])
	
		# Augment data before computing objective/grad for speed
		augmented_data = [np.vstack((data, np.ones((1, data.shape[1]))))] 
		for t in tqdm (range(params['max_iter']), desc="Training..."):
			# Compute the objective and gradients
			obj_time = time.time()
			_, grad = mlp_objective(model, augmented_data, labels, loss_function)
			rate = 1.0 / np.sqrt(t + 1)

			total_change = 0
	
			update_layer_time = time.time()

			# Use the gradient to update each layer's weights
			for j in range(num_layers):
				change = - rate * (grad[j] + lam * model['weights'][j])
				total_change += np.sum(np.abs(change))

				# Clip change to [-0.1, 0.1] to avoid numerical errors
				change = np.clip(change, -0.1, 0.1)
	
				model['weights'][j] += change
	
			# print(f"update_layer_time: {time.time() - update_layer_time}")

			if total_change < 1e-8:
				print("Exiting because total change was %e, a sign that we have reached a local minimum." % total_change)
				# Stop if we are no longer changing the weights much
				break

		return model


	def find_optimal_params(self, train_data, train_labels, structures=None, lambda_vals=None):
		"""
		Cross validate model on every combination of given structures and lambda values
		to find the best combination of model architecture and lambda value hyper-parameter.
		:param structure: MLP model structures to train with
		:type structures: list of lists
		:param lambda_vals: lambda values to train with
		:type lambda_vals: list
		:param train_data: ndarray of shape (8, n), where each column is a data example
		:type train_data: ndarray
		:param train_labels: array of length n whose entries range from 1-8
		:type train_labels: array
		"""
		
		# Set default structures to test if not given
		if structures == None:
			structures = [[100,10], [200,50], [7200,7200], [500,100,50], [5000,1000,500]]

		# Set default lambda values to test if not given
		if lambda_vals == None:
			lambda_vals = [0.01, 0.1, 1]
  
		# Initialize best score to 0
		best_score = 0
		num_folds = 4
  
		params = {
			'max_iter': 400,
			'activation_function': relu,
			'loss_function': nll
		}

		# Train a model for each combination of structure and lambda value
		start_all = time.time()
		for j in range(len(structures)):
			for k in range(len(lambda_vals)):
				start_single = time.time()
				params['num_hidden_units'] = structures[j]
				params['lambda'] = lambda_vals[k]
				
				# Cross validate with given structure and lambda values
				cv_score, models = cross_validate(mlp_train, mlp_predict, self.train_data, self.train_labels, num_folds, params)

				# Update params if best score
				if cv_score > best_score:
					best_score = cv_score
					best_params = copy.copy(params)
		
				print(f"For structure {j}, lambda {k}:")
				print(f"Training time: {time.time() - start_single} seconds.")
				print(f"Cross validation score: {cv_score}.\n")
		
		print(f"Total training time: {time.time() - start_all} seconds.")
		print(f"Best model with structure {best_params['num_hidden_units']} and lambda {best_params['lambda']}.")

		self.final_params = best_params
		return best_params

	def load_data(self, show_data=False):
		# Load image data
		print("Loading dataset")
		recaptcha_data_f = h5py.File('recaptcha_data.h5', 'r')
		self.train_data = recaptcha_data_f['train_data'][:]
		self.test_data = recaptcha_data_f['test_data'][:]
		self.train_labels = recaptcha_data_f['train_labels'][:]
		self.test_labels = recaptcha_data_f['test_labels'][:]
		recaptcha_data_f.close()
		print("Data loaded")

		# Normalize grayscale data to between 0-1
		self.train_data = train_data/255
		self.test_data = test_data/255

		if show_data:
			# Show first 10 images and labels
			print(f" train_data: {train_data[:10]}")
			print(f" test_data: {test_data[:10]}")
			print(f" train_labels: {train_labels[:10]}")
			print(f" test_labels: {test_labels[:10]}")
   
	# Train MLP with a given structure and hyper parameter
	def train_with(self, structure, lambda_value):
     
		start_time = time.time()
  
		params = {
			'max_iter': 400,
			'activation_function': relu,
			'loss_function': nll,
			'num_hidden_units': structure,
			'lambda': lambda_value
		}
  
		self.cur_model = mlp_train(self.train_data, self.train_labels, params)		
  		print(f"Training time: {time.time() - start_time}")

  
	# Tests model on all training and testing data. Creates classification report
	def test_model(self):
     
		params = {
			'max_iter': 400,
			'activation_function': relu,
			'loss_function': nll
		}

		best_params = []
		best_score = 0

		train_start = time.time()

		print(f"training")
		best_params = copy.copy(params)
		best_params['num_hidden_units'] = [1000, 1000]
		best_params['lambda'] = .1
					
		mlp_model = mlp_train(train_data, train_labels, best_params)
		predictions, _, _, _ = mlp_predict(test_data, mlp_model)
		test_accuracy = np.mean(predictions == test_labels)

		print(f"MLP had test accuracy {test_accuracy}")
		print(f"with structure {self.final_params['num_hidden_units']} and lambda {self.final_params['lambda']}.")
		print(f"Total testing time: {time.time() - train_start}.")

		test_images_truth = self.test_labels
  
		print(classification_report(test_images_truth, predictions, digits=3))