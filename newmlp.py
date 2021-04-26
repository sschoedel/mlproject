from __future__ import division
import h5py
import cv2
import numpy as np
import copy
import os
import time
import json
from scipy.special import softmax
from tqdm import tqdm

dir_names = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Crosswalk', 'Hydrant', 'Palm', 'Traffic Light']
NUM_CLASSES = len(dir_names)

def cross_validate(trainer, predictor, all_data, all_labels, folds, params):
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

def logistic(x):
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

def relu(x):
    # t3 = time.time()
    y = np.maximum(0,x)
    # print(f"activation max time: {time.time() -t3}")
    # t4 = time.time()
    grad = np.where(x<0, 0, 1)
    # print(f"grad time: {time.time() -t4}")
    # print(f"total relu time: {time.time() - t3}")
    return y, grad


def nll(score, labels):
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
 
	print(f"score.shape: {score.shape}")
 
	print(f"labels08.shape: {labels08.shape}")
	print(f"labels08 sample: {labels08}")
 
	labels_onehot = np.eye(NUM_CLASSES)[labels08]
	print(f"labels_onehot.shape: {labels_onehot.shape}")
	print(f"labels_onehot sample: {labels_onehot[5000:5010]}")
	
	# objective = np.sum(labels08 * np.log(score) - (1 - labels08) * np.log(1 - score), axis=1)
	labels_onehot_bool = np.array(labels_onehot, dtype=bool)
	true_scores = score[labels_onehot_bool.T]
	print(f"true_scores.shape: {true_scores.shape}")
	print(f"true_scores sample: {true_scores[5000:5010]}")
	objective = np.sum(-np.log(true_scores))
	print(f"objective: {objective}")
	
	print(f"NUM ZEROS IN SCORES (inside nll)!:::")
	print(np.count_nonzero(score==0))
 
 
	gradient = np.zeros((new_weights.shape))
	for i in range(num_classes):
		sum = np.zeros(data.shape[0])
		for j, example in enumerate(np.transpose(data)):
			indicator = 0
			if i == labels[j]:
				indicator = 1
			sum += example * (P_mat[i, j] - indicator)
		gradient[:,i] = sum
	# grad = np.divide((score - labels08), (score * (1 - score)))
	# grad = np.divide(1, (1-true_scores))
	print(f"grad.shape: {grad.shape}")
	print(f"NUM NANS in GRAD:")
	print(np.count_nonzero(np.isnan(grad)))

	return objective, grad


def mlp_predict(data, model, pre_augmented=False):
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
		# t = time.time()
		# print(f"for layer {layer}, weights[layer].shape: {weights[layer].shape}")
		# print(f"activations[layer].shape: {activations[layer].shape}")
		# dot_layer = weights[layer].dot(activations[layer])
		# print(f"weights[layer].shape: {weights[layer].shape}")
		# print(f"activations[layer].shape: {activations[layer].shape}")
		# dot_layer = np.tensordot(weights[layer], activations[layer], axes=1)
		# print(f"dot computation for layer {layer}: {time.time() - t}")
		# print(f"shape dot_layer: {dot_layer.shape}")
		new_activations, activation_derivative = activation_function(weights[layer].dot(activations[layer]))
		# t2 = time.time()
		activations.append(new_activations)
		activation_derivatives.append(activation_derivative)
		# print(f"append time: {time.time() - t2}")
		# print(f"total activation time for layer {layer}: {time.time() - t}")

	# print(f"activation time: {time.time() - start2}")
 
	# TODO: Make this select between all 8 classes
 
	# for activation in activations:
		# print(f"activation.shape: {activation.shape}")
 
	scores = softmax(activations[-1], axis=0)
	# labels = 2 * (scores > 0.5) - 1
	labels = np.argmax(scores, axis=0)
 
	print(f"NUM ZEROS IN SCORES (inside predict)!:::")
	print(np.count_nonzero(scores==0))
 
	# score_sums = np.sum(scores, axis=0)
	# print(f"score_sums.shape: {score_sums.shape}")
	# print(f"first 50 score sums: {score_sums[:50]}")
	# print(f"scores.shape: {scores.shape}")
	# print(f"labels.shape: {labels.shape}")
 
	# print(f"predict time: {time.time() - start}")

	return labels, scores, activations, activation_derivatives


def mlp_objective(model, data, labels, loss_function):
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
	predicted_labels, scores, activations, activation_derivatives = mlp_predict(data, model, pre_augmented=True)
	# print(f"predict_time: {time.time() - predict_time}")
	# print(f"blank_labels length: {len(_)}")
	# print(f"scores.shape: {scores.shape}")
	# for i in range(len(activations)):
	# 	print(f"activations[{i}].shape: {activations[i].shape}")
    
	# for i in range(len(activation_derivatives)):
	# 	print(f"activation_derivatives[{i}].shape: {activation_derivatives[i].shape}")

	start2 = time.time()

	# back propagation
	layer_delta = [None] * num_layers
	layer_gradients = [None] * num_layers

	# compute objective value and gradient of the loss function
	objective, gradient = loss_function(scores, labels)
	gradient = np.transpose(gradient)
	layer_delta[-1] = gradient.reshape((8, -1))

	# print(f"objective.shape: {objective.shape}")
	# print(f"gradient.shape: {gradient.shape}")
	# print(objective)

	# back-propagate error to previous layers
	for i in reversed(range(num_layers - 1)):
		# print(f"back propagating layer {i+1} to layer {i}")
		# print(f"weights[i+1].shape: {weights[i+1].shape}")
		# print(f"layer_delta[i+1].shape: {layer_delta[i+1].shape}")
		# print(f"activation_ders[i+1].shape: {activation_derivatives[i+1].shape}")
		# toMult = layer_delta[i+1] * activation_derivatives[i+1]
		# print(f"layer_delta[i+1] * activation_ders[i+1].shape: {toMult.shape}")
		layer_delta[i] = weights[i+1].T.dot(layer_delta[i+1] * activation_derivatives[i+1])
  

	# print(f"backpropagation time: {time.time() - start2}")
 
	layer_gradient_computation_time = time.time()
	# use layer_delta to compute gradients for each layer
	for i in range(num_layers):
		t = time.time()
		print(f"layer_delta[{i}]: {layer_delta[i]}")
		# unique1, counts1 = np.unique(layer_delta[i], return_counts=True)
		# print(dict(zip(unique1, counts1)))
		print(np.count_nonzero(np.isnan(layer_delta[i])))
		print(f"activation_derivatives[{i}]: {activation_derivatives[i]}")
		# unique2, counts2 = np.unique(activation_derivatives[i], return_counts=True)
		# print(dict(zip(unique2, counts2)))
		print(np.count_nonzero(np.isnan(activation_derivatives[i])))
		todot = layer_delta[i] * activation_derivatives[i]
		layer_gradients[i] = (todot).dot(activations[i].T)
		# print(f"layer {i} time: {time.time() - t}")
  
	# print(f"layer_gradient_computation_time: {time.time()-layer_gradient_computation_time}")

	return objective, layer_gradients


def mlp_train(data, labels, params, model=None):
	"""
	Train a multi-layered perceptron (MLP) with gradient descent and
	back-propagation.

	:param data: ndarray of shape (2, n), where each column is a data example
	:type data: ndarray
	:param labels: array of length n whose entries are all +1 or -1
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

	# print(f"train_data shape: {train_data.shape}, train_data: {train_data[:10,:]}")
	# print(f"test_data shape: {test_data.shape}, test_data: {test_data[:10,:]}")
	# print(f"input_dim: {input_dim}, num_train: {num_train}")
	# print(f"labels.shape: {labels.shape}")

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
		model['weights'].append(0.1 * np.random.randn(NUM_CLASSES, num_hidden_units[-1]))
		model['activation_function'] = params['activation_function']

	loss_function = params['loss_function']
	num_layers = len(model['weights'])
	lam = params['lambda']
 
	# for i in range(len(model['weights'])):
		# print(f"model['weights'][{i}].shape: {model['weights'][i].shape}")

	# Begin training

	objective = np.zeros(params['max_iter'])
	augmented_data = [np.vstack((data, np.ones((1, data.shape[1]))))] # Augment data before computing objective/grad for speed
	for t in tqdm (range(params['max_iter']), desc="Training..."):
		# print(f"computing objective and gradient for iteration {t}")
		# compute the objective and gradient, which is a list of gradients for each layer's weight matrices
		# objective[t], grad = mlp_objective(model, data, labels, loss_function)
		obj_time = time.time()
		_, grad = mlp_objective(model, augmented_data, labels, loss_function)
		# print(f"mlp_objective time: {time.time() - obj_time}")
		rate = 1.0 / np.sqrt(t + 1)

		total_change = 0
  
		update_layer_time = time.time()

		# use the gradient to update each layer's weights
		for j in range(num_layers):
			change = - rate * (grad[j] + lam * model['weights'][j])
			total_change += np.sum(np.abs(change))

			# clip change to [-0.1, 0.1] to avoid numerical errors
			change = np.clip(change, -0.1, 0.1)
   
			# print(f"change: {change}")

			model['weights'][j] += change
   
		# print(f"update_layer_time: {time.time() - update_layer_time}")

		if total_change < 1e-8:
			print("Exiting because total change was %e, a sign that we have reached a local minimum." % total_change)
			# stop if we are no longer changing the weights much
			break

	return model

# Open json file to which the model will be saved
json_file = open("model.json", "w")

# Load image data
print("Loading dataset")
recaptcha_data_f = h5py.File('recaptcha_data.h5', 'r')
train_data = recaptcha_data_f['train_data'][:]
test_data = recaptcha_data_f['test_data'][:]
train_labels = recaptcha_data_f['train_labels'][:]
test_labels = recaptcha_data_f['test_labels'][:]
recaptcha_data_f.close()
print("Data loaded")

# Normalize grayscale data to between 0-1
train_data = train_data/255
test_data = test_data/255

# Show first 10 images and labels
print(f" train_data: {train_data.shape}")
# print(train_data[:10,:])
print(f" test_data: {test_data.shape}")
# print(test_data[:10,:])
print(f" train_labels: {train_labels.shape}")
# print(train_labels[:10])
print(f" test_labels: {test_labels.shape}")
# print(test_labels[:10])


# set constants for convenience
num_models = 5
num_folds = 4

# initialize matrix to store test accuracies
test_accuracy = np.zeros(num_models)


structures = [[7200,7200], [10], [100,10], [200,50], [500,100,50]]
lambda_vals = [0.01, 0.1, 1]

params = {
	'max_iter': 400,
	'activation_function': relu,
	'loss_function': nll
}

best_params = []
best_score = 0

train_start = time.time()

# for j in range(len(structures)):
# 	for k in range(len(lambda_vals)):
# 		print(f"training on structure {j} lambda {k}")
# 		params['num_hidden_units']= structures[j]
# 		params['lambda'] = lambda_vals[k]
		
# 		cv_score, models = cross_validate(mlp_train, mlp_predict, train_data, train_labels, num_folds, params)
  
# 		if cv_score > best_score:
# 			best_score = cv_score
# 			best_params = copy.copy(params)

print(f"training")
best_params = copy.copy(params)
best_params['num_hidden_units'] = [10]
best_params['lambda'] = .1
# best_score, models = cross_validate(mlp_train, mlp_predict, train_data, train_labels, num_folds, best_params)
			
mlp_model = mlp_train(train_data, train_labels, best_params)
predictions, _, _, _ = mlp_predict(test_data, mlp_model)
test_accuracy = np.mean(predictions == test_labels)

print("MLP had test accuracy %f" % (test_accuracy))
print("with structure %s and lambda = %f" % (repr(best_params['num_hidden_units']), best_params['lambda']))
print(f"Total training time: {time.time() - train_start}")

# json.dump(mlp_model, json_file, indent=4)

print(type(mlp_model))
print(f"model: {mlp_model}")