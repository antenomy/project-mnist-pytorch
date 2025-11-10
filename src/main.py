import torch as pt
# from matplotlib import pyplot

import pickle, gzip, time
from operator import add, truediv

from settings import *
from functions import *
from neuralnetwork import NeuralNetwork


def test_NN(test_x, test_y):
    correct_guesses = 0
    result_array = list([0] * pt.numel(test_y))

    for iter in range(len(test_x)):
        result, _ = neural_network.solve_nn(test_x[iter])
        correct_result = int(test_y[iter])
        result_array.append(result == correct_result) 
        correct_guesses += 1 if (result == correct_result) else 0

        # if iter % 100 == 0:
        #     print(result, correct_result)
    
    return correct_guesses



# Load Data
print("Loading MNIST data...")

with gzip.open('data/mnist.pkl.gz', 'rb') as file:
    (train_x, train_y), validation_data, (test_x, test_y) = pickle.load(file, encoding="latin1")

assert train_x[0].shape == test_x[0].shape, "Unmatching train and test data"


# Normalise each Image
train_x, test_x = train_x / 255, test_x / 255
train_x, train_y, test_x, test_y = map(tensor32, (train_x, train_y, test_x, test_y))

print(test_y)


# Create Neural Network
print("Initialising Layers...")
image_size = pt.numel(train_x[0])
neural_network = NeuralNetwork([RESULT_LAYER_SIZE, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, image_size])


# Training Phase
print("Training...")

for batch in range(len(train_x) // BATCH_SIZE):

    # print(neural_network.weight_layers)
    # print(neural_network.bias_layers)

    weight_delta_array_list = [pt.zeros_like(layer) for layer in neural_network.weight_layers]
    bias_delta_array_list = [pt.zeros_like(layer) for layer in neural_network.bias_layers]

    for iter in range(BATCH_SIZE):
        total_index = (batch * 100) + iter
        
        batch_weight_delta_array_list, batch_bias_delta_array_list = neural_network.backpropagate(input_array = train_x[total_index], 
                                                                                                correct_index = train_y[total_index])
        
        weight_delta_array_list = list(map(add, weight_delta_array_list, batch_weight_delta_array_list))
        bias_delta_array_list = list(map(add, bias_delta_array_list, batch_bias_delta_array_list))

    weight_delta_array_list = list(map(lambda x : truediv(x, BATCH_SIZE), weight_delta_array_list))
    bias_delta_array_list = list(map(lambda x : truediv(x, BATCH_SIZE), bias_delta_array_list))

    neural_network.adjust_paramters(weight_delta_array_list, bias_delta_array_list)
    if batch % 25 == 0:
        print(f"Batch {batch} {test_NN(test_x, test_y)}")

print(f"Batch {batch} {test_NN(test_x, test_y)}")
