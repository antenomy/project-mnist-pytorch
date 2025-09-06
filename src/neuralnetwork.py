from torch import tensor, zeros, dot, flatten, mv, zeros_like
from torch.nn import Sigmoid
from torch.linalg import lstsq
from operator import add

from settings import *
from functions import *

class NeuralNetwork:
    def __init__(self, layer_size_array):
        self.layer_sizes = layer_size_array
        self.layer_count = len(layer_size_array)
        self.sigmoid = Sigmoid()

        iteration_range = range(len(layer_size_array) - 1)
        self.weight_layers = [zeros([layer_size_array[iter],layer_size_array[iter + 1]]) for iter in iteration_range]
        self.bias_layers =   [zeros(layer_size_array[iter]) for iter in iteration_range]


    def backpropagate_neuron(self,  layer_index: int,
                                    neuron_index: int, 
                                    current_neuron_value: tensor, 
                                    reference_value: tensor, 
                                    previous_layer_values: tensor):
        
        # Calculates the delta for weights and bias in one neuron
    
        bias_delta = (sigmoid_prime(dot(self.weight_layers[layer_index][neuron_index], previous_layer_values)) 
                        * 2 * ((current_neuron_value - reference_value))) + LEARNING_RATE

        weight_delta_vector = previous_layer_values * bias_delta

        return weight_delta_vector, bias_delta


    def backpropagate(self, input_array: tensor,
                            correct_index: int):
        
        weight_delta_array_list = [zeros_like(layer) for layer in self.weight_layers]
        bias_delta_array_list = [zeros_like(layer) for layer in self.bias_layers]

        reference_values = zeros([self.layer_sizes[0]])
        reference_values[int(correct_index)] += 1
        
        # Calculate the resulting layers using the input array
        _, layer_array = self.solve_nn(input_array)

        for layer_index in range(self.layer_count - 1):
            weight_delta_array = zeros_like(self.weight_layers[layer_index])
            bias_delta_array = zeros_like(self.bias_layers[layer_index])

            current_layer_values = layer_array[self.layer_count - 1 - layer_index]
            previous_layer_values = layer_array[self.layer_count - 2 - layer_index]

            for iter in range(len(self.weight_layers[layer_index])):
                # print(f"Layer: {layer_index}, neuron {iter}")
                # print(len(reference_values))
                weight_delta_array[iter], bias_delta_array[iter] = self.backpropagate_neuron(layer_index = layer_index,
                                                                                            neuron_index = iter,
                                                                                            current_neuron_value = current_layer_values[iter],
                                                                                            reference_value = reference_values[iter],
                                                                                            previous_layer_values = previous_layer_values)

            # Calculate reference values for next layer
            # print(reference_values.shape)
            reference_values = lstsq(self.weight_layers[layer_index], (reference_values + self.bias_layers[layer_index])).solution
            # print(reference_values)
            # print(self.weight_layers[layer_index].shape)
            # print(self.bias_layers[layer_index].shape)

            weight_delta_array_list[layer_index], bias_delta_array_list[layer_index] = weight_delta_array, bias_delta_array
            
        return weight_delta_array_list, bias_delta_array_list


    def adjust_paramters(self, weights: list, biases: list):

        self.weight_layers = list(map(add, weights, self.weight_layers))
        self.bias_layers = list(map(add, biases, self.bias_layers))


    def calculate_layer(self, input_vector, weight_array, bias_vector):
        # print(mv(weight_array, input_vector) + bias_vector)
        return self.sigmoid(mv(weight_array, input_vector) + bias_vector)


    def solve_nn(self, input_array : tensor):

        layer_array = list()
        current_layer = flatten(input_array)
        layer_array.append(current_layer)

        for iter in range(len(self.weight_layers), 0, -1):

            current_layer = self.calculate_layer(input_vector = current_layer,
                                                    weight_array = self.weight_layers[iter - 1], 
                                                    bias_vector  = self.bias_layers[iter - 1])
            
            layer_array.append(current_layer)
        
        max_index = max_index_1D(current_layer)

        return max_index, layer_array
