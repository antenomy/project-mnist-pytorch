from torch import tensor
from neuralnetwork import NeuralNetwork

from functions import *

def unit_test_backpropagation():
    
    test_NN = NeuralNetwork([2, 2])

    test_NN.weight_layers = tensor32([[[0.1, -0.5], [0.2, 0.4]]])
    test_NN.bias_layers = tensor32([[1, -2]])

    input_array = tensor32([5, 6])

    current_node_values, layers = test_NN.solve_nn(input_array)

    print(layers)

    correct_value = (tensor32([1, 1]), tensor32([1]))
    calculated_value = test_NN.backpropagate_neuron(layer_index = 0, 
                                                    neuron_index = 0, 
                                                    current_neuron_value = tensor32(layers[1][0]),
                                                    reference_value = tensor32([1]),
                                                    previous_layer_values = tensor32([5, 6]))

    print("Correct Value:\n", correct_value, '\n\n', "Calculated Value:\n", calculated_value, "\n")

    # assert correct_value[0].all() == calculated_value[0].all(), "Weights"
    # assert correct_value[1] == calculated_value[1], f"Bias, {correct_value[1]}, {calculated_value[1]}"
    print(calculated_value)

    calculated_value = test_NN.backpropagate(input_array,
                                            correct_index = 1)
    
    for iter in range(100):
        weights, biases = test_NN.backpropagate(input_array,
                                            correct_index = 1)
        
        test_NN.adjust_paramters(weights, biases)

        current_node_values, layers = test_NN.solve_nn(input_array)
        
        print(layers[1])

unit_test_backpropagation()
