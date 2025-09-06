from math import e as eul
from torch import tensor, float32

from settings import *

def sigmoid_prime(x):
    return (eul ** (-x)) / ((1 + eul ** (-x)) ** 2)

def tensor32(input):
    return tensor(input, dtype = float32)

def max_index_1D(input_array):
    assert input_array.dim() == 1

    max_index = 0

    for iter in range(len(input_array)):
        if input_array[max_index] < input_array[iter]:
            max_index = iter
    return max_index
