"""
Analyses the output array of deepneuralnetwork
"""

import numpy as np
import matplotlib.pyplot as plt
from training import deepneuralnetwork


if __name__ == '__main__':

    outarray = deepneuralnetwork('./data/txt/train_array_prova.txt','./data/txt/data_array_prova.txt', plotflag = True, verb = 0)

