"""
"""

import numpy as np
from matplotlib import pyplot as plt
from utilities.dnn_settings import dnn_settings
from machine_learning.deepnn import train_dnn , eval_dnn
from utilities.utils import default_txtpaths



if __name__ == '__main__':

    settings = dnn_settings()
    settings.layers = [75, 60, 45, 30, 20]
    settings.batch_size = 128
    settings.epochnum = 10
    settings.verbose = 2
    settings.batchnorm = False
    settings.dropout = 0.005
    settings.learning_rate = 5e-4
    settings.showhistory =False

    train_array_path, data_array_path = default_txtpaths
    training_set, data_set = np.loadtxt(train_array_path), np.loadtxt(data_array_path)
    deepnn = train_dnn(training_set, settings)
    
