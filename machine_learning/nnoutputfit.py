"""
Analyses the output array of deepneuralnetwork
"""

import numpy as np
import matplotlib.pyplot as plt
from training import dnn

def find_cut(pi_array, k_array, efficiency, efficiency_mode = True):

    y = 0
    if efficiency_mode: #Could test what algorithm is fastest
        y = -np.sort(-k_array)[int(efficiency*len(k_array))] # piu o meno 1?
    else:
        y = np.sort(pi_array)[int(efficiency*len(pi_array))]
    
    return y

if __name__ == '__main__':

    pi_eval, k_eval, data_eval = dnn('train_array_prova.txt','data_array_prova.txt',flagged_data=True)
    y_cut = find_cut(pi_eval, k_eval, 0.9)
    plt.axvline(x = y_cut, color = 'green')
    plt.show()

