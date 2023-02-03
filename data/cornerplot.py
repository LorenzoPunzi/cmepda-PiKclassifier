"""
Generates corner plots between the specified numpy arrays in order to evaluate the separation better
"""

import numpy as np
import corner
from matplotlib import pyplot as plt

def cornerplot(array):

    return corner.corner(array[:,:-1])


#if __name__ == '___main__':
arr_trainset = np.loadtxt('./txt/train_array_prova.txt')
figure = cornerplot(arr_trainset)
figure.set_size_inches(5, 5)
figure.savefig('cornerplot.png')
plt.show()

