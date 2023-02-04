"""
Generates corner plots between the specified numpy arrays in order to evaluate the separation better
"""

import numpy as np
import corner
from matplotlib import pyplot as plt

def cornerplot(array):

    array_new = array[:,:-1]
    figure = corner.corner(array_new)
    corner.overplot_lines(figure, np.mean(array_new,axis=0), color="C1")
    corner.overplot_lines(figure, np.mean(array_new,axis=1), color="C2")
    #corner.overplot_points(figure, value1[None], marker="s", color="C1")
    #corner.overplot_points(figure, value2[None], marker="s", color="C2")

    return figure

#if __name__ == '___main__':
arr_trainset = np.loadtxt('./txt/train_array_prova.txt')
figure = cornerplot(arr_trainset)
figure.set_size_inches(9,7)
figure.savefig('cornerplot.png')
plt.show()

