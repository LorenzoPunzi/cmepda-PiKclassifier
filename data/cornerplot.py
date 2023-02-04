"""
Generates corner plots between the specified numpy arrays in order to evaluate the separation better
"""

import numpy as np
import corner
from matplotlib import pyplot as plt
from import_functions import loadvars

def cornerplot(array,figname):

    array_new = array[:,:-1]
    figure = corner.corner(array_new)
    figure.set_size_inches(9,7)
    figure.savefig(figname)
    #corner.overplot_lines(figure, np.mean(array_new,axis=0), color="C1")
    #corner.overplot_lines(figure, np.mean(array_new,axis=1), color="C2")

    return figure

#if __name__ == '___main__':

file_mc_pi = '../root_files/tree_B0PiPi_mc.root'
file_mc_k = '../root_files/tree_B0sKK_mc.root'
tree = 't_M0pipi;2'
vars = ('M0_Mpipi', 'M0_MKK', 'M0_p', 'M0_eta', 'h1_p', 'h1_eta', 'h2_p', 'h2_eta')

arr_mc_pi, arr_mc_k = loadvars(file_mc_pi, file_mc_k, tree, vars)
print("test")

figure_pi = cornerplot(arr_mc_pi,'fig/cornerplot_pi.png')
figure_k = cornerplot(arr_mc_k,'fig/cornerplot_k.png')

plt.show()

