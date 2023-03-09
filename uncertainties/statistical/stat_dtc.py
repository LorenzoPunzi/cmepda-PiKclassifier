"""
"""
import numpy as np
from matplotlib import pyplot as plt
from machine_learning.dtc import dt_classifier
from utilities.utils import default_txtpaths, default_figpath

if __name__ == '__main__':
    predicted_array, eff, misid = dt_classifier(('txt',default_txtpaths()), print_tree='')

    print(predicted_array.size)
    subdata = np.split(predicted_array,400)

    fractions = [data.sum()/len(data) for data in subdata]

    plt.figure('Fraction distribution for dtc')
    plt.hist(fractions,bins=20, histtype='step')
    plt.savefig(default_figpath('fractionsdtc'))
    plt.show()




