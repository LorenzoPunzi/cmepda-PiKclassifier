"""
"""
import numpy as np
from matplotlib import pyplot as plt
from machine_learning.dtc import dt_classifier
from utilities.utils import default_txtpaths

if __name__ == '__main__':
    predicted_array, eff, misid = dt_classifier(('txt',default_txtpaths()), print_tree='')

    print(predicted_array.size)
    subdata = np.split(predicted_array,20)

    fractions = [data.sum()/len(data) for data in subdata]

    plt.figure('Fraction distribution for dtc')
    plt.hist(fractions,bins=300, histtype='step')
    plt.savefig('fig/fractionsdtc.pdf')
    plt.show()




