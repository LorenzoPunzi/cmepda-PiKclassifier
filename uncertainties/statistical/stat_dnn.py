"""
"""

import numpy as np
from matplotlib import pyplot as plt
from utilities.dnn_settings import dnn_settings
from machine_learning.deepnn import dnn
from utilities.utils import find_cut




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

    pi_eval, k_eval, data_eval = dnn(settings)
    efficiency = 0.95

    y_cut, misid = find_cut(pi_eval, k_eval, efficiency)
    plt.axvline(x=y_cut, color='green', label='y cut for '
                + str(efficiency)+' efficiency')
    plt.legend()
    plt.savefig('fig/ycut.pdf')

    subdata = np.split(data_eval,100)

    fractions = [((dat > y_cut).sum()/dat.size-misid)/(efficiency-misid) for dat in subdata]

    plt.figure('Fraction distribution')
    plt.hist(fractions,bins=300, histtype='step')
    plt.savefig('fig/fractionsdnn.py')
    plt.show()


    
