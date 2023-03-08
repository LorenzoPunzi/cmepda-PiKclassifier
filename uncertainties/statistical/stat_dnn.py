"""
"""

import numpy as np
from matplotlib import pyplot as plt
from utilities.dnn_settings import dnn_settings
from machine_learning.deepnn import dnn, train_dnn
from utilities.utils import find_cut, default_txtpaths




if __name__ == '__main__':

    settings = dnn_settings()
    settings.layers = [75, 60, 45, 30, 20]
    settings.batch_size = 128
    settings.epochnum = 100
    settings.verbose = 2
    settings.batchnorm = False
    settings.dropout = 0.005
    settings.learning_rate = 5e-4
    settings.showhistory =False

    mc_path, data_path = default_txtpaths()
    training_set, data_set = np.loadtxt(mc_path), np.loadtxt(data_path)

    deepneunet = train_dnn(training_set, settings, savefig=True)

    pi_set = np.array([training_set[i, :] for i in range(
        np.shape(training_set)[0]) if training_set[i, -1] == 0])
    k_set = np.array([training_set[i, :] for i in range(
        np.shape(training_set)[0]) if training_set[i, -1] == 1])

    pi_eval, k_eval, data_eval = deepneunet.predict(pi_set), deepneunet.predict(k_set), deepneunet.predict(data_set)
    efficiency = 0.95

    y_cut, misid = find_cut(pi_eval, k_eval, efficiency)
    plt.axvline(x=y_cut, color='green', label='y cut for '
                + str(efficiency)+' efficiency')
    plt.legend()
    plt.savefig('fig/ycut.pdf')

    subdata = np.split(data_eval,400)

    fractions = [((dat > y_cut).sum()/dat.size-misid)/(efficiency-misid) for dat in subdata]

    plt.figure('Fraction distribution for deepnn')
    plt.hist(fractions,bins=20, histtype='step')
    plt.savefig('fig/fractionsdnn.pdf')
    plt.show()


    
