"""
"""

import numpy as np
from matplotlib import pyplot as plt
from utilities.dnn_settings import dnn_settings
from machine_learning.deepnn import dnn
from utilities.utils import find_cut , default_txtpaths, default_figpath
from keras.models import Model, model_from_json

def stat_dnn(paths = default_txtpaths(), json_path = '../../machine_learning/deepnn.json',weights_path = '../../machine_learning/deepnn.h5', efficiency = 0.95):

    train_path, data_path = paths
    train_set, data_set = np.loadtxt(train_path), np.loadtxt(data_path) 

    pi_set = np.array([train_set[i, :] for i in range(
        np.shape(train_set)[0]) if train_set[i, -1] == 0])
    k_set = np.array([train_set[i, :] for i in range(
        np.shape(train_set)[0]) if train_set[i, -1] == 1])

    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    deepneunet = model_from_json(loaded_model_json)
    deepneunet.load_weights(weights_path)

    pi_eval, k_eval, data_eval = deepneunet.predict(pi_set), deepneunet.predict(k_set), deepneunet.predict(data_set)

    y_cut, misid = find_cut(pi_eval, k_eval, efficiency)
    subdata = np.split(data_eval,400)

    fractions = [((dat > y_cut).sum()/dat.size-misid)/(efficiency-misid) for dat in subdata]

    plt.figure('Fraction distribution for deepnn')
    plt.hist(fractions,bins=20, histtype='step')
    plt.savefig(default_figpath('fractionsdnn'))




if __name__ == '__main__':

    """
    settings = dnn_settings(layers = [75, 60, 45, 30, 20], batch_size = 128, epochnum = 200, learning_rate = 5e-5 showhistory = False)

    pi_eval, k_eval, data_eval = dnn(settings = settings)
    efficiency = 0.95

    y_cut, misid = find_cut(pi_eval, k_eval, efficiency)
    plt.axvline(x=y_cut, color='green', label='y cut for '
                + str(efficiency)+' efficiency')
    plt.legend()
    plt.savefig('fig/ycut.pdf')
    """
    stat_dnn()
    
    plt.show()


    
