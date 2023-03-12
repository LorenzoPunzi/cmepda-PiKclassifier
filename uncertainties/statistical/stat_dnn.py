"""
"""

import sys
import numpy as np
from keras.models import Model, model_from_json
from matplotlib import pyplot as plt
from utilities.dnn_settings import dnn_settings
from machine_learning.deepnn import eval_dnn
from utilities.exceptions import InvalidSourceError
from utilities.utils import find_cut, default_txtpaths, default_figpath


def stat_dnn(source=('array', ("", "", "")),
             trained_model=('../../machine_learning/deepnn.json',
                            '../../machine_learning/deepnn.h5'),
             eff=0.95, stat_split=100, figpath=''):

    with open(trained_model[0], 'r') as json_file:
        loaded_model_json = json_file.read()
    deepneunet = model_from_json(loaded_model_json)
    deepneunet.load_weights(trained_model[1])

    try:
        if source[0] == 'array':
            pi_eval, k_eval, data_eval = source[1]
        elif source[0] == 'txt':
            train_set = np.loadtxt(source[1][0])
            data_set = np.loadtxt(source[1][1])
            pi_set = np.array([train_set[i, :] for i in range(
                np.shape(train_set)[0]) if train_set[i, -1] == 0])
            k_set = np.array([train_set[i, :] for i in range(
                np.shape(train_set)[0]) if train_set[i, -1] == 1])
            pi_eval = eval_dnn(deepneunet, pi_set, flag_data=False)
            k_eval = eval_dnn(deepneunet, k_set, flag_data=False),
            data_eval = eval_dnn(deepneunet, data_set, flag_data=True)
        else:
            raise InvalidSourceError(source[0])
    except InvalidSourceError as err:
        print(err)
        sys.exit()

    y_cut, misid = find_cut(pi_eval, k_eval, eff)

    subdata = np.array_split(data_eval, stat_split)
    fractions = [((dat > y_cut).sum()/dat.size-misid)/(eff-misid)
                 for dat in subdata]

    stat_err = np.sqrt(np.var(fractions, ddof=1, dtype='float64'))

    plt.figure('Fraction distribution for deepnn')
    plt.hist(fractions, bins=20, histtype='step')
    plt.savefig(default_figpath('fractionsdnn')) if figpath == '' \
        else plt.savefig(figpath+'/dnn_distrib.png')

    return stat_err


if __name__ == '__main__':

    """
    settings = dnn_settings(layers = [75, 60, 45, 30, 20], batch_size = 128, epochnum = 200, learning_rate = 5e-5, showhistory = False)

    pi_eval, k_eval, data_eval = dnn(settings = settings)
    efficiency = 0.95

    y_cut, misid = find_cut(pi_eval, k_eval, efficiency)
    plt.axvline(x=y_cut, color='green', label='y cut for '
                + str(efficiency)+' efficiency')
    plt.legend()
    plt.savefig('fig/ycut.pdf')
    """
    a = np.zeros(100)
    b = np.ones(100)
    c = np.ones(100)*2
    stat_dnn(source=('array', (a, b, c)), stat_split=10)

    plt.show()
