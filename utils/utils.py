import numpy as np
import matplotlib.pyplot as plt
from machine_learning.deepnn import dnn
from machine_learning.dnn_utils import dnn_settings
import os
from sklearn import metrics


class dnn_settings:
    """
    """

    def __init__(self):
        self._layers = [20, 20, 15, 10]
        self._epochnum = 200
        self._learning_rate = 0.001
        self._batch_size = 128
        self._batchnorm = False
        self._dropout = 0
        self._verbose = 1
        self._showhistory = True

    @property
    def layers(self):
        return self._layers

    @property
    def epochnum(self):
        return self._epochnum

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batchnorm(self):
        return self._batchnorm

    @property
    def dropout(self):
        return self._dropout

    @property
    def verbose(self):
        return self._verbose

    @property
    def showhistory(self):
        return self._showhistory

    @layers.setter
    def layers(self, neurons_list):
        if not len(neurons_list) == 0:
            self._layers = neurons_list
        else:
            print('ERROR: the "epochs" value cannot be 0')

    @epochnum.setter
    def epochnum(self, epochs):
        if not int(epochs) == 0:
            self._epochnum = int(epochs)

    @learning_rate.setter
    def learning_rate(self, lr):
        if lr > 0:
            self._learning_rate = lr
        else:
            print('ERROR: the learning rate must be > 0')

    @batch_size.setter
    def batch_size(self, batch):
        if not int(batch) == 0:
            self._batch_size = int(batch)
        else:
            print('ERROR: the "batch size" value cannot be 0')

    @batchnorm.setter
    def batchnorm(self, bnorm):
        if type(bnorm) is bool:
            self._batchnorm = bnorm
        else:
            print('ERROR: "batchnorm" method must be a boolean value')

    @dropout.setter
    def dropout(self, dr):
        if dr >= 0:
            self._dropout = dr
        else:
            print('ERROR: dropout rate must be >= 0')

    @verbose.setter
    def verbose(self, verb):
        if (verb == 0 or verb == 1 or verb == 2):
            self._verbose = verb
        else:
            print('ERROR: uncorrect value given to "verbose" method')

    @showhistory.setter
    def showhistory(self, show):
        if type(show) == bool:
            self._showhistory = show
        else:
            print('ERROR: "showhistory" method must be a boolean value')





def find_cut(pi_array, k_array, efficiency, specificity_mode=False, inverse_mode=False):

    if inverse_mode:
        efficiency = 1-efficiency
    cut = -np.sort(-k_array)[int(efficiency*(len(k_array)-1))
                             ] if not specificity_mode else np.sort(pi_array)[int(efficiency*(len(k_array)-1))]
    if inverse_mode:  # !!!! NOT DRY
        misid = (pi_array < cut).sum(
        )/pi_array.size if not specificity_mode else (k_array < cut).sum()/k_array.size
    else:
        misid = (pi_array > cut).sum(
        )/pi_array.size if not specificity_mode else (k_array > cut).sum()/k_array.size

    # .sum() sums how many True there are in the masked (xx_array>y)

    return cut, misid


def roc(pi_array, k_array, inverse_mode=False, makefig=True, eff_line=0):
    true_array = np.concatenate(
        (np.zeros(pi_array.size), np.ones(k_array.size)))
    y_array = np.concatenate((pi_array, k_array))
    rocx, rocy, _ = metrics.roc_curve(true_array, y_array)
    auc = metrics.roc_auc_score(true_array, y_array)
    if inverse_mode:
        rocx, rocy = np.ones(rocx.size)-rocx, np.ones(rocy.size)-rocy
        auc = 1-auc
    if makefig:
        plt.figure('ROC')
        plt.plot(rocx, rocy, label='ROC curve', color='red')
        print(f'AUC of the ROC is {auc}')
        plt.xlabel('False Positive Probability')
        plt.xlim(0, 1)
        plt.ylabel('True Positive Probability')
        plt.ylim(0, 1)
        plt.draw()
        if eff_line:
            plt.axhline(y=eff_line, color='green', linestyle='--', label='efficiency chosen at '+str(eff_line)
                        )
        plt.axline((0, 0), (1, 1), linestyle='--', label='AUC = 0.5')
        plt.legend()
        # !!! How to make it so it saves in the /fig folder of the directory from which the function is CALLED, not the one where nnoutputfit.py IS.
        current_path = os.path.dirname(__file__)
        rel_path = './fig'
        figurepath = os.path.join(current_path, rel_path, 'roc.pdf')
        plt.savefig(figurepath)

    return rocx, rocy, auc

