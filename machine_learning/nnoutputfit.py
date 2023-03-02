"""
Analyses the output dnn() function in training.py
"""

import numpy as np
import matplotlib.pyplot as plt
from machine_learning.training import dnn
from machine_learning.dnn_utils import dnn_settings
import os
from sklearn import metrics


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


"""
def roc_homebrew(pi_array, k_array, eff_span=(0.5, 0.95, 50)):
    rocpnts = [((pi_array > y).sum()/pi_array.size, (k_array
                > y).sum()/k_array.size) for y in np.linspace(*eff_span)]
    rocx, rocy = zip(*rocpnts)
    plt.figure('roc')
    plt.plot(rocx, rocy)
    plt.xlabel('False Positive Probability')
    plt.ylabel('True Positive Probability')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.draw()
    current_path = os.path.dirname(__file__)
    rel_path = './fig'
    figurepath = os.path.join(current_path, rel_path, 'roc.pdf')
    plt.savefig(figurepath)
"""


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


if __name__ == '__main__':

    settings = dnn_settings()
    settings.layers = [75, 60, 45, 30, 20]
    settings.batch_size = 128
    settings.epochnum = 400
    settings.verbose = 2
    settings.learning_rate = 5e-5

    pi_eval, k_eval, data_eval = dnn(
        ['train_array.txt', 'data_array.txt'], settings)
    efficiency = 0.95

    y_cut, misid = find_cut(pi_eval, k_eval, efficiency)
    plt.axvline(x=y_cut, color='green', label='y cut for '
                + str(efficiency)+' efficiency')
    plt.legend()
    plt.savefig('./fig/ycut.pdf')
    _, _, _ = roc(pi_eval, k_eval, eff_line=efficiency)

    print(f'y cut is {y_cut} , misid is {misid}')
    f = ((data_eval > y_cut).sum()/data_eval.size-misid)/(efficiency-misid)
    print(f'The estimated fraction of K events is {f}')

    plt.show()
