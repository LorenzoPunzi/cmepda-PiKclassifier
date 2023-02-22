"""
Analyses the output dnn() function in training.py
"""

import numpy as np
import matplotlib.pyplot as plt


def find_cut(pi_array, k_array, efficiency, inverse_mode = False):

    y = -np.sort(-k_array)[int(efficiency*(len(k_array)-1))] if not inverse_mode else np.sort(pi_array)[int(efficiency*(len(k_array)-1))]
    misid = (pi_array>y).sum()/pi_array.size if not inverse_mode else (k_array>y).sum()/k_array.size
    #.sum() sums how many True there are in the masked (xx_array>y)
    return y , misid

def roc(pi_array, k_array, eff_span = (0.5,0.95,50)):
    rocpnts = [((pi_array>y).sum()/pi_array.size,(k_array>y).sum()/k_array.size) for y in np.linspace(*eff_span)]
    rocx, rocy = zip(*rocpnts)
    plt.figure('roc')
    plt.plot(rocx,rocy)
    plt.xlabel('False Positive Probability')
    plt.ylabel('True Positive Probability')
    plt.draw()
    plt.savefig('./fig/roc.pdf')

if __name__ == '__main__':
    
    from training import dnn

    pi_eval, k_eval, data_eval = dnn('train_array_prova.txt','data_array_prova.txt',flagged_data=True)
    efficiency = 0.9

    y_cut , misid = find_cut(pi_eval, k_eval, efficiency)
    plt.axvline(x = y_cut, color = 'green', label = 'y cut for '+str(efficiency)+' efficiency')
    plt.legend()
    plt.savefig('./fig/ycut.pdf')
    roc(pi_eval,k_eval)
    
    print(f'y cut is {y_cut} , misid is {misid}')
    f = ((data_eval>y_cut).sum()/data_eval.size-misid)/(efficiency-misid)
    print(f'The estimated fraction of K events is {f}') 

    plt.show()
