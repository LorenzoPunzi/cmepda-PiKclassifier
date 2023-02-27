"""
Estimates f using a cut on Mhh only
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from data.import_functions import loadvars , get_filepaths
from machine_learning.nnoutputfit import find_cut , rocsklearn


if __name__ == '__main__':

    cut_var = 'M0_Mpipi'
    eff = 0.95

    filepath_pi, filepath_k, _ = get_filepaths()
    M_pi, M_k = loadvars(filepath_pi, filepath_k, 't_M0pipi;1', [cut_var], flag_column=False)
    mcut, misid = find_cut(M_k, M_pi, eff,inverse_mode=True)
    plt.figure('Cut on ' + cut_var)
    plt.hist(M_pi, bins=300, range=(5.0,5.4), histtype='step',
                 color='red', label=cut_var + ' for Pions')
    plt.hist(M_k, bins=300, range=(5.0,5.4), histtype='step',
                 color='blue', label=cut_var + ' for Kaons')
    plt.axvline(x=mcut, color='green', label=cut_var + ' cut for '
                + str(eff)+' efficiency')
    plt.draw()
    plt.xlabel(cut_var)
    plt.ylabel('Events per ?????')  # MAKE IT BETTER
    plt.legend()
    plt.savefig('./fig/mcut_'+cut_var+'.pdf')

    rocsklearn(M_pi,M_k, effpnt = eff)


    plt.show()

