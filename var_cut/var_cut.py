"""
Estimates f using a cut on Mhh only
"""
import numpy as np
from matplotlib import pyplot as plt
from utilities.utils import default_rootpaths, find_cut, roc
from utilities.import_datasets import loadvars


def var_cut(rootpaths=default_rootpaths(), tree='tree;1', cut_var='M0_Mpipi',
            eff=0.95, inverse_mode=False, specificity_mode=False,
            draw_roc=False, draw_fig=True):
    """
    """
    rootpath_pi, rootpath_k, rootpath_data = rootpaths
    M_pi, M_k = loadvars(rootpath_pi, rootpath_k, tree,
                         [cut_var], flag_column=False)
    M_data, _ = loadvars(rootpath_data, rootpath_data, tree,
                         [cut_var], flag_column=False)

    m_cut, misid = find_cut(M_pi, M_k, eff, inverse_mode=inverse_mode,
                            specificity_mode=specificity_mode)

    if draw_fig:
        nbins = 300
        range = (5.0, 5.6)
        plt.figure('Cut on ' + cut_var)
        plt.hist(M_pi, nbins, range=range, histtype='step',
                 color='red', label=cut_var + ' for Pions')  # !!! (range)
        plt.hist(M_k, nbins, range=range, histtype='step',
                 color='blue', label=cut_var + ' for Kaons')
        plt.axvline(x=m_cut, color='green', label=cut_var + ' Cut for '
                    + str(eff)+' efficiency')
        plt.draw()
        plt.xlabel(cut_var)
        plt.ylabel(f'Events per {range[1]-range[0]/nbins} [{cut_var}]')
        plt.legend()
        plt.savefig('./fig/cut_'+cut_var+'.pdf')

    rocx, rocy, auc = roc(M_pi, M_k, eff=eff, inverse_mode=inverse_mode,
                          makefig=draw_roc)

    print(f'{cut_var} cut is {m_cut} for {eff} efficiency')
    print(f'Misid. is {misid} for {eff} efficiency')

    f = ((M_data < m_cut).sum()/M_data.size-misid)/(eff-misid)
    print(f'The estimated fraction of K events is {f}')

    return rocx, rocy, auc


if __name__ == '__main__':

    var_cut(eff=0.95, inverse_mode=True, specificity_mode=False, draw_roc=True)

    plt.show()
