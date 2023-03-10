"""
Estimates f using a cut on Mhh only
"""
import numpy as np
from matplotlib import pyplot as plt
from utilities.utils import default_rootpaths, find_cut, roc, default_figpath
from utilities.import_datasets import loadvars


def var_cut(rootpaths=default_rootpaths(), tree='tree;1', cut_var='M0_Mpipi',
            eff=0.95, inverse_mode=False, specificity_mode=False,
            draw_roc=False, draw_fig=False, figpath='', stat_split=0):
    """
    """
    rootpath_pi, rootpath_k, rootpath_data = rootpaths
    var_pi, var_k = loadvars(rootpath_pi, rootpath_k, tree,
                             [cut_var], flag_column=False)
    var_data, _ = loadvars(rootpath_data, rootpath_data, tree,
                           [cut_var], flag_column=False)

    cut, misid = find_cut(var_pi, var_k, eff, inverse_mode=inverse_mode,
                          specificity_mode=specificity_mode)

    if draw_fig:
        nbins = 300
        range = (5.0, 5.6)
        plt.figure('Cut on ' + cut_var)
        plt.hist(var_pi, nbins, range=range, histtype='step',
                 color='red', label=cut_var + ' for Pions')  # !!! (range)
        plt.hist(var_k, nbins, range=range, histtype='step',
                 color='blue', label=cut_var + ' for Kaons')
        plt.axvline(x=cut, color='green', label=cut_var + ' Cut for '
                    + str(eff)+' efficiency')
        plt.draw()
        plt.xlabel(cut_var)
        plt.ylabel(f'Events per {range[1]-range[0]/nbins} [{cut_var}]')
        plt.legend()
        if figpath == '':
            plt.savefig(default_figpath('cut_'+cut_var))
        else:
            plt.savefig(figpath+'/cut_'+cut_var+'.png')

    rocx, rocy, auc = roc(var_pi, var_k, eff=eff,
                          inverse_mode=inverse_mode, makefig=draw_roc)

    print(f'{cut_var} cut is {cut} for {eff} efficiency')
    print(f'Misid. is {misid} for {eff} efficiency')

    f = ((var_data < cut).sum()/var_data.size - misid)/(eff - misid) \
        if inverse_mode else ((var_data > cut).sum()/var_data.size-misid)/(eff - misid)
    print(f'The estimated fraction of K events is {f}')

    if stat_split:
        subdata = np.split(var_data, stat_split)
        fractions = [((dat < cut).sum()/dat.size-misid)/(eff-misid) for dat in subdata] if inverse_mode \
            else [((dat > cut).sum()/dat.size-misid)/(eff-misid) for dat in subdata]
        plt.figure('Fraction distribution for '+cut_var+' cut')
        plt.hist(fractions, bins=20, histtype='step')
        plt.savefig(default_figpath('fractionsvarcut'))

    return f, cut, misid, rocx, rocy, auc


if __name__ == '__main__':

    var_cut(eff=0.95, inverse_mode=True, specificity_mode=False, draw_roc=True)

    plt.show()
