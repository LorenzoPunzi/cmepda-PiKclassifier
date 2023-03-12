"""
Estimates f using a cut on Mhh only
"""
import numpy as np
from matplotlib import pyplot as plt
from utilities.utils import default_rootpaths, find_cut, roc, default_figpath
from utilities.import_datasets import loadvars


def var_cut(rootpaths=default_rootpaths(), tree='t_M0pipi;1', cut_var='M0_Mpipi',
            eff=0.95, inverse_mode=False, specificity_mode=False,
            draw_roc=False, draw_fig=False, figpath='', stat_split=0):
    """
    Estimates the fraction of kaons in the mixed sample by performing a cut on
    the distribution of the templates. The cut point is chosen so that the
    probability that a Kaon event has a larger value than that (smaller, in the
    inverse-mode) is equal to the efficiency chosen (parameter "eff")

    Parameters:
        rootpaths: list or tuple

        tree: string

        cut_var: string

        eff: float

        inverse_mode: bool

        specificity_mode: bool

        draw_roc, draw_fig: bool

        figpath: string

        stat_split: int

    """
    rootpath_pi, rootpath_k, rootpath_data = rootpaths
    var_pi, var_k = loadvars(rootpath_pi, rootpath_k, tree,
                             [cut_var], flag_column=False)
    var_data, _ = loadvars(rootpath_data, rootpath_data, tree,
                           [cut_var], flag_column=False)

    cut, misid = find_cut(var_pi, var_k, eff, inverse_mode=inverse_mode,
                          specificity_mode=specificity_mode)

    if (specificity_mode is not True and misid > eff) or \
            (specificity_mode is True and 1 - eff > misid):
        inverse_mode = not inverse_mode
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
        plt.savefig(default_figpath('cut_'+cut_var)) \
            if figpath == '' else plt.savefig(figpath+'/cut_'+cut_var+'.png')

    rocx, rocy, auc = roc(var_pi, var_k, eff=eff, inverse_mode=inverse_mode,
                          makefig=draw_roc, name=default_figpath('ROC_'+cut_var+'_cut')) \
        if figpath == '' else roc(var_pi, var_k, eff=eff, inverse_mode=inverse_mode,
                                  makefig=draw_roc, name=figpath+'/ROC_'+cut_var + '_cut.png')

    print(f'{cut_var} cut is {cut} for {eff} efficiency')
    print(f'Misid. is {misid} for {eff} efficiency')

    fraction = ((var_data < cut).sum()/var_data.size - misid)/(eff - misid) \
        if inverse_mode else ((var_data > cut).sum()/var_data.size-misid)/(eff - misid)
    print(f'The estimated fraction of K events is {fraction}')

    fr = (fraction,)

    if specificity_mode:
        eff, misid = misid, 1-eff
        ((var_data < cut).sum()/var_data.size - misid)/(eff - misid) \
            if inverse_mode else ((var_data > cut).sum()/var_data.size-misid)/(eff - misid)

    if stat_split:
        subdata = np.array_split(var_data, stat_split)
        fractions = [((dat < cut).sum()/dat.size-misid)/(eff-misid)
                     for dat in subdata] if inverse_mode \
            else [((dat > cut).sum()/dat.size-misid)/(eff-misid)
                  for dat in subdata]
        plt.figure('Fraction distribution for '+cut_var+' cut')
        plt.axvline(x=fraction, color='green')
        plt.hist(fractions, bins=7, histtype='step')
        plt.savefig(default_figpath('cut_'+cut_var+'_distrib')) if figpath == ''\
            else plt.savefig(figpath+'/cut_'+cut_var+'_distrib.png')
        stat_err = np.sqrt(np.var(fractions, ddof=1))
        # print(f"Mean = {np.mean(fractions, dtype=np.float64)}")
        # print(f"Sqrt_var = {stat_err}")
        fr = fr + (stat_err,)

    roc_info = tuple([rocx, rocy, auc])

    return fr, cut, misid, roc_info


if __name__ == '__main__':

    var_cut(eff=0.9, inverse_mode=True,
            specificity_mode=False, draw_roc=True, stat_split=5)
    plt.show()
