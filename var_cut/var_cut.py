"""
Estimates f using a cut on Mhh only
"""
import numpy as np
from matplotlib import pyplot as plt
from utilities.utils import default_rootpaths, find_cut, roc, default_figpath
from utilities.import_datasets import loadvars
import warnings


warnings.formatwarning = lambda msg, *args, **kwargs: f'\n{msg}\n'


def var_cut(rootpaths=default_rootpaths(), tree='tree;1', cut_var='M0_Mpipi',
            eff=0.95, inverse_mode=False, specificity_mode=False,
            draw_roc=False, draw_fig=False, figpath='', stat_split=0):
    """
    Estimates the fraction of kaons in the mixed sample by performing a cut on
    the distribution of the templates. The cut point is chosen so that the
    probability that a Kaon event has a larger value than that (smaller, in the
    inverse-mode) is equal to the efficiency chosen (parameter "eff")

    :param rootpaths: Three rootpaths where to search for the root files. The first is the rootpath for the "negative" species, the second for the "positive" species, the third  for the data to be evaluated
    :type rootpaths: tuple[str]
    :param tree: Tree from which to load
    :type tree: str
    :param cut_var: Variable to load and test
    :type cut_var: str
    :param eff: Sensitivity required from the test (specificity in specificity mode)
    :type eff: float
    :param inverse_mode: To activate if the "positive" events have lower values of the cut variable
    :type inverse_mode: bool
    :param specificity_mode: To activate if the efficiency given is the specificity
    :type specificity_mode: bool
    :param draw_roc: Draw the roc function of the test
    :type draw_roc: bool
    :param draw_fig: Draw the figure of the variable distribution for the two species and the cut
    :type draw_fig: bool
    :param figpath: Path to where to save the figure
    :type figpath: str
    :param stat_split: How many parts to split the dataset in, to study the distribution of the fraction estimated with this test
    :type stat_split: int

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
        msg = f'***WARNING*** \ninverse mode was called as {inverse_mode} in \
              var_cut, but the test is not unbiased this way, switching \
              to inverse_mode = {not inverse_mode}'
        warnings.warn(msg, stacklevel=2)
        inverse_mode = not inverse_mode
        cut, misid = find_cut(var_pi, var_k, eff, inverse_mode=inverse_mode,
                              specificity_mode=specificity_mode)

    if draw_fig:
        nbins = 300
        range = (min(min(var_pi), min(var_k)), max(max(var_pi), max(var_k)))
        plt.figure('Cut on ' + cut_var)
        plt.hist(var_pi, nbins, range=range, histtype='step',
                 color='red', label=cut_var + ' for Pions')
        plt.hist(var_k, nbins, range=range, histtype='step',
                 color='blue', label=cut_var + ' for Kaons')
        plt.axvline(x=cut, color='green', label=cut_var + ' Cut for '
                    + str(eff)+' efficiency')
        plt.draw()
        plt.xlabel(cut_var)
        plt.ylabel(f'Events per {range[1]-range[0]/nbins} {cut_var}')
        plt.legend()
        plt.savefig(default_figpath('cut_'+cut_var)) \
            if figpath == '' else plt.savefig(figpath+'/cut_'+cut_var+'.png')

    rocx, rocy, auc = roc(var_pi, var_k, eff=eff, inverse_mode=inverse_mode,
                          makefig=draw_roc,
                          name=default_figpath(f'ROC_{cut_var}_cut')) \
        if figpath == '' else roc(var_pi, var_k, eff=eff,
                                  inverse_mode=inverse_mode, makefig=draw_roc,
                                  name=f'{figpath}/ROC_{cut_var}_cut.png')

    fraction = ((var_data < cut).sum()/var_data.size - misid)/(eff - misid) \
        if inverse_mode else ((var_data > cut).sum()/var_data.size-misid)/(eff - misid)

    if specificity_mode:
        eff, misid = misid, 1-eff
        fraction = ((var_data < cut).sum()/var_data.size - misid)/(eff - misid) \
            if inverse_mode else ((var_data > cut).sum()/var_data.size-misid)/(eff - misid)

    fr = (fraction,)

    print(f'{cut_var} cut is {cut} for {eff} efficiency')
    print(f'Misid is {misid} for {eff} efficiency')
    print(f'The estimated fraction of K events is {fraction}')

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

    return fr, misid, cut, roc_info


if __name__ == '__main__':

    var_cut(eff=0.9, inverse_mode=True,
            specificity_mode=False, draw_roc=True, stat_split=5)
    plt.show()
