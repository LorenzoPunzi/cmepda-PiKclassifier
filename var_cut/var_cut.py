"""

"""
import numpy as np
from matplotlib import pyplot as plt
from utilities.utils import default_rootpaths, default_figpath, \
                            find_cut, roc, stat_error, syst_error
from utilities.import_datasets import loadvars
import warnings


warnings.formatwarning = lambda msg, *args, **kwargs: f'\n{msg}\n'


def var_cut(rootpaths=default_rootpaths(), tree='tree;1', cut_var='M0_Mpipi',
            eff=0.90, inverse_mode=False, specificity_mode=False,
            savefig=False, figpath=''):
    """
    Estimates the fraction of kaons in the mixed sample by performing a cut on
    a specified variable in the template dataset. The cut point is chosen so
    that the probability that a Kaon event has a larger value than that
    (smaller, in the inverse-mode) is equal to the efficiency ("eff") chosen

    :param rootpaths: Tuple containing three .root file paths, for the "background" sample (flag=0), the "signal" sample (flag=1) and the mixed one, in this order.
    :type rootpaths: tuple[str]
    :param tree: Name of the tree from which to load
    :type tree: str
    :param cut_var: Variable to load and test
    :type cut_var: str
    :param eff: Sensitivity required from the test (specificity in specificity mode)
    :type eff: float
    :param inverse_mode: Set to ``True`` if the "signal" events have lower values of the cut variable than the "background" ones
    :type inverse_mode: bool
    :param specificity_mode: Set to ``True`` if the efficiency given is the specificity
    :type specificity_mode: bool
    :param savefig: Draw the figure of the variable distribution for the two species and the cut
    :type savefig: bool
    :param figpath: Path to where to save the figure
    :type figpath: str
    :return: Estimated fraction of Kaons (with uncertainties), parameters of the test algorithm and arrays containing the DNN evaluation of the testing array (divided for species)
    :rtype: tuple[float], tuple[float], tuple[numpy.array[float]]

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
        msg = f'***WARNING*** \ninverse mode was called as {inverse_mode} in var_cut, but the test is not unbiased this way, switching to inverse_mode = {not inverse_mode}'
        warnings.warn(msg, stacklevel=2)
        inverse_mode = not inverse_mode
        cut, misid = find_cut(var_pi, var_k, eff, inverse_mode=inverse_mode,
                              specificity_mode=specificity_mode)

    if savefig:
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

    fraction = ((var_data < cut).sum()/var_data.size - misid)/(eff - misid) \
        if inverse_mode else ((var_data > cut).sum()/var_data.size-misid)/(eff - misid)

    if specificity_mode:
        eff, misid = misid, 1-eff
        fraction = ((var_data < cut).sum()/var_data.size - misid)/(eff - misid) \
            if inverse_mode else ((var_data > cut).sum()/var_data.size-misid)/(eff - misid)

    df_stat = stat_error(fraction, var_data.size, eff, misid)

    df_syst = syst_error(fraction, (var_pi.size, var_k.size), eff, misid)

    fr = (fraction, df_stat, df_syst)

    print(f'{cut_var} cut is {cut} for {eff} efficiency')
    print(f'Misid is {misid} for {eff} efficiency')
    print(f'The estimated fraction of K events is {fraction}')

    '''
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
    '''

    algorithm_parameters = (eff, misid, cut)

    eval_arrays = (var_pi, var_k)

    return fr, algorithm_parameters, eval_arrays


if __name__ == '__main__':

    var_cut(eff=0.9, inverse_mode=True,
            specificity_mode=False, draw_roc=True)
    plt.show()
