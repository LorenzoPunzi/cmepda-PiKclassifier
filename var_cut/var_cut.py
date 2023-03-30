"""
"""
import sys
import warnings
import numpy as np
from matplotlib import pyplot as plt
from utilities.utils import default_rootpaths, default_figpath, \
                            find_cut, stat_error, syst_error
from utilities.import_datasets import loadvars
from utilities.exceptions import IncorrectEfficiencyError

warnings.formatwarning = lambda msg, *args, **kwargs: f'\n{msg}\n'


def var_cut(rootpaths=default_rootpaths(), tree='tree;1', cut_var='M0_Mpipi',
            eff=0.90, error_optimization=True, inverse_mode=False,
            specificity_mode=False, savefig=False, figpath=''):
    """
    Estimates the signal fraction in the data set by performing a cut on a
    specified variable in the MC sets.

    :param rootpaths: Tuple containing three .root file paths, for the "background" set (flag=0), the "signal" set (flag=1) and the data one, in this order.
    :type rootpaths: tuple[str]
    :param tree: Name of the tree from which to load the events.
    :type tree: str
    :param cut_var: Variable to load and test.
    :type cut_var: str
    :param eff: Sensitivity required from the test (specificity in specificity mode).
    :type eff: float
    :param error_optimization: Performs error optimization instead of using a fixed efficiency value.
    :type error_optimization: bool
    :param inverse_mode: Set to ``True`` if the "signal" events have lower values of the cut variable than the "background" ones.
    :type inverse_mode: bool
    :param specificity_mode: Set to ``True`` if the efficiency given is the specificity.
    :type specificity_mode: bool
    :param savefig: Draw the figure of the variable distribution for the two MC sets and the cut performed.
    :type savefig: bool
    :param figpath: Path to where to save the figure.
    :type figpath: str
    :return: Estimated fraction of signal (with uncertainties), parameters of the cut algorithm and MC sets of the variable used.
    :rtype: tuple[float], tuple[float], tuple[numpy.array[float]]

    """

    var_pi, var_k = loadvars(
        rootpaths[0], rootpaths[1], tree, [cut_var], flag_column=False)
    var_data, _ = loadvars(
        rootpaths[2], rootpaths[2], tree, [cut_var], flag_column=False)

    used_eff = 0
    df_opt = -99999

    try:
        if (eff <= 0) or (eff >= 1):
            raise IncorrectEfficiencyError(eff)
    except IncorrectEfficiencyError as err:
        print(err)
        sys.exit()

    if error_optimization is True:  # Enables FOM maximization
        efficiencies = np.linspace(0.25, 0.999, 200)
        for tmp_eff in efficiencies:
            tmp_cut, tmp_misid = find_cut(var_pi, var_k, tmp_eff,
                                          inverse_mode=inverse_mode,
                                          specificity_mode=False)

            tmp_frac = ((var_data < tmp_cut).sum()/var_data.size-tmp_misid)/(tmp_eff - tmp_misid) \
                if inverse_mode else ((var_data > tmp_cut).sum()/var_data.size-tmp_misid)/(tmp_eff - tmp_misid)

            tmp_dfopt = -np.sqrt(
                stat_error(tmp_frac, var_data.size, tmp_eff, tmp_misid)**2
                + syst_error(tmp_frac, (var_pi.size, var_k.size), tmp_eff, tmp_misid)**2)

            if tmp_dfopt >= df_opt:
                df_opt = tmp_dfopt
                used_eff = tmp_eff
    else:
        used_eff = eff

    cut, misid = find_cut(var_pi, var_k, used_eff, inverse_mode=inverse_mode,
                          specificity_mode=specificity_mode)

    if (specificity_mode is not True and misid > used_eff) or \
       (specificity_mode is True and 1 - used_eff > misid):
        msg = f'***WARNING***\nInverse mode was called as {inverse_mode} in \
var_cut, but the test is not unbiased this way; switching to \
inverse_mode = {not inverse_mode} \n*************\n'
        warnings.warn(msg, stacklevel=2)
        inverse_mode = not inverse_mode
        cut, misid = find_cut(var_pi, var_k, used_eff, inverse_mode=inverse_mode,
                              specificity_mode=specificity_mode)

    if savefig:
        nbins = 300
        range = (min(min(var_pi), min(var_k)), max(max(var_pi), max(var_k)))
        plt.figure('Cut on ' + cut_var)
        plt.hist(var_pi, nbins, range=range, histtype='step',
                 color='red', label=f'{cut_var} for Pions')
        plt.hist(var_k, nbins, range=range, histtype='step',
                 color='blue', label=f'{cut_var} for Kaons')
        plt.axvline(x=cut, color='green',
                    label=f'{cut_var} Cut for {used_eff} efficiency')
        plt.title(f'Varibale Cut on {cut_var}')
        plt.xlabel(cut_var)
        plt.ylabel(
            f'Events per {round((range[1]-range[0])/nbins, 4)} [{cut_var}]')
        plt.legend()
        plt.draw()
        plt.savefig(default_figpath(f'cut_{cut_var}')) \
            if figpath == '' else plt.savefig(f'{figpath}/cut_{cut_var}.png')

    fraction = ((var_data < cut).sum()/var_data.size - misid)/(used_eff - misid) \
        if inverse_mode else ((var_data > cut).sum()/var_data.size-misid)/(used_eff - misid)

    if specificity_mode:
        used_eff, misid = misid, 1-used_eff
        fraction = ((var_data < cut).sum()/var_data.size - misid)/(used_eff - misid) \
            if inverse_mode else ((var_data > cut).sum()/var_data.size-misid)/(used_eff - misid)

    df_stat = stat_error(fraction, var_data.size, used_eff, misid)

    df_syst = syst_error(fraction, (var_pi.size, var_k.size), used_eff, misid)

    fr = (fraction, df_stat, df_syst)

    print(f'{cut_var} cut is {cut} for {used_eff} efficiency')
    print(
        f'Misid is {misid} +- {np.sqrt(misid*(1-misid)/var_pi.size)} for {used_eff} efficiency')
    print(
        f'The estimated fraction of K events is {fraction} +- {df_stat} (stat) +- {df_syst} (syst)')

    cut_parameters = (used_eff, misid, cut)

    eval_arrays = (var_pi, var_k)

    return fr, cut_parameters, eval_arrays


if __name__ == '__main__':

    var_cut(eff=0.9, inverse_mode=True,
            specificity_mode=False, draw_roc=True)
    plt.show()
