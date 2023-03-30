"""
Module that takes two variables from the MC root files and combines them to
build a new one. The goal of this operation is to have more separated
distributions for the the two species

"""
import sys
import os
import warnings
import time
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp as KS
from utilities.utils import default_rootpaths
from utilities.exceptions import IncorrectIterableError

warnings.formatwarning = lambda msg, *args, **kwargs: f'\n{msg}\n'

def mix_function(y, x, m):
    """
    Mixing function of the two variables (x,y), with the external parameter m
    """
    return y + m*x


def KS_optimization(v_pi, v_k, parlims=(-20, 20), n_try=1001):
    """
    Performs different attempts of variable mixing, using mix_function(),
    calculating for each case the Kolmogorov-Smirnov statistic.

    :param v_pi: Two-column array containing the variables to be mixed, taken from the Pion template dataset
    :type v_pi: numpy.array
    :param v_k: Two-column array containing the variables to be mixed, taken from the Kaon template dataset
    :type v_k: numpy.array
    :param parlims: Limits of the parameter "m" where to apply the algorithm
    :type parlims: tuple[float]
    :param n_try: Number of attempts of the algorithm
    :type n_try: int
    :return: The maximum value found of the KS statistic and the corresponding value of "m"
    :rtype: float, float

    """
    pars = np.linspace(parlims[0], parlims[1], n_try)
    ks_stats = np.zeros(n_try)
    idx_sel = 0
    max_stat = 0.
    for i in range(len(pars)):
        v_merge_pi = mix_function(v_pi[0], v_pi[1], pars[i])
        v_merge_k = mix_function(v_k[0], v_k[1], pars[i])
        ks_stats[i] = KS(v_merge_pi, v_merge_k,
                         alternative='twosided').statistic
        if i != 0 and ks_stats[i] > max_stat:
            max_stat = ks_stats[i]
            idx_sel = i

    selected_m = round(pars[idx_sel], 3)
    return max_stat, selected_m


def mergevar(filepaths=default_rootpaths(), tree='tree;1',
             vars=('M0_MKpi', 'M0_MpiK'), savefig=False, savetxt=False):
    """
    Function that takes two variables stored in TTrees and mixes them to create
    a new feature, by using the algorithm defined implemented previously in
    this module. This feature is then computed for both the MCs and for data
    and stored in 3 arrays, returned by the function.

    :param filepaths: Three element list or tuple containing .root file paths, of the "background" set (flag=0), the "signal" set (flag=1) and the mixed data set, in this order.
    :type filepaths: list[str] or tuple[str]
    :param tree: Name of the tree from which to load
    :type tree: str
    :param vars: Two element list or tuple of variables that are going to be merged
    :type vars: list[str] or tuple[str]
    :param savefig: If ``True`` the figure of the new variable's distribution is saved
    :type savefig: bool
    :param savetxt: If ``True`` the array containing the new variable's events is saved as .txt file
    :type savetxt: bool
    :return: Three element tuple containing respectively: a three element tuple of numpy arrays of the new variable (MC background, MC signal, data); a two element tuple of value retrieved by KS_optimization() and the KS statistic of the original variables; the parameter "m" which the original variables were merged with
    :rtype: tuple[tuple[numpy.array[float]], tuple[float], float]

    """

    if len(filepaths)>=4:
        msg = f'***WARNING*** \nInput filepaths given are more than three. Using only the first three...\n*************\n'
        warnings.warn(msg, stacklevel=2)
    try:
        if len(filepaths)<3 or not (type(filepaths)==list or type(filepfilepathsaths_in)==tuple):
            raise IncorrectIterableError(filepaths,3) 
    except IncorrectIterableError as err:
        print(err)
        sys.exit()
    
    if len(vars)>=3:
        msg = f'***WARNING*** \nVars to merge given are more than two. Using only the first two...\n*************\n'
        warnings.warn(msg, stacklevel=2)
    try:
        if len(vars)<2 or not (type(vars)==list or type(vars)==tuple):
            raise IncorrectIterableError(vars,2) 
    except IncorrectIterableError as err:
        print(err)
        sys.exit()


    tree_pi, tree_k, tree_data = [uproot.open(file)[tree]
                                  for file in filepaths]

    n_vars = 2  # Merging only couples of variables
    v_pi, v_k, v_data = [0]*n_vars, [0]*n_vars, [0]*n_vars
    for i in range(n_vars):
        var = vars[i]
        v_pi[i] = tree_pi[var].array(library='np')
        v_k[i] = tree_k[var].array(library='np')
        v_data[i] = tree_data[var].array(library='np')

    # The range of the original variables' distribution is normalized, so that
    # an unique range for the parameter "m" can be used for the mixing
    v1lim = [min(v_pi[0], v_k[0]), max(v_pi[0], v_k[0])]
    v2lim = [min(v_pi[1], v_k[1]), max(v_pi[1], v_k[1])]

    v1_pi = (v_pi[0]-v1lim[0])/(v1lim[1]-v1lim[0])
    v2_pi = (v_pi[1]-v2lim[0])/(v2lim[1]-v2lim[0])
    v1_k = (v_k[0]-v1lim[0])/(v1lim[1]-v1lim[0])
    v2_k = (v_k[1]-v2lim[0])/(v2lim[1]-v2lim[0])

    v1_data = (v_data[0]-v1lim[0])/(v1lim[1]-v1lim[0])
    v2_data = (v_data[1]-v1lim[0])/(v1lim[1]-v1lim[0])

    kolmogorov_stat, m = KS_optimization(
        (v1_pi, v2_pi), (v1_k, v2_k), parlims=(-20., 20.), n_try=1001)

    ks_stats = []
    ks_stats.append(kolmogorov_stat)
    ks_stats.append(KS(v1_pi, v1_k, alternative='twosided').statistic)
    ks_stats.append(KS(v2_pi, v2_k, alternative='twosided').statistic)
    ks_stats = tuple(ks_stats)

    v_merge_pi = mix_function(v1_pi, v2_pi, m)
    v_merge_k = mix_function(v1_k, v2_k, m)
    v_merge_data = mix_function(v1_data, v2_data, m)

    if savetxt:
        # The MC variables are saved in the same file, so an equal length of
        # the arrays is required. This could have the meaning of a formal check
        # because in the best situation this request is already satisfied
        if len(v_merge_pi) != len(v_merge_k):
            length = min(len(v_merge_pi), len(v_merge_k))
            mc_array = np.stack(
                (v_merge_pi[:length], v_merge_k[:length]), axis=1)
        else:
            mc_array = np.stack((v_merge_pi, v_merge_k), axis=1)
        np.savetxt(f'../data/txt/newvars/{vars[0]}_{vars[1]}_merged__mc.txt',
                   mc_array, delimiter='  ', header=f'mc_pi,  mc_k,  m={m}')
        np.savetxt(f'../data/txt/newvars/{vars[0]}_{vars[1]}_merged__data.txt',
                   v_merge_data, delimiter='  ', header=f'data  m={m}')

    if savefig:
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.hist(v_merge_pi, 100, color='blue', histtype='step')
        plt.hist(v_merge_k, 100, color='red', histtype='step')
        plt.subplot(2, 1, 2)
        plt.hist(v1_pi, 100, color='blue', histtype='step')
        plt.hist(v1_k, 100, color='red', histtype='step')
        plt.hist(v2_pi, 100, color='blue', histtype='step')
        plt.hist(v2_k, 100, color='red', histtype='step')
        plt.savefig('fig/'+vars[0]+'_'+vars[1]+'_merged_'+str(m)+'.pdf')
        plt.close()

    merged_arr = (v_merge_pi, v_merge_k, v_merge_data)

    return merged_arr, ks_stats, m


if __name__ == '__main__':
    t0 = time.time()
    current_path = os.path.dirname(__file__)
    tree = 't_M0pipi;1'

    filepaths = default_rootpaths()

    combinations = (('M0_MKpi', 'M0_MpiK'), )

    # ('MKK', 'p'), ('Mpipi', 'p'), ('MKK', 'MpiK'), ('Mpipi', 'MKpi')]

    # vars = ['M0_MKK', 'M0_MKpi', 'M0_MpiK', 'M0_Mpipi', 'M0_p', 'M0_pt']
    # vars = ['M0_MKK', 'M0_MKpi', 'M0_MpiK', 'M0_Mpipi',
    #         'M0_p', 'h1_thetaC0', 'h1_thetaC1', 'h1_thetaC2']

    stats = []
    str_combinations = []

    for comb in combinations:
        print(comb)
        new_arrays, stats_new, m = mergevar(filepaths, tree, comb)
        stats.append(stats_new)
        string_combination = comb[0]+'_'+comb[1]+'_merged'
        str_combinations.append(string_combination+'  ||  ')

    arr_stats = np.array(stats).reshape(len(combinations), 3)
    # np.savetxt('txt/output_KS_merged_firstset.txt', arr_stats,
    #            delimiter='    ', header=''.join(str_combinations))

    t1 = time.time()

    print(f"Tempo totale = {t1-t0}")
