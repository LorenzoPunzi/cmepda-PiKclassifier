"""
Module that takes two variables from the MC root files and combines them to
build a new one. The goal of this operation is to have more separated
distributions for the the two species
"""
import sys
import os
import time
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp as KS
from utilities.utils import default_rootpaths


def mix_function(y, x, m):
    """
    Mixing function of the two variables (x,y), with the external parameter m
    """
    return y + m*x


def KS_optimization(v_pi, v_k, parlims=(-20, 20), n_try=1001):
    """
    Performs different attempts of variable mixing, using mix_function(),
    calculating for each case the Kolmogorov-Smirnov statistic.

    Parameters:
        v_pi : numpy array
            Two-column array containing the variables to be mixed, taken
            from the Pion MC dataset
        v_k : numpy array
            Two-column array containing the variables to be mixed, taken
            from the Kaon MC dataset
        parlims : tuple
            Limits of the parameter \'m\' where to apply the algorithm
        n_try : int
            Number of attempts of the algorithm

    Returns:
        max_stat : float
            Maximum value of the Kolmogorov-Smirnov statistic found
        selected_m : float
            Value of \'m\' corresponding to the maximum KS statistic value
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


def mergevar(filepaths, tree, vars, savefig=False, savetxt=False):
    """
    Function that takes two variables stored in TTrees and mixes them to create
    a new feature, by using the algorithm defined implemented previously in
    this module. This feature is then computed for both the MCs and for data
    and stored in 3 arrays, returned by the function.

    Parameters:
        filepaths : tuple
            Paths (with name) of the root files where the variables are stored.
            It must include both the MC files for Pi and K and the mixed data
            file, in this order
        tree : string
            Tree containing the datasets (is the same for both the files)
        vars : tuple
            Couple of variables that are going to be merged
        savefig, savetxt : bool
            Flags that allow to save the figure of the distributions of the new
            variable and to save on .txt files the new datasets
            Default: False

    Returns:
        merged_arr : tuple
            Tuple containing the three sets of the new variable
        ks_stats : tuple
            Value of the Kolmogorov-Smirnov retrieved by KS_optimization() and
            values of the KS statistic for the original values
        m : float
            Value of \'m\' which the original variables were merged with
    """

    if len(filepaths) != 3:
        print("Errore nella lunghezza delle liste passate a \'mergevar\'")
        sys.exit()
    if len(vars) != 2:
        print("Errore nella lunghezza delle liste passate a \'mergevar\'")
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
