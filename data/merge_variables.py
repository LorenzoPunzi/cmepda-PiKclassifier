"""
Module that takes two variables from the MC root files and combines them to
build a new one. The goal of this operation is to have more separated
distributions for the the two species
"""
import uproot
import os
import time
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp as KS


def mix_function(y, x, m):
    return y + m*x


def KS_optimization(v_pi, v_K, parlims=[-10, 10], n_try=1000):
    """
    """
    if not len(parlims) == 2:
        pass
    pars = np.linspace(parlims[0], parlims[1], n_try)
    ks_stats = np.zeros(n_try)
    idx_sel = 0.
    max_stat = 0.
    for i in range(len(pars)):
        v_merge_pi = mix_function(v_pi[0], v_pi[1], pars[i])
        v_merge_K = mix_function(v_K[0], v_K[1], pars[i])
        ks_stats[i] = KS(v_merge_pi, v_merge_K,
                         alternative='twosided').statistic
        if i != 0:
            if ks_stats[i] > max_stat:
                max_stat = ks_stats[i]
                idx_sel = i
    return max_stat, round(pars[idx_sel], 3)


def mergevar(tree_files, tree, vars):
    """
    Function that takes two variables stored in TTrees and mixes them to create
    a new feature. This feature is then computed for both the MCs and for data
    and stored in a 3-column array, returned by the function.
    In this version, the function relies on the two functions implemented
    previously in this module

    Parameters
    ----------
    trees: string
        Trees that store the original variables. A list of 3 arguments has to be
        passed, containing the trees for Pi and K MCs and data (in this order)
    vars: string
        Couple of variables that have to be mixed by he function. In this
        version, it must be alist of two arguments
    """
    if not len(tree_files) == 3:
        print("Errore nella lunghezza delle liste passate a \'mergevar\'")
    if not len(vars) == 2:
        print("Errore nella lunghezza delle liste passate a \'mergevar\'")

    tree_pi, tree_k, tree_data = [
        uproot.open(file)[tree] for file in tree_files]

    n_vars = 2  # Merging only couples of variables
    v_pi, v_k, v_data = [0]*n_vars, [0]*n_vars, [0]*n_vars
    for i in range(n_vars):
        var = vars[i]
        v_pi[i] = tree_pi[var].array(library='np')
        v_k[i] = tree_k[var].array(library='np')
        v_data[i] = tree_data[var].array(library='np')

    v1lim = [min(min(v_pi[0]), min(v_k[0])), max(max(v_pi[0]), max(v_k[0]))]
    v2lim = [min(min(v_pi[1]), min(v_k[1])), max(max(v_pi[1]), max(v_k[1]))]

    v1_pi = (v_pi[0]-v1lim[0])/(v1lim[1]-v1lim[0])
    v2_pi = (v_pi[1]-v2lim[0])/(v2lim[1]-v2lim[0])
    v1_k = (v_k[0]-v1lim[0])/(v1lim[1]-v1lim[0])
    v2_k = (v_k[1]-v2lim[0])/(v2lim[1]-v2lim[0])

    v1_data = (v_data[0]-v1lim[0])/(v1lim[1]-v1lim[0])
    v2_data = (v_data[1]-v1lim[0])/(v1lim[1]-v1lim[0])

    kolmogorov_stat, m = KS_optimization(
        [v1_pi, v2_pi], [v1_k, v2_k], parlims=[-100., 100.], n_try=1000)

    KS_stats = []
    KS_stats.append(kolmogorov_stat)
    KS_stats.append(KS(v1_pi, v1_k, alternative='twosided').statistic)
    KS_stats.append(KS(v2_pi, v2_k, alternative='twosided').statistic)

    v_merge_pi = mix_function(v1_pi, v2_pi, m)
    v_merge_k = mix_function(v1_k, v2_k, m)
    v_merge_data = mix_function(v1_data, v2_data, m)

    # Controllo aggiuntivo nel caso gli array MC avessero lunghezza diversa.
    # In generale la cosa ottimale è che i due datasets abbiano lo stesso numero
    # di eventi
    if not len(v_merge_pi) == len(v_merge_k):
        length = min(len(v_merge_pi), len(v_merge_k))

    merged_arrays = [v_merge_pi[:length], v_merge_k[:length], v_merge_data]

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.hist(v_merge_pi, 100, color='blue', histtype='step')
    plt.hist(v_merge_k, 100, color='red', histtype='step')
    plt.subplot(2, 1, 2)
    plt.hist(v1_pi, 100, color='blue', histtype='step')
    plt.hist(v1_k, 100, color='red', histtype='step')
    plt.hist(v2_pi, 100, color='blue', histtype='step')
    plt.hist(v2_k, 100, color='red', histtype='step')
    # plt.show()
    plt.savefig('fig/'+vars[0]+'_'+vars[1]+'_merged_'+str(m)+'.pdf')
    plt.close()

    return merged_arrays, m, KS_stats


if __name__ == '__main__':
    t0 = time.time()
    current_path = os.path.dirname(__file__)
    tree = 't_M0pipi;1'
    filepath = '../root_files'
    filenames = ['tree_B0PiPi.root',
                 'tree_B0sKK.root', 'tree_Bhh_data.root']
    files = [os.path.join(
        current_path, filepath, filename) for filename in filenames]

    vars = ['M0_MKK', 'M0_MKpi', 'M0_MpiK', 'M0_Mpipi', 'M0_p', 'M0_pt']
    # vars = ['M0_MKK', 'M0_MKpi', 'M0_MpiK', 'M0_Mpipi',
    #         'M0_p', 'h1_thetaC0', 'h1_thetaC1', 'h1_thetaC2']
    # Si può sicuramente fare in modo più figo con un parser

    combinations = []

    combinations.append(np.array([1, 2]))  # MKpi - MpiK
    combinations.append(np.array([0, 4]))  # MKK - p
    combinations.append(np.array([3, 4]))  # Mpipi - p
    combinations.append(np.array([0, 2]))  # MKK - MpiK
    combinations.append(np.array([3, 1]))  # Mpipi - MKpi
    '''
    combinations.append(np.array([0, 5]))  # MKK - thetaC0
    combinations.append(np.array([0, 6]))  # MKK - thetaC1
    combinations.append(np.array([0, 7]))  # MKK - thetaC2
    '''
    stats = []
    str_combinations = []

    for comb in combinations:
        selected_vars = [vars[comb[0]], vars[comb[1]]]
        print(selected_vars)
        new_arrays, m, stats_new = mergevar(files, tree, selected_vars)
        stats.append(stats_new)
        # risolvere il problema per cui non si può fare lo stack di due array di lunghezza diversa
        mc_array = np.stack((new_arrays[0], new_arrays[1]), axis=1)
        string_combination = vars[comb[0]]+'_'+vars[comb[1]]+'_merged'
        np.savetxt('txt/newvars/'+string_combination+'__mc.txt', mc_array,
                   delimiter='  ', header='mc_pi,  mc_k,    m = '+str(m))
        np.savetxt('txt/newvars/'+string_combination+'__data.txt',
                   np.array(new_arrays[2]), delimiter='  ', header='m = '+str(m))
        str_combinations.append(string_combination+'  ||  ')

    arr_stats = np.array(stats).reshape(len(combinations), 3)
    np.savetxt('txt/output_KS_merged_firstset.txt', arr_stats,
               delimiter='    ', header=''.join(str_combinations))

    t1 = time.time()

    print(f"Tempo totale = {t1-t0}")
