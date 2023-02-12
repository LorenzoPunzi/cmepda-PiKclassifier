"""
Module that takes two variables from the MC root files and combines them to
build a new one. The goal of this operation is to have more separated
distributions for the the two species
"""
import uproot
import os
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


def mergevar(tree_pi, tree_k, *vars):
    """
    """
    [v1_pi, v2_pi] = [tree_pi[var].array(library='np') for var in vars]
    [v1_K, v2_K] = [tree_k[var].array(library='np') for var in vars]

    KS_stat_1 = KS(v1_pi, v1_K, alternative='twosided').statistic
    KS_stat_2 = KS(v2_pi, v2_K, alternative='twosided').statistic

    v1lim = [min(min(v1_pi), min(v1_K)), max(max(v1_pi), max(v1_K))]
    v2lim = [min(min(v2_pi), min(v2_K)), max(max(v2_pi), max(v2_K))]

    v1_pi = (v1_pi - v1lim[0])/(v1lim[1]-v1lim[0])
    v2_pi = (v2_pi - v2lim[0])/(v2lim[1]-v2lim[0])
    v1_K = (v1_K - v1lim[0])/(v1lim[1]-v1lim[0])
    v2_K = (v2_K - v2lim[0])/(v2lim[1]-v2lim[0])

    v_pi = [v1_pi, v2_pi]
    v_K = [v1_K, v2_K]

    kolmogorov_stat, m = KS_optimization(
        v_pi, v_K, parlims=[-100., 100.], n_try=1000)

    v_merge_pi = mix_function(v1_pi, v2_pi, m)
    v_merge_K = mix_function(v1_K, v2_K, m)
    print(f'Statistica di KS per la nuova variabile = {kolmogorov_stat}')
    print(f'Stat. di KS per variabili originali: {KS_stat_1}, {KS_stat_2}')
    print(' ')

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.hist(v_merge_pi, 100, color='blue', histtype='step')
    plt.hist(v_merge_K, 100, color='red', histtype='step')
    plt.subplot(2, 1, 2)
    plt.hist(v1_pi, 100, color='blue', histtype='step')
    plt.hist(v1_K, 100, color='red', histtype='step')
    plt.hist(v2_pi, 100, color='blue', histtype='step')
    plt.hist(v2_K, 100, color='red', histtype='step')
    # plt.show()
    plt.savefig('fig/'+vars[0]+'_'+vars[1]+'_merged_'+str(m)+'.pdf')
    plt.close()

    return v_merge_pi, v_merge_K, m


if __name__ == '__main__':
    current_path = os.path.dirname(__file__)
    tree = 't_M0pipi;2'
    filepath = '../root_files'
    filenames = ['tree_B0PiPi_mc.root', 'tree_B0sKK_mc.root']
    files = [os.path.join(
        current_path, filepath, filename) for filename in filenames]
    [tree_pi, tree_k] = [uproot.open(file)[tree] for file in files]

    vars = ['M0_MKK', 'M0_MKpi', 'M0_MpiK', 'M0_Mpipi', 'M0_p', 'M0_pt']
    # Si può sicuramente fare in modo più figo con un parser
    comb1 = np.array([0, 4])
    comb2 = np.array([3, 4])
    comb3 = np.array([1, 2])
    # comb4 = np.array([])
    combinations = [comb1, comb2, comb3]

    # merged_pi = []
    # merged_K = []

    for comb in combinations:
        selected_vars = [vars[comb[0]], vars[comb[1]]]
        print(selected_vars)
        v_merge_pi, v_merge_K, m = mergevar(tree_pi, tree_k, *selected_vars)
        # merged_pi.append(v_merge_pi)
        # merged_K.append(v_merge_K)
        print(m)
        np.savetxt('txt/newvars_pi/'+vars[comb[0]]+'_'
                   + vars[comb[1]]+'_merged_'+str(m)+'.txt', v_merge_pi)
        np.savetxt('txt/newvars_k/'+vars[comb[0]]+'_'
                   + vars[comb[1]]+'_merged_'+str(m)+'.txt', v_merge_K)

    # newvars_pi = np.stack(merged_pi, axis=1)
    # newvars_pi = np.stack(merged_pi, axis=1)
