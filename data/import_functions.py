"""
Module containing functions that import the dataset from the TTrees and store
the variables of interest in a numpy array for ML treatment
"""

import os
import time
import ROOT
import uproot
import numpy as np
import matplotlib.pyplot as plt


def loadvars(file_pi, file_k, tree, vars):  # FUNDAMENTAL
    """
    Returns numpy arrays containing the requested variables, stored originally
    in MC root files. The LAST column of the output array contains a flag (0
    for Pions, 1 for Kaons)

    Parameters
    ----------
    file_pi: string
        MC file of B0->PiPi process
    file_k: string
        MC file of B0s->KK process
    tree: string
        Tree containing the dataset (is the same for both the files)
    *vars: string
        Tuple containing names of the variables requested
    """

    # In generale la sintassi è un po' da migliorare/snellire. Attenzione però
    # che loadvars è chiamato da diverse altre funzioni che potrebbero quindi
    # non più funzionare

    if (len(vars) == 0):
        # We should put an error here
        print("No variables passed to loadvars function!")
        pass

    tree_pi, tree_k = uproot.open(file_pi)[tree], uproot.open(file_k)[tree]

    flag_pi = np.zeros(tree_pi.num_entries)
    flag_k = np.ones(tree_k.num_entries)

    list_pi, list_k = [], []

    for variable in vars:
        list_pi.append(tree_pi[variable].array(library='np'))
        list_k.append(tree_k[variable].array(library='np'))

    list_pi.append(flag_pi)
    list_k.append(flag_k)

    v_pi, v_k = np.stack(list_pi, axis=1), np.stack(list_k, axis=1)

    return v_pi, v_k


def array_generator(filepaths, tree, vars, *mixed_vars, Ntrain=50000, Ndata=10000,
                    for_training=True, for_testing=True, mixing=False):
    """
    Generates arrays for ML treatment (training and testing) and saves them in
    .txt files

    Parameters
    ----------
    filepaths: list of strings
        Paths of tree files that are used. By default, it must contain the MC
        file for Pi and K and the file of mixed data (in this order)
    tree: string
        Name of the TTree in the files
    *vars: string
        Names of variables included in the analysis
    Ntrain, Ndata: int
        Number of total events requested for training and for testing
    for_training, for_testing: bool
        Option that selects which dataset has to be created
    """
    if for_training & for_testing:
        filepath_pi, filepath_k, filepath_data = filepaths
    elif len(filepaths) == 2 & for_training & (for_testing is False):
        filepath_pi, filepath_k = filepaths
    elif len(filepaths) == 1 & for_testing & (for_training is False):
        filepath_data = filepaths[0]
    else:
        print('Errore nell istanziamento di array_generator()')
        pass

    if mixed_vars & len(*mixed_vars) != 0:
        pass
    # Da capire cosa non va se una delle due flag for_XX è messa FALSE

    train_array, test_array = np.zeros(len(vars)), np.zeros(len(vars))

    if for_training:
        if mixing:
            v1, v2 = loadvars(filepath_pi, filepath_k, tree, vars)
            v_mc_pi, v_mc_pi = [], []
            v_mc_pi.append(v1)
            v_mc_pi.append(v2)
            for mv in mixed_vars:
                v_a, v_b, m, KS_stat = mergevar([filepath_pi, filepath_k], mv)
                v_mc_pi.append(v_a)
                v_mc_pi.append(v_b)
            v_mc_pi, v_mc_k = np.array(v_mc_pi), np.array(v_mc_k)
        else:
            v_mc_pi, v_mc_k = loadvars(filepath_pi, filepath_k, tree, vars)
        train_array = np.concatenate(
            (v_mc_pi[:int(Ntrain/2), :], v_mc_k[:int(Ntrain/2), :]), axis=0)
        np.random.shuffle(train_array)
        print(train_array.shape)
        # np.savetxt("txt/training_array.txt", train_array)

    if for_testing:
        if mixing & for_training:
            v0, v0 = loadvars(filepath_data, filepath_data, tree, vars)
            v_data = []
            for mv in mixed_vars:
                v_a, v_b, m, KS_stat = mergevar([filepath_pi, filepath_k], mv)

        test_array = v_data[:Ndata, :]
        # np.savetxt("txt/data_array_prova.txt", v_testing)

    if mixed_vars:
        # Chiamare qui la funzione include_merged_values
        a = []

    return train_array, test_array


def include_merged_values(train_array, test_array, mixed variables):
    """
    """


if __name__ == '__main__':
    t1 = time.time()

    current_path = os.path.dirname(__file__)
    rel_path = '../root_files'
    filenames = ['tree_B0PiPi_mc.root',
                 'tree_B0sKK_mc.root', 'tree_Bhh_data.root']

    filepaths = [os.path.join(
        current_path, rel_path, file) for file in filenames]

    print(len(filepaths))

    tree = 't_M0pipi;1'
    var = ('M0_Mpipi', 'M0_MKK', 'M0_MKpi', 'M0_MpiK', 'M0_p',
           'M0_pt', 'M0_eta', 'h1_thetaC0', 'h1_thetaC1', 'h1_thetaC2')

    v1, v2 = array_generator(filepaths, tree, var)

    # v = loadmass(file, tree)
    # v1, v2 = loadmass_uproot(file1, file2, tree, var)

    t2 = time.time()

    print(f'Tempo totale = {t2-t1}')
