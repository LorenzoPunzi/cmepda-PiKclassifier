"""
Module containing functions that import the dataset from the TTrees and store
the variables of interest in a numpy array for ML treatment
"""

import os
import time
import ROOT
import uproot
import numpy as np
from utilities.merge_variables import mergevar
from utilities.utils import default_rootpaths


def loadvars(file_pi, file_k, tree, vars, flag_column=True, flatten1d = True):
    """
    Returns numpy arrays containing the requested variables, stored originally
    in MC root files. The LAST column of the output array contains a flag (0
    for Pions, 1 for Kaons)

    Parameters
    ----------
    file_pi: string
        Path of MC file of B0->PiPi process
    file_k: string
        Path of MC file of B0s->KK process
    tree: string
        Tree containing the dataset (is the same for both the files)
    *vars: string
        Tuple containing names of the variables requested
    """

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

    if flag_column:
        list_pi.append(flag_pi)
        list_k.append(flag_k)

    v_pi, v_k = np.stack(list_pi, axis=1), np.stack(list_k, axis=1)

    if (len(vars) == 1 and flatten1d == True): 
        v_pi, v_k = v_pi.flatten(), v_k.flatten() # Otherwise becomes a 1D column vector when 1D
    

    return v_pi, v_k


def include_merged_variables(filepaths, tree, initial_arrays, new_variables):
    """
    """

    n_old_vars = len(initial_arrays[2][0, :])
    new_arrays = []

    l0_pi = len(initial_arrays[0][:, 0])
    l0_k = len(initial_arrays[1][:, 0])
    if not l0_pi == l0_k:
        length = min(l0_pi, l0_k)
    else:
        length = l0_pi

    list_pi, list_k, list_data = [], [], []

    for newvars in new_variables:
        merged_arrays, m, KS_stats = mergevar(
            filepaths, tree, newvars, savefig=False, savetxt=False)
        list_pi.append(merged_arrays[0][:length])
        list_k.append(merged_arrays[1][:length])
        list_data.append(merged_arrays[2])

    for col in range(n_old_vars):
        list_pi.append(initial_arrays[0][:length, col])
        list_k.append(initial_arrays[1][:length, col])
        list_data.append(initial_arrays[2][:, col])

    # Appending also the flag column to mc arrays
    list_pi.append(initial_arrays[0][:length, -1])
    list_k.append(initial_arrays[1][:length, -1])

    new_arrays = [np.stack(list_pi, axis=1), np.stack(list_k, axis=1),
                  np.stack(list_data, axis=1)]

    return new_arrays


def array_generator(filepaths, tree, vars, Ntrain=100000, Ndata=15000,
                    for_training=True, for_testing=True, new_variables=[]):
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
    if (for_training and for_testing):
        filepath_pi, filepath_k, filepath_data = filepaths
    elif (len(filepaths) == 2 and for_training):
        filepath_pi, filepath_k = filepaths
    elif (len(filepaths) == 1 and for_testing):
        filepath_data = filepaths[0]
    else:
        print('Errore nell istanziamento di array_generator()')
        pass

    train_array, test_array = np.zeros(len(vars)), np.zeros(len(vars))

    '''
    if (mixing and len(newvars == 0)):
        newvars = ['M0_MKpi_M0_MpiK', 'M0_MKK_M0_p', 'M0_Mpipi_M0_p',
                   'M0_MKK_M0_MpiK', 'M0_Mpipi_M0_MKpi']
    '''

    if not new_variables:
        if for_training:
            v_mc_pi, v_mc_k = loadvars(filepath_pi, filepath_k, tree, vars)
            train_array = np.concatenate(
                (v_mc_pi[:int(Ntrain/2), :], v_mc_k[:int(Ntrain/2), :]), axis=0)
            np.random.shuffle(train_array)
        if for_testing:
            v_data, _ = loadvars(filepath_data, filepath_data, tree, vars,
                                 flag_column=False)  # Non mette la flag alla fine perché sono dati
            test_array = v_data[:Ndata, :]

    # If a mixing is requested, both the training and the testing arrays are
    # modified, with obviously the same mixing
    elif new_variables and len(filepaths) == 3:
        v1, v2 = loadvars(filepath_pi, filepath_k, tree, vars)
        v_data, _ = loadvars(filepath_data, filepath_data,
                             tree, vars, flag_column=False)
        initial_arrays = [v1, v2, v_data]
        [v_mc_pi, v_mc_k, v_data_new] = include_merged_variables(
            filepaths, tree, initial_arrays, new_variables)
        train_array = np.concatenate(
            (v_mc_pi[:int(Ntrain/2), :], v_mc_k[:int(Ntrain/2), :]), axis=0)
        test_array = v_data_new[:Ndata, :]
    else:
        pass

    return train_array, test_array




if __name__ == '__main__':
    t1 = time.time()

    filepaths = default_rootpaths()

    file_pi, file_k, file_data = filepaths

    combinations = [('M0_MKpi', 'M0_MpiK'), ('M0_MKK', 'M0_p'), ('M0_Mpipi', 'M0_p'),
                    ('M0_MKK', 'M0_MpiK'), ('M0_Mpipi', 'M0_MKpi')]

    tree = 'tree;1'
    vars = ('M0_Mpipi', 'M0_MKK', 'M0_MKpi', 'M0_MpiK', 'M0_p', 'M0_pt',
            'M0_eta', 'h1_thetaC0', 'h1_thetaC1', 'h1_thetaC2', 'h2_thetaC0',
            'h2_thetaC1', 'h2_thetaC2')

    v_train, v_test = array_generator(filepaths, tree, vars,
                                      Ntrain=560000,
                                      new_variables=[])

    np.savetxt('txt/train_array.txt', v_train)
    np.savetxt('txt/data_array.txt', v_test)


    t2 = time.time()

    print(f'Tempo totale = {t2-t1}')
